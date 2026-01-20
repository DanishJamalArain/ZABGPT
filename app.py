import os
import re
import uuid
import json
from flask import Flask, render_template, request, jsonify, stream_with_context, Response
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.helper import download_hugging_face_embeddings
from src.prompt import system_prompt
from src.memory_manager import ConversationMemory
from src.query_expansion import QueryExpander
from src.result_reranker import ResultReranker
from src.hybrid_search import HybridSearch
from src.citation_ranker import CitationRanker
from src.response_formatter import ResponseFormatter

load_dotenv()

app = Flask(__name__)

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("PINECONE_API_KEY or OPENAI_API_KEY missing")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

embeddings = download_hugging_face_embeddings()
index_name = "newapproach2"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

query_expander = QueryExpander()
result_reranker = ResultReranker()
citation_ranker = CitationRanker()
response_formatter = ResponseFormatter()
hybrid_search = None

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    max_tokens=1500,
    top_p=0.9,
)


def get_metadata_filter_from_query(user_query: str):
    """Build Pinecone metadata filter from query."""
    query_lower = user_query.lower()
    filter_dict = {}

    if "basr" in query_lower:
        filter_dict["meeting_type"] = {"$eq": "BASR"}
    elif "dc" in query_lower and "ac" not in query_lower:
        filter_dict["meeting_type"] = {"$eq": "DC"}
    elif "ac" in query_lower:
        filter_dict["meeting_type"] = {"$eq": "AC"}

    match = re.search(
        r"(?:ac|basr|dc)?\s*meeting\s*(\d{1,3})(?:st|nd|rd|th)?"
        r"|(\d{1,3})(?:st|nd|rd|th)?\s*(?:ac|basr|dc)?\s*meeting",
        query_lower,
    )

    if match:
        number = match.group(1) or match.group(2)
        if number:
            filter_dict["meeting_number"] = {"$eq": number}

    return filter_dict if filter_dict else None


def get_enhanced_system_prompt(memory: ConversationMemory) -> str:
    """Enhance system prompt with conversation context."""
    history_context = memory.get_formatted_history()
    enhanced_prompt = system_prompt
    if history_context:
        enhanced_prompt += f"\n\n{history_context}"
    return enhanced_prompt


def retrieve_with_full_pipeline(user_query: str, metadata_filter=None):
    """Full pipeline: expansion → hybrid → reranking → citation ranking"""
    global hybrid_search
    
    print(f"\n🔍 Processing: {user_query}")
    
    expanded_queries = query_expander.get_all_expansions(user_query)
    
    retriever = docsearch.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 30,
            "lambda_mult": 0.7,
            **({"filter": metadata_filter} if metadata_filter else {}),
        },
    )
    
    semantic_docs = retriever.invoke(user_query)
    seen_content = {doc.page_content[:100] for doc in semantic_docs}
    
    for expanded_query in expanded_queries[1:3]:
        try:
            expanded_docs = retriever.invoke(expanded_query)
            for doc in expanded_docs:
                if doc.page_content[:100] not in seen_content:
                    semantic_docs.append(doc)
                    seen_content.add(doc.page_content[:100])
        except:
            continue
    
    if len(semantic_docs) >= 3:
        if hybrid_search is None:
            hybrid_search = HybridSearch(documents=semantic_docs, semantic_weight=0.6)
        else:
            hybrid_search.documents = semantic_docs
        
        semantic_results = [(doc, 0.8) for doc in semantic_docs]
        hybrid_docs = hybrid_search.hybrid_search(user_query, semantic_results, top_k=len(semantic_docs))
        all_docs = [doc for doc, score in hybrid_docs]
    else:
        all_docs = semantic_docs
    
    reranked_docs = result_reranker.get_reranked_documents_only(user_query, all_docs, top_k=None)
    ranked_citations = citation_ranker.rank_citations(reranked_docs, top_k=None)
    
    return reranked_docs, ranked_citations


def build_rag_chain(user_query: str, memory: ConversationMemory):
    """Build RAG chain with complete stack."""
    metadata_filter = get_metadata_filter_from_query(user_query)
    enhanced_system = get_enhanced_system_prompt(memory)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", enhanced_system),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    retriever = docsearch.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 30,
            "lambda_mult": 0.7,
            **({"filter": metadata_filter} if metadata_filter else {}),
        },
    )
    
    class CompleteRAGChain:
        def __init__(self, chain, metadata_filter):
            self.chain = chain
            self.metadata_filter = metadata_filter
        
        def invoke(self, inputs):
            original_query = inputs["input"]
            docs, citations = retrieve_with_full_pipeline(original_query, self.metadata_filter)
            result = self.chain.invoke(inputs)
            result["context"] = docs
            result["citations"] = citations
            return result
    
    base_chain = create_retrieval_chain(retriever, question_answer_chain)
    complete_chain = CompleteRAGChain(base_chain, metadata_filter)
    
    return complete_chain


def stream_llm_response(user_query: str, documents):
    """Stream LLM response word by word."""
    # Create streaming LLM
    streaming_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        max_tokens=1500,
        top_p=0.9,
        streaming=True,  # ✅ ENABLE STREAMING
    )
    
    # Format documents for context
    context_text = "\n\n".join([doc.page_content for doc in documents[:5]])
    
    # Create prompt
    from langchain_core.messages import HumanMessage, SystemMessage
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {user_query}")
    ]
    
    # Stream the response
    for chunk in streaming_llm.stream(messages):
        if chunk.content:
            yield chunk.content


def query_rag_streaming(user_query: str, session_id: str):
    """Query RAG and return formatted streaming response."""
    memory = ConversationMemory(session_id)
    rag_chain = build_rag_chain(user_query, memory)
    result = rag_chain.invoke({"input": user_query})
    
    ranked_citations = result.get("citations", [])
    documents = result.get("context", [])
    
    formatted_citations = []
    sources_seen = set()
    
    for doc, citation_score, metadata in ranked_citations:
        source_file = metadata["file"]
        
        if source_file not in sources_seen:
            formatted_citations.append({
                "file": source_file,
                "type": metadata["type"],
                "date": metadata["date"],
                "score": citation_score
            })
            sources_seen.add(source_file)
    
    return {
        "documents": documents,
        "citations": formatted_citations,
        "session_id": session_id,
    }


@app.route("/")
def index():
    """Render chat interface."""
    return render_template("chat.html")


@app.route("/new-session", methods=["POST"])
def new_session():
    """Create new session."""
    session_id = str(uuid.uuid4())
    return jsonify({"session_id": session_id})


@app.route("/get-stream", methods=["GET", "POST"])
def chat_stream():
    """Handle streaming chat response."""
    msg = request.form.get("msg", "").strip()
    session_id = request.form.get("session_id", str(uuid.uuid4()))
    
    if not msg:
        return jsonify({"error": "Empty message"}), 400
    
    print(f"\n[Session {session_id}] User: {msg}")
    
    try:
        # Get RAG context and citations
        rag_result = query_rag_streaming(msg, session_id)
        documents = rag_result["documents"]
        citations = rag_result["citations"]
        
        # ✅ STREAMING GENERATOR FUNCTION
        def generate():
            # Send initial metadata
            yield f'data: {{"type": "start", "citations_count": {len(citations)}}}\n\n'
            
            # Stream the LLM response
            full_response = ""
            for chunk in stream_llm_response(msg, documents):
                full_response += chunk
                # Send each chunk to frontend
                yield f'data: {json.dumps({"type": "chunk", "content": chunk})}\n\n'
            
            # Format the complete answer
            formatted_answer = response_formatter.format_answer_with_structure(
                answer=full_response,
                query=msg,
                citations=citations,
                confidence_level="high"
            )
            
            html_answer = response_formatter.convert_to_html(formatted_answer)
            
            # Save to memory
            memory = ConversationMemory(session_id)
            memory.add_message("user", msg, {"sources_used": len(citations)})
            memory.add_message("assistant", html_answer, {"citations": citations})
            
            # Send formatted complete answer
            yield f'data: {json.dumps({"type": "complete", "answer": html_answer, "citations": citations})}\n\n'
        
        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route("/get", methods=["GET", "POST"])
def chat():
    """Handle non-streaming chat (fallback)."""
    msg = request.form.get("msg", "").strip()
    session_id = request.form.get("session_id", str(uuid.uuid4()))
    
    if not msg:
        return jsonify({"error": "Empty message"}), 400
    
    print(f"\n[Session {session_id}] User: {msg}")
    
    try:
        memory = ConversationMemory(session_id)
        rag_chain = build_rag_chain(msg, memory)
        result = rag_chain.invoke({"input": msg})
        
        answer = result.get("answer", "No response generated").strip()
        ranked_citations = result.get("citations", [])
        
        formatted_citations = []
        sources_seen = set()
        
        for doc, citation_score, metadata in ranked_citations:
            source_file = metadata["file"]
            
            if source_file not in sources_seen:
                formatted_citations.append({
                    "file": source_file,
                    "type": metadata["type"],
                    "date": metadata["date"],
                    "score": citation_score
                })
                sources_seen.add(source_file)
        
        # ✅ FORMAT ANSWER USING ResponseFormatter
        formatted_answer = response_formatter.format_answer_with_structure(
            answer=answer,
            query=msg,
            citations=formatted_citations,
            confidence_level="high"
        )
        
        # ✅ CONVERT TO HTML
        html_answer = response_formatter.convert_to_html(formatted_answer)
        
        memory.add_message("user", msg, {"sources_used": len(formatted_citations)})
        memory.add_message("assistant", html_answer, {"citations": formatted_citations})
        
        return jsonify({
            "answer": html_answer,
            "session_id": session_id,
            "citations_count": len(formatted_citations),
            "citations": formatted_citations
        })
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error: {str(e)}"}), 500


@app.route("/history", methods=["GET"])
def get_history():
    """Get conversation history."""
    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    
    memory = ConversationMemory(session_id)
    history = memory.get_history()
    
    return jsonify({
        "session_id": session_id,
        "history": history,
        "message_count": len(history)
    })


@app.route("/clear-session", methods=["POST"])
def clear_session():
    """Clear session history."""
    session_id = request.form.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id required"}), 400
    
    memory = ConversationMemory(session_id)
    memory.clear_history()
    
    return jsonify({"status": "cleared", "session_id": session_id})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)