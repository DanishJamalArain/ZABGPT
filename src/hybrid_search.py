# src/hybrid_search.py

import re
from typing import List, Tuple, Dict
from langchain.schema import Document
from collections import defaultdict

class BM25Search:
    """
    BM25 (Best Matching 25) keyword search implementation.
    Complements semantic search for hybrid retrieval.
    """
    
    def __init__(self, documents: List[Document] = None, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 searcher.
        
        Args:
            documents: List of documents to index
            k1: Saturation parameter (controls term frequency saturation)
            b: Length normalization parameter
        """
        self.k1 = k1
        self.b = b
        self.documents = documents or []
        self.inverted_index = defaultdict(set)
        self.document_freqs = {}
        self.avg_doc_length = 0
        
        if documents:
            self._build_index()
    
    def _build_index(self):
        """Build inverted index from documents."""
        total_length = 0
        
        for doc_idx, doc in enumerate(self.documents):
            tokens = self._tokenize(doc.page_content)
            self.document_freqs[doc_idx] = len(tokens)
            total_length += len(tokens)
            
            # Track which documents contain each term
            for token in set(tokens):
                self.inverted_index[token].add(doc_idx)
        
        self.avg_doc_length = total_length / len(self.documents) if self.documents else 0
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into keywords."""
        # Remove punctuation and convert to lowercase
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Split on whitespace and filter short words
        tokens = [t for t in text.split() if len(t) > 2]
        return tokens
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Search for documents matching query using BM25.
        
        Returns:
            List of (doc_index, score) tuples sorted by score
        """
        query_tokens = self._tokenize(query)
        scores = defaultdict(float)
        
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            
            # Get documents containing this token
            docs_with_token = self.inverted_index[token]
            
            # Calculate IDF (Inverse Document Frequency)
            idf = self._calculate_idf(token)
            
            # Calculate BM25 score contribution
            for doc_idx in docs_with_token:
                term_freq = self._get_term_frequency(doc_idx, token)
                doc_length = self.document_freqs.get(doc_idx, 0)
                
                # BM25 formula
                numerator = term_freq * (self.k1 + 1)
                denominator = term_freq + self.k1 * (
                    1 - self.b + self.b * (doc_length / self.avg_doc_length)
                )
                
                scores[doc_idx] += idf * (numerator / denominator)
        
        # Sort by score and return top-k
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
    
    def _calculate_idf(self, token: str) -> float:
        """Calculate Inverse Document Frequency for a token."""
        if token not in self.inverted_index:
            return 0.0
        
        docs_with_token = len(self.inverted_index[token])
        total_docs = len(self.documents)
        
        # IDF formula: log(N / df)
        idf = (total_docs - docs_with_token + 0.5) / (docs_with_token + 0.5)
        return max(0, idf)
    
    def _get_term_frequency(self, doc_idx: int, token: str) -> int:
        """Get term frequency in a specific document."""
        if doc_idx >= len(self.documents):
            return 0
        
        doc = self.documents[doc_idx]
        tokens = self._tokenize(doc.page_content)
        return tokens.count(token)


class HybridSearch:
    """
    Combines semantic (vector) search with keyword (BM25) search.
    Provides best of both worlds for document retrieval.
    """
    
    def __init__(self, documents: List[Document] = None, semantic_weight: float = 0.6):
        """
        Initialize hybrid search.
        
        Args:
            documents: List of documents for BM25 indexing
            semantic_weight: Weight for semantic search (0.0-1.0)
                            Keyword search weight = 1.0 - semantic_weight
        """
        self.documents = documents or []
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight
        self.bm25 = BM25Search(documents) if documents else None
    
    def update_weights(self, semantic_weight: float):
        """Update search weights."""
        if 0 <= semantic_weight <= 1:
            self.semantic_weight = semantic_weight
            self.keyword_weight = 1.0 - semantic_weight
    
    def hybrid_search(
        self,
        query: str,
        semantic_results: List[Tuple[Document, float]],
        top_k: int = 8
    ) -> List[Tuple[Document, float]]:
        """
        Combine semantic and keyword search results.
        
        Args:
            query: Search query
            semantic_results: Results from semantic (vector) search
            top_k: Number of results to return
        
        Returns:
            Combined and reranked results
        """
        # Get keyword search results
        if not self.bm25:
            # If no BM25 index, just return semantic results
            return [(doc, score) for doc, score in semantic_results][:top_k]
        
        keyword_results = self.bm25.search(query, top_k=top_k * 2)
        
        # Create score dictionary for combining results
        combined_scores = {}
        
        # Add semantic search scores (normalize to 0-1)
        max_semantic_score = max([score for _, score in semantic_results]) if semantic_results else 1.0
        for doc, score in semantic_results:
            doc_id = id(doc)  # Use object id as unique identifier
            normalized_score = score / max_semantic_score if max_semantic_score > 0 else 0
            combined_scores[doc_id] = {
                "doc": doc,
                "semantic_score": normalized_score,
                "keyword_score": 0.0,
                "final_score": 0.0
            }
        
        # Add keyword search scores
        if keyword_results:
            max_keyword_score = max([score for _, score in keyword_results]) if keyword_results else 1.0
            for doc_idx, score in keyword_results:
                if doc_idx < len(self.documents):
                    doc = self.documents[doc_idx]
                    doc_id = id(doc)
                    normalized_score = score / max_keyword_score if max_keyword_score > 0 else 0
                    
                    if doc_id not in combined_scores:
                        combined_scores[doc_id] = {
                            "doc": doc,
                            "semantic_score": 0.0,
                            "keyword_score": normalized_score,
                            "final_score": 0.0
                        }
                    else:
                        combined_scores[doc_id]["keyword_score"] = normalized_score
        
        # Calculate final scores
        for doc_id in combined_scores:
            entry = combined_scores[doc_id]
            entry["final_score"] = (
                (self.semantic_weight * entry["semantic_score"]) +
                (self.keyword_weight * entry["keyword_score"])
            )
        
        # Sort by final score and return top-k
        ranked = sorted(
            combined_scores.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )
        
        return [(entry["doc"], entry["final_score"]) for entry in ranked[:top_k]]
    
    def get_debug_info(
        self,
        query: str,
        semantic_results: List[Tuple[Document, float]],
        top_k: int = 3
    ) -> List[Dict]:
        """
        Get debug information showing how results were scored.
        Useful for understanding and tuning the hybrid search.
        """
        if not self.bm25:
            return []
        
        keyword_results = self.bm25.search(query, top_k=top_k * 2)
        
        # Normalize scores
        max_semantic_score = max([score for _, score in semantic_results]) if semantic_results else 1.0
        max_keyword_score = max([score for _, score in keyword_results]) if keyword_results else 1.0
        
        debug_info = []
        
        for i, (doc, sem_score) in enumerate(semantic_results[:top_k]):
            normalized_sem = sem_score / max_semantic_score if max_semantic_score > 0 else 0
            final = (self.semantic_weight * normalized_sem)
            
            debug_info.append({
                "rank": i + 1,
                "file": doc.metadata.get("meeting_file", "Unknown"),
                "semantic_score": round(normalized_sem, 3),
                "keyword_score": 0.0,
                "final_score": round(final, 3),
                "source": "semantic_only"
            })
        
        return debug_info


class HybridSearchConfig:
    """Configuration presets for hybrid search."""
    
    @staticmethod
    def semantic_focused() -> float:
        """Heavily favor semantic/vector search."""
        return 0.8  # 80% semantic, 20% keyword
    
    @staticmethod
    def balanced() -> float:
        """Balanced between semantic and keyword."""
        return 0.6  # 60% semantic, 40% keyword
    
    @staticmethod
    def keyword_focused() -> float:
        """Heavily favor keyword/BM25 search."""
        return 0.4  # 40% semantic, 60% keyword
    
    @staticmethod
    def keyword_only() -> float:
        """Use only keyword search."""
        return 0.0  # 0% semantic, 100% keyword
    
    @staticmethod
    def semantic_only() -> float:
        """Use only semantic search."""
        return 1.0  # 100% semantic, 0% keyword


# Test cases
if __name__ == "__main__":
    from langchain.schema import Document
    
    # Sample documents
    docs = [
        Document(
            page_content="AC meeting discussed budget allocation for departments and research funding",
            metadata={"meeting_type": "AC", "meeting_file": "AC_1.md"}
        ),
        Document(
            page_content="BASR meeting covered student discipline cases and conduct policies",
            metadata={"meeting_type": "BASR", "meeting_file": "BASR_1.md"}
        ),
        Document(
            page_content="AC meeting approved new academic program in computer science",
            metadata={"meeting_type": "AC", "meeting_file": "AC_2.md"}
        ),
    ]
    
    # Initialize hybrid search
    hybrid = HybridSearch(docs, semantic_weight=0.6)
    
    query = "What was discussed about budget in AC meeting?"
    
    # Simulate semantic results (in real usage, these come from vector search)
    semantic_results = [
        (docs[0], 0.92),
        (docs[2], 0.75),
        (docs[1], 0.60),
    ]
    
    print(f"Query: {query}\n")
    
    # Get hybrid results
    hybrid_results = hybrid.hybrid_search(query, semantic_results)
    
    print("Hybrid Search Results:")
    for i, (doc, score) in enumerate(hybrid_results, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   File: {doc.metadata.get('meeting_file')}")
        print(f"   Content: {doc.page_content[:50]}...\n")