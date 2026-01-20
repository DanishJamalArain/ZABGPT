# src/result_reranker.py

import re
from typing import List, Tuple
from langchain.schema import Document

class ResultReranker:
    """
    Re-ranks retrieved documents based on relevance to user query.
    Uses multiple scoring strategies for accuracy.
    """
    
    def __init__(self):
        self.weights = {
            "keyword_match": 0.35,        # Exact keyword matches
            "meeting_type_match": 0.25,   # Meeting type relevance
            "recency": 0.15,              # Recent documents prioritized
            "query_length_match": 0.15,   # Content length vs query complexity
            "position_bonus": 0.10,       # Original position bonus
        }
    
    @staticmethod
    def extract_keywords(query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove common words
        stopwords = {
            "what", "when", "where", "why", "how", "is", "are", "in", "the", 
            "a", "an", "and", "or", "but", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "should"
        }
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords
    
    @staticmethod
    def calculate_keyword_match_score(doc_content: str, keywords: List[str]) -> float:
        """
        Score based on keyword matches in document.
        
        Scoring:
        - Exact phrase match: 1.0
        - Single word match: 0.5
        - Multiple matches: cumulative
        """
        if not keywords:
            return 0.0
        
        doc_lower = doc_content.lower()
        score = 0.0
        max_score = len(keywords)
        
        for keyword in keywords:
            # Check for exact phrase match (higher weight)
            if keyword in doc_lower:
                # Count occurrences
                occurrences = doc_lower.count(keyword)
                score += min(occurrences * 0.5, 1.0)  # Cap at 1.0
        
        # Normalize to 0-1 range
        return min(score / max_score, 1.0) if max_score > 0 else 0.0
    
    @staticmethod
    def calculate_meeting_type_score(
        query: str, 
        doc_metadata: dict
    ) -> float:
        """
        Score based on meeting type relevance.
        
        If user asks about AC, prioritize AC documents.
        """
        query_lower = query.lower()
        doc_type = doc_metadata.get("meeting_type", "").upper()
        
        # Extract meeting type from query
        if "ac" in query_lower:
            query_type = "AC"
        elif "basr" in query_lower:
            query_type = "BASR"
        elif "dc" in query_lower and "ac" not in query_lower:
            query_type = "DC"
        else:
            query_type = None
        
        if query_type and doc_type:
            return 1.0 if query_type == doc_type else 0.3
        
        return 0.5  # Neutral if not specified
    
    @staticmethod
    def calculate_recency_score(doc_metadata: dict) -> float:
        """
        Score based on document recency.
        
        Recent meetings are generally more relevant.
        """
        date_str = doc_metadata.get("meeting_date", "")
        
        if not date_str:
            return 0.5  # Neutral if no date
        
        try:
            # Parse date (expecting YYYY-MM-DD format)
            from datetime import datetime
            doc_date = datetime.strptime(date_str, "%Y-%m-%d")
            today = datetime.now()
            
            # Calculate days old
            days_old = (today - doc_date).days
            
            # Scoring: newer is better
            # 0 days old = 1.0
            # 365 days old = 0.5
            # 730+ days old = 0.2
            if days_old <= 365:
                return 1.0 - (days_old / 365) * 0.5
            elif days_old <= 730:
                return 0.5 - (days_old - 365) / 365 * 0.3
            else:
                return 0.2
        except:
            return 0.5
    
    @staticmethod
    def calculate_content_length_score(
        doc_content: str, 
        query: str
    ) -> float:
        """
        Score based on content relevance to query complexity.
        
        More complex queries should be answered by longer documents.
        """
        query_words = len(query.split())
        doc_words = len(doc_content.split())
        
        # Ideal: longer documents for complex queries
        if query_words < 5:
            # Simple query - medium-length docs are good
            ideal_length = 200
        elif query_words < 15:
            # Complex query - longer docs preferred
            ideal_length = 500
        else:
            # Very complex - very long docs preferred
            ideal_length = 800
        
        # Calculate score (peak at ideal length, decrease as you move away)
        ratio = doc_words / ideal_length if ideal_length > 0 else 0
        
        if ratio < 0.5:
            return ratio / 0.5 * 0.5 + 0.5  # 0.5-1.0 range
        elif ratio < 1.5:
            return 1.0
        else:
            return max(0, 1.0 - (ratio - 1.5) / 3)
    
    @staticmethod
    def calculate_position_bonus(position: int) -> float:
        """
        Give slight bonus to earlier results.
        
        Original ranking had some merit.
        """
        # First 3 results get bonus
        if position == 0:
            return 1.0
        elif position == 1:
            return 0.95
        elif position == 2:
            return 0.9
        else:
            return 1.0 - (position * 0.02)  # Gradually decrease
    
    def calculate_relevance_score(
        self,
        query: str,
        doc: Document,
        position: int = 0
    ) -> float:
        """
        Calculate overall relevance score for a document.
        
        Combines multiple scoring strategies.
        """
        keywords = self.extract_keywords(query)
        
        # Calculate individual scores
        keyword_score = self.calculate_keyword_match_score(doc.page_content, keywords)
        meeting_type_score = self.calculate_meeting_type_score(query, doc.metadata)
        recency_score = self.calculate_recency_score(doc.metadata)
        content_length_score = self.calculate_content_length_score(doc.page_content, query)
        position_score = self.calculate_position_bonus(position)
        
        # Weighted combination
        final_score = (
            self.weights["keyword_match"] * keyword_score +
            self.weights["meeting_type_match"] * meeting_type_score +
            self.weights["recency"] * recency_score +
            self.weights["query_length_match"] * content_length_score +
            self.weights["position_bonus"] * position_score
        )
        
        return final_score
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Re-rank documents by relevance to query.
        
        Returns:
            List of (Document, score) tuples sorted by score descending
        """
        if not documents:
            return []
        
        # Calculate scores for all documents
        scored_docs = []
        for position, doc in enumerate(documents):
            score = self.calculate_relevance_score(query, doc, position)
            scored_docs.append((doc, score))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k if specified
        if top_k:
            return scored_docs[:top_k]
        
        return scored_docs
    
    def get_reranked_documents_only(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Document]:
        """
        Return only the re-ranked documents (without scores).
        """
        scored = self.rerank_documents(query, documents, top_k)
        return [doc for doc, score in scored]
    
    def get_reranked_with_scores(
        self,
        query: str,
        documents: List[Document],
        top_k: int = None
    ) -> List[Tuple[Document, float]]:
        """
        Return re-ranked documents with scores for debugging.
        """
        return self.rerank_documents(query, documents, top_k)
    
    def update_weights(self, new_weights: dict):
        """
        Update scoring weights.
        
        Example:
            reranker.update_weights({
                "keyword_match": 0.4,
                "meeting_type_match": 0.3,
                "recency": 0.2,
                "query_length_match": 0.1,
                "position_bonus": 0.0,
            })
        """
        self.weights.update(new_weights)
        
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}


# Test cases
if __name__ == "__main__":
    from langchain.schema import Document
    
    reranker = ResultReranker()
    
    # Sample documents
    docs = [
        Document(
            page_content="AC meeting discussed budget allocation for departments...",
            metadata={"meeting_type": "AC", "meeting_date": "2025-01-15"}
        ),
        Document(
            page_content="BASR meeting covered student discipline cases...",
            metadata={"meeting_type": "BASR", "meeting_date": "2024-12-10"}
        ),
        Document(
            page_content="AC meeting approved new academic program...",
            metadata={"meeting_type": "AC", "meeting_date": "2024-11-20"}
        ),
    ]
    
    query = "What was discussed in AC meeting about budget?"
    
    print(f"Query: {query}\n")
    print("Re-ranked Results:")
    
    scored = reranker.get_reranked_with_scores(query, docs)
    for i, (doc, score) in enumerate(scored, 1):
        print(f"{i}. Score: {score:.3f}")
        print(f"   Type: {doc.metadata.get('meeting_type')}")
        print(f"   Date: {doc.metadata.get('meeting_date')}")
        print(f"   Content: {doc.page_content[:60]}...\n")