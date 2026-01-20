# src/citation_ranker.py

from typing import List, Dict, Tuple
from langchain.schema import Document
from collections import defaultdict
import re

class CitationRanker:
    """
    Ranks and prioritizes document citations based on relevance.
    Ensures sources shown to users are most relevant and credible.
    """
    
    def __init__(self):
        """Initialize citation ranker."""
        self.weights = {
            "relevance_score": 0.35,      # How relevant to query
            "recency": 0.25,              # How recent the document
            "meeting_importance": 0.20,   # AC > BASR > DC
            "content_quality": 0.15,      # Document length/depth
            "citation_frequency": 0.05,   # How often mentioned
        }
    
    @staticmethod
    def calculate_meeting_importance(meeting_type: str) -> float:
        """
        Calculate importance score based on meeting type.
        
        Hierarchy:
        - AC (Academic Council): 1.0 (highest authority)
        - BASR (Board): 0.8 (important)
        - DC (Disciplinary): 0.6 (specialized)
        - Other: 0.4
        """
        type_upper = meeting_type.upper() if meeting_type else "OTHER"
        
        importance_map = {
            "AC": 1.0,
            "ACADEMIC COUNCIL": 1.0,
            "BASR": 0.8,
            "BOARD": 0.8,
            "DC": 0.6,
            "DISCIPLINARY": 0.6,
        }
        
        return importance_map.get(type_upper, 0.4)
    
    @staticmethod
    def calculate_recency_score(meeting_date: str) -> float:
        """
        Calculate recency score based on date.
        
        Recent (2024-2025): 1.0
        Recent (2023): 0.85
        Old (2022): 0.60
        Very old (2021 or earlier): 0.30
        """
        if not meeting_date:
            return 0.5  # Neutral if no date
        
        try:
            from datetime import datetime
            doc_date = datetime.strptime(meeting_date, "%Y-%m-%d")
            current_year = datetime.now().year
            years_old = current_year - doc_date.year
            
            if years_old <= 1:
                return 1.0
            elif years_old <= 2:
                return 0.85
            elif years_old <= 3:
                return 0.60
            else:
                return 0.30
        except:
            return 0.5
    
    @staticmethod
    def calculate_content_quality(doc_content: str) -> float:
        """
        Calculate content quality based on length and detail.
        
        Scoring:
        - Very detailed (1000+ words): 1.0
        - Detailed (500-999 words): 0.85
        - Moderate (200-499 words): 0.70
        - Brief (<200 words): 0.50
        """
        word_count = len(doc_content.split())
        
        if word_count >= 1000:
            return 1.0
        elif word_count >= 500:
            return 0.85
        elif word_count >= 200:
            return 0.70
        else:
            return 0.50
    
    @staticmethod
    def extract_source_file(metadata: dict) -> str:
        """
        Extract and format source file name for display.
        """
        file_name = metadata.get("meeting_file", "Unknown Source")
        
        # Clean up file name (remove .md extension, etc)
        file_name = file_name.replace(".md", "").replace(".txt", "")
        
        return file_name
    
    def calculate_citation_score(
        self,
        doc: Document,
        relevance_score: float,
        query_terms: List[str] = None,
        mention_count: int = 1
    ) -> float:
        """
        Calculate overall citation quality score.
        
        Args:
            doc: Document to score
            relevance_score: How relevant to query (0-1)
            query_terms: Terms from the query (for frequency)
            mention_count: How many times document is used
        
        Returns:
            Composite score (0-1)
        """
        # Extract individual scores
        recency = self.calculate_recency_score(
            doc.metadata.get("meeting_date", "")
        )
        
        meeting_importance = self.calculate_meeting_importance(
            doc.metadata.get("meeting_type", "")
        )
        
        content_quality = self.calculate_content_quality(
            doc.page_content
        )
        
        # Citation frequency (normalized)
        # Assuming max mentions per source = 5
        citation_frequency = min(mention_count / 5.0, 1.0)
        
        # Weighted combination
        final_score = (
            self.weights["relevance_score"] * relevance_score +
            self.weights["recency"] * recency +
            self.weights["meeting_importance"] * meeting_importance +
            self.weights["content_quality"] * content_quality +
            self.weights["citation_frequency"] * citation_frequency
        )
        
        return final_score
    
    def rank_citations(
        self,
        documents: List[Document],
        relevance_scores: List[float] = None,
        top_k: int = None
    ) -> List[Tuple[Document, float, Dict]]:
        """
        Rank citations for display to user.
        
        Args:
            documents: Documents to rank
            relevance_scores: Relevance scores (if available)
            top_k: Return only top-k citations
        
        Returns:
            List of (Document, score, metadata_dict) tuples
        """
        if not documents:
            return []
        
        if relevance_scores is None:
            relevance_scores = [0.8] * len(documents)
        
        # Count mention frequency
        mention_counts = defaultdict(int)
        for doc in documents:
            source_file = self.extract_source_file(doc.metadata)
            mention_counts[source_file] += 1
        
        # Calculate scores for all documents
        scored_citations = []
        
        for i, doc in enumerate(documents):
            relevance = relevance_scores[i] if i < len(relevance_scores) else 0.8
            source_file = self.extract_source_file(doc.metadata)
            mention_count = mention_counts[source_file]
            
            citation_score = self.calculate_citation_score(
                doc,
                relevance,
                mention_count=mention_count
            )
            
            metadata_dict = {
                "file": source_file,
                "type": doc.metadata.get("meeting_type", "Unknown"),
                "date": doc.metadata.get("meeting_date", "N/A"),
                "citation_score": citation_score,
                "mention_count": mention_count,
                "recency": self.calculate_recency_score(
                    doc.metadata.get("meeting_date", "")
                ),
                "importance": self.calculate_meeting_importance(
                    doc.metadata.get("meeting_type", "")
                ),
            }
            
            scored_citations.append((doc, citation_score, metadata_dict))
        
        # Sort by score (descending)
        scored_citations.sort(key=lambda x: x[1], reverse=True)
        
        # Remove duplicates (keep first occurrence of each source)
        seen_sources = set()
        unique_citations = []
        
        for doc, score, metadata in scored_citations:
            source_file = metadata["file"]
            if source_file not in seen_sources:
                unique_citations.append((doc, score, metadata))
                seen_sources.add(source_file)
        
        # Return top-k if specified
        if top_k:
            return unique_citations[:top_k]
        
        return unique_citations
    
    def format_citation_for_display(
        self,
        doc: Document,
        score: float,
        metadata_dict: Dict,
        include_score: bool = False
    ) -> str:
        """
        Format citation for user display.
        
        Returns:
            Formatted citation string
        """
        file_name = metadata_dict["file"]
        meeting_type = metadata_dict["type"]
        meeting_date = metadata_dict["date"]
        
        # Build citation
        parts = [file_name]
        
        if meeting_type:
            parts.append(f"({meeting_type})")
        
        if meeting_date and meeting_date != "N/A":
            parts.append(f"[{meeting_date}]")
        
        citation = " ".join(parts)
        
        if include_score:
            citation += f" - Relevance: {score:.0%}"
        
        return citation
    
    def get_citations_for_answer(
        self,
        documents: List[Document],
        query: str = "",
        relevance_scores: List[float] = None,
        max_citations: int = 5,
        include_scores: bool = False
    ) -> List[str]:
        """
        Get formatted citations ready for display in answer.
        
        Args:
            documents: Source documents
            query: User query (for context)
            relevance_scores: Relevance scores from ranking
            max_citations: Maximum citations to show
            include_scores: Include relevance scores in display
        
        Returns:
            List of formatted citation strings
        """
        ranked = self.rank_citations(
            documents,
            relevance_scores,
            top_k=max_citations
        )
        
        formatted_citations = []
        
        for i, (doc, score, metadata) in enumerate(ranked, 1):
            citation = self.format_citation_for_display(
                doc,
                score,
                metadata,
                include_score=include_scores
            )
            formatted_citations.append(citation)
        
        return formatted_citations
    
    def get_citation_ranking_info(
        self,
        documents: List[Document],
        relevance_scores: List[float] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Get detailed ranking information for debugging.
        
        Returns:
            List of dicts with ranking details
        """
        ranked = self.rank_citations(
            documents,
            relevance_scores,
            top_k=top_k
        )
        
        ranking_info = []
        
        for rank, (doc, score, metadata) in enumerate(ranked, 1):
            ranking_info.append({
                "rank": rank,
                "file": metadata["file"],
                "type": metadata["type"],
                "date": metadata["date"],
                "score": round(score, 3),
                "relevance": metadata["recency"],
                "importance": metadata["importance"],
                "mentions": metadata["mention_count"],
            })
        
        return ranking_info
    
    def update_weights(self, new_weights: dict):
        """
        Update citation ranking weights.
        
        Example:
            ranker.update_weights({
                "relevance_score": 0.40,
                "meeting_importance": 0.30,
            })
        """
        self.weights.update(new_weights)
        
        # Normalize weights
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v/total for k, v in self.weights.items()}


# Test cases
if __name__ == "__main__":
    from langchain.schema import Document
    
    ranker = CitationRanker()
    
    # Sample documents
    docs = [
        Document(
            page_content="AC meeting discussed comprehensive budget allocation across all departments and research initiatives with detailed financial planning...",
            metadata={
                "meeting_type": "AC",
                "meeting_file": "AC_Meeting_1_2024",
                "meeting_date": "2024-12-15"
            }
        ),
        Document(
            page_content="BASR meeting covered student discipline cases.",
            metadata={
                "meeting_type": "BASR",
                "meeting_file": "BASR_Meeting_5_2023",
                "meeting_date": "2023-11-20"
            }
        ),
        Document(
            page_content="AC meeting approved new academic program in computer science with extensive discussion on curriculum design and implementation timeline...",
            metadata={
                "meeting_type": "AC",
                "meeting_file": "AC_Meeting_15_2024",
                "meeting_date": "2024-10-10"
            }
        ),
    ]
    
    print("Citation Ranking Demo\n")
    print("=" * 60)
    
    # Get citation ranking info
    ranking_info = ranker.get_citation_ranking_info(docs, top_k=3)
    
    print("\nRanked Citations:")
    for info in ranking_info:
        print(f"\n{info['rank']}. {info['file']}")
        print(f"   Type: {info['type']}")
        print(f"   Date: {info['date']}")
        print(f"   Score: {info['score']}")
        print(f"   Importance: {info['importance']:.0%}")
        print(f"   Recency: {info['recency']:.0%}")
    
    # Get formatted citations for display
    print("\n" + "=" * 60)
    print("\nFormatted for Display:")
    citations = ranker.get_citations_for_answer(
        docs,
        max_citations=3,
        include_scores=True
    )
    
    for i, citation in enumerate(citations, 1):
        print(f"{i}. {citation}")