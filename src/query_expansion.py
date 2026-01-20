# src/query_expansion.py

import re
from typing import List, Set

class QueryExpander:
    """
    Expands user queries to improve RAG retrieval.
    Handles abbreviations, synonyms, and context.
    """
    
    # SZABIST-specific abbreviations and expansions
    ABBREVIATION_MAP = {
        "ac": ["Academic Council", "AC meeting", "Academic meeting"],
        "basr": ["Board", "BASR meeting", "Board meeting"],
        "dc": ["Disciplinary Committee", "DC meeting", "Discipline"],
        "mom": ["Minutes of Meeting", "minutes", "meeting notes"],
        "hec": ["Higher Education Commission", "HEC"],
        "qec": ["Quality Enhancement Cell", "QEC"],
        "sdo": ["Syndicate Development Officer", "SDO"],
        "iqac": ["Internal Quality Assurance Cell", "IQAC"],
    }
    
    # Related terms for better matching
    SYNONYM_MAP = {
        "decision": ["approved", "resolved", "concluded", "finalized", "adopted"],
        "meeting": ["session", "gathering", "discussion", "assembly"],
        "agenda": ["topic", "item", "subject", "matter"],
        "discussion": ["debate", "deliberation", "conversation", "dialogue"],
        "proposal": ["suggestion", "recommendation", "initiative", "plan"],
        "issue": ["problem", "concern", "matter", "challenge"],
        "approval": ["endorsement", "consent", "authorization", "acceptance"],
        "policy": ["procedure", "guidelines", "rules", "regulation"],
        "budget": ["funding", "finances", "allocation", "resources"],
        "academic": ["educational", "scholarly", "curriculum", "teaching"],
        "administrative": ["management", "operational", "governance", "bureaucratic"],
        "student": ["learner", "pupil", "scholar", "participant"],
        "faculty": ["professor", "instructor", "teacher", "staff"],
        "department": ["division", "unit", "section", "faculty"],
        "program": ["course", "curriculum", "degree", "initiative"],
    }
    
    # Context expansion for meeting numbers and dates
    CONTEXT_EXPANSIONS = {
        "1st": ["first", "1", "meeting 1"],
        "2nd": ["second", "2", "meeting 2"],
        "3rd": ["third", "3", "meeting 3"],
        "4th": ["fourth", "4", "meeting 4"],
        "5th": ["fifth", "5", "meeting 5"],
        "10th": ["tenth", "10", "meeting 10"],
        "11th": ["eleventh", "11", "meeting 11"],
        "15th": ["fifteenth", "15", "meeting 15"],
    }
    
    @staticmethod
    def expand_abbreviations(query: str) -> List[str]:
        """
        Expand known abbreviations in the query.
        
        Example:
            "What happened in AC meeting 5?"
            →
            [
                "What happened in Academic Council meeting 5?",
                "What happened in AC meeting meeting 5?",
            ]
        """
        query_lower = query.lower()
        expanded_queries = [query]  # Keep original
        
        for abbrev, expansions in QueryExpander.ABBREVIATION_MAP.items():
            if abbrev in query_lower:
                for expansion in expansions:
                    new_query = re.sub(
                        r'\b' + re.escape(abbrev) + r'\b',
                        expansion,
                        query,
                        flags=re.IGNORECASE
                    )
                    if new_query not in expanded_queries:
                        expanded_queries.append(new_query)
        
        return expanded_queries
    
    @staticmethod
    def expand_synonyms(query: str) -> List[str]:
        """
        Expand queries with related synonyms.
        
        Example:
            "Was the proposal approved?"
            →
            [
                "Was the proposal approved?",
                "Was the proposal endorsed?",
                "Was the proposal accepted?",
            ]
        """
        query_lower = query.lower()
        expanded_queries = [query]  # Keep original
        
        for term, synonyms in QueryExpander.SYNONYM_MAP.items():
            if term in query_lower:
                for synonym in synonyms:
                    new_query = re.sub(
                        r'\b' + re.escape(term) + r'\b',
                        synonym,
                        query,
                        flags=re.IGNORECASE
                    )
                    if new_query not in expanded_queries:
                        expanded_queries.append(new_query)
        
        return expanded_queries
    
    @staticmethod
    def expand_ordinals(query: str) -> List[str]:
        """
        Expand ordinal numbers (1st, 2nd, etc.) with alternatives.
        
        Example:
            "Tell me about the 1st meeting"
            →
            [
                "Tell me about the 1st meeting",
                "Tell me about the first meeting",
                "Tell me about the 1 meeting",
            ]
        """
        expanded_queries = [query]
        
        for ordinal, alternatives in QueryExpander.CONTEXT_EXPANSIONS.items():
            if ordinal in query:
                for alt in alternatives:
                    new_query = query.replace(ordinal, alt)
                    if new_query not in expanded_queries:
                        expanded_queries.append(new_query)
        
        return expanded_queries
    
    @staticmethod
    def add_meeting_context(query: str) -> List[str]:
        """
        Add meeting context to vague queries.
        
        Example:
            "What was item 5?"
            →
            [
                "What was item 5?",
                "What was agenda item 5 in the meeting?",
                "What was topic 5 discussed in the meeting?",
            ]
        """
        query_lower = query.lower()
        expanded_queries = [query]
        
        # If query mentions "item" but not "agenda" or "meeting"
        if "item" in query_lower and "agenda" not in query_lower:
            expanded_queries.append(query.replace("item", "agenda item"))
        
        # If query is vague about what was discussed
        if any(word in query_lower for word in ["it", "that", "this"]) and "meeting" not in query_lower:
            expanded_queries.append(f"{query} in the meeting")
        
        # If asking about numbers without context
        if re.search(r'\b\d+\b', query) and "meeting" not in query_lower and "agenda" not in query_lower:
            expanded_queries.append(f"{query} meeting")
        
        return expanded_queries
    
    @staticmethod
    def get_all_expansions(query: str) -> List[str]:
        """
        Get all possible query expansions.
        
        Returns:
            List of expanded query variations (deduplicated)
        """
        expansions: Set[str] = set()
        
        # Original query
        expansions.add(query)
        
        # Expand abbreviations
        abbrev_expanded = QueryExpander.expand_abbreviations(query)
        expansions.update(abbrev_expanded)
        
        # Expand synonyms on original and abbreviation-expanded queries
        for abbrev_query in abbrev_expanded:
            synonym_expanded = QueryExpander.expand_synonyms(abbrev_query)
            expansions.update(synonym_expanded)
        
        # Expand ordinals
        ordinal_expanded = QueryExpander.expand_ordinals(query)
        expansions.update(ordinal_expanded)
        
        # Add meeting context
        context_expanded = QueryExpander.add_meeting_context(query)
        expansions.update(context_expanded)
        
        # Return as sorted list (original first, then expansions)
        result = [query]  # Original first
        result.extend(sorted(set(expansions) - {query}))
        
        return result[:5]  # Return top 5 expansions to avoid too many searches
    
    @staticmethod
    def get_optimal_expansion(query: str) -> str:
        """
        Get the single best expanded version of the query.
        Uses heuristics to pick the most likely to improve retrieval.
        """
        expansions = QueryExpander.get_all_expansions(query)
        
        # Prefer expansions with full terms (not abbreviations)
        for expansion in expansions:
            if "Academic Council" in expansion or "BASR" in expansion:
                return expansion
        
        # Otherwise return first expansion
        return expansions[0] if expansions else query


# Test cases
if __name__ == "__main__":
    expander = QueryExpander()
    
    test_queries = [
        "What happened in AC meeting 1?",
        "Was the proposal approved in BASR?",
        "Tell me about DC meeting",
        "What is item 3?",
        "Agenda for AC 5th meeting",
    ]
    
    for query in test_queries:
        print(f"\nOriginal: {query}")
        expansions = expander.get_all_expansions(query)
        print(f"Expansions:")
        for i, exp in enumerate(expansions, 1):
            print(f"  {i}. {exp}")