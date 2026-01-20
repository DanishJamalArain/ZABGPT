# src/memory_manager.py

import os
import json
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

# In-memory store for conversations (for production, use Redis/database)
CONVERSATION_STORE = defaultdict(list)
MAX_HISTORY = 10  # Keep last 10 messages per session


class ConversationMemory:
    """Manages multi-turn conversation history for each session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.max_history = MAX_HISTORY
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """
        Add a message to conversation history.
        
        Args:
            role: "user" or "assistant"
            content: Message content
            metadata: Optional metadata (sources, retrieval_score, etc.)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        CONVERSATION_STORE[self.session_id].append(message)
        
        # Keep only last N messages
        if len(CONVERSATION_STORE[self.session_id]) > self.max_history:
            CONVERSATION_STORE[self.session_id] = CONVERSATION_STORE[self.session_id][-self.max_history:]
    
    def get_history(self) -> List[Dict]:
        """Get full conversation history for this session."""
        return CONVERSATION_STORE[self.session_id]
    
    def get_formatted_history(self) -> str:
        """
        Format conversation history as context for LLM.
        
        Returns:
            Formatted string of recent conversation
        """
        history = self.get_history()
        if not history:
            return ""
        
        formatted = "Recent conversation context:\n"
        for msg in history[-4:]:  # Last 4 messages (2 turns)
            role_label = "User" if msg["role"] == "user" else "Assistant"
            formatted += f"\n{role_label}: {msg['content'][:200]}..."
        
        return formatted
    
    def clear_history(self):
        """Clear conversation history for this session."""
        if self.session_id in CONVERSATION_STORE:
            del CONVERSATION_STORE[self.session_id]
    
    def get_last_user_query(self) -> str:
        """Get the last user query."""
        history = self.get_history()
        for msg in reversed(history):
            if msg["role"] == "user":
                return msg["content"]
        return ""
    
    def get_context_summary(self) -> str:
        """
        Generate a brief summary of conversation for query expansion.
        Helps the LLM understand what the user was asking about before.
        """
        history = self.get_history()
        if len(history) < 2:
            return ""
        
        recent_queries = []
        for msg in history[-5:]:  # Last 5 messages
            if msg["role"] == "user":
                recent_queries.append(msg["content"][:150])
        
        if recent_queries:
            return f"Previous context: {' | '.join(recent_queries[-3:])}"
        return ""


def clean_expired_sessions(max_age_minutes: int = 120):
    """Clean up old sessions (for production, implement proper expiration)."""
    # This is a simplified version
    # In production, use Redis with TTL or database cleanup jobs
    pass