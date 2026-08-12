"""
Conversation Memory and State Management Module.
Maintains session-based chat message histories with pruning and serialization.
"""

from typing import Dict, List, Any
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage


class MemoryManager:
    """Manages chat histories across multiple sessions."""

    def __init__(self, max_history_messages: int = 20):
        self._store: Dict[str, InMemoryChatMessageHistory] = {}
        self.max_history_messages = max_history_messages

    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Retrieves or initializes the chat history for a given session ID.
        Ensures history does not exceed max_history_messages.
        """
        if session_id not in self._store:
            self._store[session_id] = InMemoryChatMessageHistory()
        
        history = self._store[session_id]
        
        # Prune if history is too long (keep latest N messages)
        if len(history.messages) > self.max_history_messages:
            history.messages = history.messages[-self.max_history_messages :]

        return history

    def clear_session(self, session_id: str) -> None:
        """Clears the history of a specific session."""
        if session_id in self._store:
            self._store[session_id].clear()

    def clear_all(self) -> None:
        """Clears all stored conversation sessions."""
        self._store.clear()

    def get_formatted_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        Exports the session history in a format ready for UI display or JSON serialization.
        """
        history = self.get_session_history(session_id)
        formatted: List[Dict[str, str]] = []
        for msg in history.messages:
            if isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
            else:
                formatted.append({"role": msg.type, "content": msg.content})
        return formatted
