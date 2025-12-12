"""
Chat functionality module.
Handles chat history persistence.
"""
from .history import ChatHistoryManager, chat_history_manager

__all__ = [
    "ChatHistoryManager",
    "chat_history_manager",
]
