import os
import json
import logging
from typing import List, Dict, Any
from .models import Message

logger = logging.getLogger(__name__)

class ChatHistoryManager:
    """Manages chat history persistence for repositories."""
    
    def __init__(self, repos_base_path: str = "/app/repos"):
        self.repos_base_path = repos_base_path

    def _get_history_path(self, repo_id: str) -> str:
        return os.path.join(self.repos_base_path, repo_id, "chat_history.json")

    def get_history(self, repo_id: str) -> List[Dict[str, Any]]:
        """Retrieve chat history for a repository."""
        history_path = self._get_history_path(repo_id)
        if not os.path.exists(history_path):
            return []
        
        try:
            with open(history_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load chat history for {repo_id}: {e}")
            return []

    def add_message(self, repo_id: str, message: Message):
        """Add a message to the repository's chat history."""
        history = self.get_history(repo_id)
        
        # Convert Pydantic model to dict if needed
        msg_dict = message.dict() if hasattr(message, 'dict') else message
        
        history.append(msg_dict)
        
        # Save updated history
        self._save_history(repo_id, history)

    def clear_history(self, repo_id: str):
        """Clear chat history for a repository."""
        self._save_history(repo_id, [])

    def _save_history(self, repo_id: str, history: List[Dict[str, Any]]):
        """Save history list to file."""
        history_path = self._get_history_path(repo_id)
        try:
            # Ensure repo dir exists (it should)
            os.makedirs(os.path.dirname(history_path), exist_ok=True)
            
            with open(history_path, 'w') as f:
                json.dump(history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save chat history for {repo_id}: {e}")

# Global instance
chat_history_manager = ChatHistoryManager()
