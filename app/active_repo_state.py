"""
Active Repository State Manager
Manages which repository is currently active for queries.
"""
import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

STATE_FILE = ".active_repo_state.json"


class ActiveRepositoryState:
    """Manages the active repository state."""
    
    def __init__(self):
        self.state_file = STATE_FILE
        self._active_repo_id: Optional[str] = None
        self._ingestion_status: dict = {}  # In-memory status tracking
        self._load_state()
    
    def _load_state(self) -> Optional[str]:
        """Load active repo ID from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    self._active_repo_id = data.get("active_repo_id")
                    return self._active_repo_id
            except Exception as e:
                logger.warning(f"Failed to load active repo state: {e}")
        return "default"  # Default repository
    
    def set_ingestion_status(self, repo_id: str, status: str):
        """Set ingestion status for a repository."""
        self._ingestion_status[repo_id] = status
        logger.info(f"Ingestion status for {repo_id}: {status}")

    def get_ingestion_status(self, repo_id: str) -> str:
        """Get ingestion status for a repository."""
        return self._ingestion_status.get(repo_id, "unknown")
    
    def _save_state(self):
        """Save active repo ID to disk."""
        try:
            with open(self.state_file, 'w') as f:
                json.dump({"active_repo_id": self._active_repo_id}, f)
        except Exception as e:
            logger.warning(f"Failed to save active repo state: {e}")
    
    def get_active_repo_id(self) -> Optional[str]:
        """Get the currently active repository ID."""
        return self._active_repo_id
    
    def set_active_repo_id(self, repo_id: str):
        """Set the active repository ID."""
        self._active_repo_id = repo_id
        self._save_state()
        logger.info(f"Active repository set to: {repo_id}")


# Global instance
active_repo_state = ActiveRepositoryState()
