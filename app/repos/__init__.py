"""
Repository management module.
Handles repository cloning, state tracking, and context analysis.
"""
from .manager import RepositoryManager, RepositoryContext, repo_manager, REPOS_BASE_PATH
from .state import ActiveRepositoryState, active_repo_state
from .context import ContextManager, ContextReport, DependencyStatus, context_manager

__all__ = [
    # Manager
    "RepositoryManager",
    "RepositoryContext",
    "repo_manager",
    "REPOS_BASE_PATH",
    # State
    "ActiveRepositoryState",
    "active_repo_state",
    # Context
    "ContextManager",
    "ContextReport",
    "DependencyStatus",
    "context_manager",
]
