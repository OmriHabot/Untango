"""
Agent 1: Repo Mapper
Builds a structural map of the repository, identifying entry points and dependencies.
"""
import os
import json
import logging
from typing import Dict, Any, List

from ..models import RepoMap

logger = logging.getLogger(__name__)

IGNORE_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.idea', '.vscode'}
IGNORE_FILES = {'.DS_Store'}

def is_entry_point(filepath: str) -> bool:
    """Check if a file is likely a python entry point."""
    if not filepath.endswith('.py'):
        return False
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if 'if __name__ == "__main__":' in content or "if __name__ == '__main__':" in content:
                return True
    except Exception:
        pass
    return False

def parse_dependencies(root_path: str) -> List[str]:
    """Extract dependencies from common files."""
    dependencies = []
    
    # Check requirements.txt
    req_path = os.path.join(root_path, 'requirements.txt')
    if os.path.exists(req_path):
        try:
            with open(req_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dependencies.append(line)
        except Exception as e:
            logger.warning(f"Failed to parse requirements.txt: {e}")

    # Check pyproject.toml (basic parsing)
    toml_path = os.path.join(root_path, 'pyproject.toml')
    if os.path.exists(toml_path):
        dependencies.append("pyproject.toml found (parsing not fully implemented)")

    return dependencies

def build_directory_tree(root_path: str) -> Dict[str, Any]:
    """Recursively build a directory tree."""
    tree = {}
    for item in os.listdir(root_path):
        if item in IGNORE_DIRS or item in IGNORE_FILES:
            continue
            
        item_path = os.path.join(root_path, item)
        if os.path.isdir(item_path):
            tree[item] = build_directory_tree(item_path)
        else:
            tree[item] = "file"
    return tree

def find_entry_points(root_path: str) -> List[str]:
    """Walk repo to find entry points."""
    entry_points = []
    for root, dirs, files in os.walk(root_path):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file in IGNORE_FILES:
                continue
            
            full_path = os.path.join(root, file)
            if is_entry_point(full_path):
                # Store relative path
                rel_path = os.path.relpath(full_path, root_path)
                entry_points.append(rel_path)
    return entry_points

def map_repo(repo_path: str, repo_name: str) -> RepoMap:
    """Generate a full repository map."""
    logger.info(f"Mapping repository: {repo_name} at {repo_path}")
    
    structure = build_directory_tree(repo_path)
    entry_points = find_entry_points(repo_path)
    dependencies = parse_dependencies(repo_path)
    
    logger.info(f"Repo Map Complete: {len(entry_points)} entry points found.")
    
    return RepoMap(
        repo_name=repo_name,
        root_path=repo_path,
        structure=structure,
        entry_points=entry_points,
        dependencies=dependencies
    )
