"""
Repository Manager
Handles multiple repository sources (GitHub, local directories).
Manages cloning, dependency parsing, and repository metadata.
"""
import os
import shutil
import hashlib
import ast
import logging
from dataclasses import dataclass
from typing import List, Set, Optional
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)

# Docker volume mount point for repositories
REPOS_BASE_PATH = "/app/repos"


@dataclass
class RepositoryContext:
    """Context information for a repository."""
    repo_id: str  # Unique identifier (hash of source)
    repo_name: str  # Display name
    repo_path: str  # Local filesystem path
    source_type: str  # "github" or "local"
    source_location: str  # Original GitHub URL or local path
    dependencies: List[str]  # Detected dependencies


class RepositoryManager:
    """Manages multiple code repositories."""
    
    def __init__(self, repos_base_path: str = REPOS_BASE_PATH):
        self.repos_base_path = repos_base_path
        os.makedirs(repos_base_path, exist_ok=True)
    
    def generate_repo_id(self, source_location: str) -> str:
        """Generate a unique ID for a repository based on its source."""
        return hashlib.sha256(source_location.encode()).hexdigest()[:16]
    
    def clone_github_repo(self, url: str, branch: str = "main") -> str:
        """
        Clone a GitHub repository.
        Returns the local path where it was cloned.
        """
        repo_id = self.generate_repo_id(url)
        repo_path = os.path.join(self.repos_base_path, repo_id)
        
        # If already cloned, pull latest changes
        if os.path.exists(repo_path):
            logger.info(f"Repository {repo_id} already exists, pulling latest changes")
            try:
                subprocess.run(
                    ["git", "pull"],
                    cwd=repo_path,
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to pull updates: {e}")
        else:
            logger.info(f"Cloning repository from {url}")
            try:
                subprocess.run(
                    ["git", "clone", "-b", branch, url, repo_path],
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError as e:
                raise Exception(f"Failed to clone repository: {e.stderr.decode()}")
        
        return repo_path
    
    def validate_local_path(self, path: str) -> str:
        """Validate and return absolute path for local directory."""
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Local path does not exist: {abs_path}")
        if not os.path.isdir(abs_path):
            raise ValueError(f"Path is not a directory: {abs_path}")
        return abs_path
    
    def parse_requirements_txt(self, repo_path: str) -> List[str]:
        """Parse requirements.txt and return list of dependencies."""
        requirements_path = os.path.join(repo_path, "requirements.txt")
        dependencies = []
        
        if not os.path.exists(requirements_path):
            logger.info(f"No requirements.txt found in {repo_path}")
            return dependencies
        
        try:
            with open(requirements_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip comments and empty lines
                    if line and not line.startswith('#'):
                        # Extract package name (before ==, >=, etc.)
                        pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].strip()
                        dependencies.append(pkg_name)
        except Exception as e:
            logger.error(f"Failed to parse requirements.txt: {e}")
        
        return dependencies
    
    def extract_imports_from_file(self, filepath: str) -> Set[str]:
        """Extract import statements from a Python file."""
        imports = set()
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Get top-level package name
                        pkg = alias.name.split('.')[0]
                        imports.add(pkg)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        pkg = node.module.split('.')[0]
                        imports.add(pkg)
        except Exception as e:
            logger.debug(f"Failed to parse {filepath}: {e}")
        
        return imports
    
    def extract_all_imports(self, repo_path: str) -> Set[str]:
        """Walk repository and extract all imports from Python files."""
        all_imports = set()
        
        for root, dirs, files in os.walk(repo_path):
            # Skip common ignore directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'venv', 'node_modules'}]
            
            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    file_imports = self.extract_imports_from_file(filepath)
                    all_imports.update(file_imports)
        
        # Filter out standard library modules (approximate)
        stdlib_modules = {'os', 'sys', 'ast', 'json', 'logging', 'typing', 'pathlib', 
                         'dataclasses', 'subprocess', 'collections', 'itertools', 'functools',
                         'datetime', 'time', 're', 'math', 'random', 'hashlib', 'asyncio'}
        
        return all_imports - stdlib_modules
    
    def get_repository_name(self, repo_path: str, source_location: str) -> str:
        """Derive repository name from path or URL."""
        if source_location.startswith('http'):
            # GitHub URL - extract repo name
            parts = source_location.rstrip('/').split('/')
            return parts[-1].replace('.git', '')
        else:
            # Local path - use directory name
            return os.path.basename(repo_path)
    
    def create_repository_context(
        self,
        source_type: str,
        source_location: str,
        branch: Optional[str] = None,
        parse_dependencies: bool = True
    ) -> RepositoryContext:
        """Create a repository context from a source."""
        
        # Step 1: Get local path
        if source_type == "github":
            repo_path = self.clone_github_repo(source_location, branch or "main")
        elif source_type == "local":
            repo_path = self.validate_local_path(source_location)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
        
        # Step 2: Generate repo ID and name
        repo_id = self.generate_repo_id(source_location)
        repo_name = self.get_repository_name(repo_path, source_location)
        
        # Step 3: Parse dependencies
        dependencies = []
        if parse_dependencies:
            # From requirements.txt
            req_deps = self.parse_requirements_txt(repo_path)
            # From import statements
            import_deps = self.extract_all_imports(repo_path)
            # Combine and deduplicate
            dependencies = list(set(req_deps) | import_deps)
        
        return RepositoryContext(
            repo_id=repo_id,
            repo_name=repo_name,
            repo_path=repo_path,
            source_type=source_type,
            source_location=source_location,
            dependencies=dependencies
        )
    
    def list_repositories(self) -> List[dict]:
        """List all available repositories."""
        repos = []
        if not os.path.exists(self.repos_base_path):
            return repos
            
        for repo_id in os.listdir(self.repos_base_path):
            repo_path = os.path.join(self.repos_base_path, repo_id)
            if os.path.isdir(repo_path):
                # Try to determine name (fallback to ID)
                # In a real app, we'd persist metadata properly.
                # For now, we'll just return what we have.
                repos.append({
                    "repo_id": repo_id,
                    "path": repo_path
                })
        return repos


# Global instance
repo_manager = RepositoryManager()
