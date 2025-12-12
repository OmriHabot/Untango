"""
Repository Manager
Handles multiple repository sources (GitHub, local directories).
Manages cloning, dependency parsing, and repository metadata.
"""
import os
import shutil
import hashlib
import ast
import json
import logging
from dataclasses import dataclass
from typing import List, Set, Optional
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)

# Docker volume mount point for repositories
# Docker volume mount point for repositories (or local cache)
REPOS_BASE_PATH = os.path.join(os.getcwd(), ".repos")


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
                cmd = ["git", "clone"]
                if branch:
                    cmd.extend(["-b", branch])
                cmd.extend([url, repo_path])
                
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True
                )
            except subprocess.CalledProcessError as e:
                raise Exception(f"Failed to clone repository: {e.stderr.decode()}")
        
        return repo_path
    
    def validate_local_path(self, path: str) -> str:
        """Validate and return absolute path for local directory.
        
        Handles Docker path translation:
        - User enters: /Users/username/Documents/Project
        - Docker sees:  /host/Documents/Project
        """
        abs_path = os.path.abspath(path)
        
        # Check if path exists directly (running locally)
        if os.path.exists(abs_path) and os.path.isdir(abs_path):
            return abs_path
        
        # Try Docker path translation: /Users/*/Documents/* -> /host/Documents/*
        import re
        docker_path = None
        
        # Match patterns like /Users/username/Documents/... or ~/Documents/...
        match = re.match(r'^(/Users/[^/]+|~)/Documents/(.+)$', path)
        if match:
            docker_path = f"/host/Documents/{match.group(2)}"
        
        # Also try /home/username/Documents/... for Linux
        if not docker_path:
            match = re.match(r'^/home/[^/]+/Documents/(.+)$', path)
            if match:
                docker_path = f"/host/Documents/{match.group(1)}"
        
        if docker_path and os.path.exists(docker_path) and os.path.isdir(docker_path):
            logger.info(f"Translated path {path} -> {docker_path}")
            return docker_path
        
        # Path doesn't exist
        if docker_path:
            raise FileNotFoundError(
                f"Local path does not exist: {abs_path}\n"
                f"(Also tried Docker mount: {docker_path})"
            )
        else:
            raise FileNotFoundError(f"Local path does not exist: {abs_path}")
    
    def find_venv_python(self, repo_path: str) -> Optional[str]:
        """Find virtual environment Python executable in the repo."""
        # Common venv locations
        venv_patterns = [
            os.path.join(repo_path, "venv", "bin", "python"),
            os.path.join(repo_path, ".venv", "bin", "python"),
            os.path.join(repo_path, "env", "bin", "python"),
            os.path.join(repo_path, ".env", "bin", "python"),
        ]
        
        for pattern in venv_patterns:
            if os.path.exists(pattern):
                logger.info(f"Found venv python at: {pattern}")
                return pattern
        
        # Search recursively for **/bin/python (limited depth)
        for root, dirs, files in os.walk(repo_path):
            # Limit depth to 2 levels
            depth = root.replace(repo_path, '').count(os.sep)
            if depth > 2:
                dirs[:] = []
                continue
            
            bin_path = os.path.join(root, "bin", "python")
            if os.path.exists(bin_path):
                logger.info(f"Found venv python at: {bin_path}")
                return bin_path
        
        return None
    
    def get_file_hashes(self, repo_path: str) -> dict:
        """Get MD5 hashes of all relevant files in the repository."""
        hashes = {}
        
        for root, dirs, files in os.walk(repo_path):
            # Skip common ignore directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.repos'}]
            
            for file in files:
                # Only hash code files
                if file.endswith(('.py', '.js', '.ts', '.tsx', '.jsx', '.md', '.txt', '.json', '.yaml', '.yml')):
                    filepath = os.path.join(root, file)
                    rel_path = os.path.relpath(filepath, repo_path)
                    try:
                        with open(filepath, 'rb') as f:
                            hashes[rel_path] = hashlib.md5(f.read()).hexdigest()
                    except Exception as e:
                        logger.debug(f"Failed to hash {filepath}: {e}")
        
        return hashes
    
    def get_sample_files(self, repo_path: str, limit: int = 10) -> List[str]:
        """Get a sample of files from the repository for preview."""
        sample_files = []
        
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.venv', 'venv', 'node_modules'}]
            
            for file in files:
                if file.endswith(('.py', '.js', '.ts', '.tsx', '.md')):
                    rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                    sample_files.append(rel_path)
                    if len(sample_files) >= limit:
                        return sample_files
        
        return sample_files
    
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
            repo_path = self.clone_github_repo(source_location, branch)
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
        
        context = RepositoryContext(
            repo_id=repo_id,
            repo_name=repo_name,
            repo_path=repo_path,
            source_type=source_type,
            source_location=source_location,
            dependencies=dependencies
        )
        
        # Persist metadata
        self.save_metadata(context)
        
        return context
    
    def save_metadata(self, context: RepositoryContext):
        """Save repository metadata to a JSON file."""
        metadata_path = os.path.join(context.repo_path, "repo_info.json")
        try:
            with open(metadata_path, 'w') as f:
                json.dump({
                    "repo_id": context.repo_id,
                    "repo_name": context.repo_name,
                    "source_type": context.source_type,
                    "source_location": context.source_location
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save metadata for {context.repo_id}: {e}")

    def get_metadata(self, repo_path: str) -> dict:
        """Get repository metadata, falling back to git config if needed."""
        metadata_path = os.path.join(repo_path, "repo_info.json")
        
        # Try reading saved metadata
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        
        # Fallback: Try to get name from git config
        repo_name = "Unknown"
        try:
            # Check for .git/config
            git_config = os.path.join(repo_path, ".git", "config")
            if os.path.exists(git_config):
                with open(git_config, 'r') as f:
                    content = f.read()
                    if 'url = ' in content:
                        url = content.split('url = ')[1].split('\n')[0].strip()
                        repo_name = self.get_repository_name(repo_path, url)
        except Exception:
            pass
            
        return {"repo_name": repo_name}

    def list_repositories(self) -> List[dict]:
        """List all available repositories."""
        repos = []
        if not os.path.exists(self.repos_base_path):
            return repos
            
        for repo_id in os.listdir(self.repos_base_path):
            repo_path = os.path.join(self.repos_base_path, repo_id)
            if os.path.isdir(repo_path):
                metadata = self.get_metadata(repo_path)
                name = metadata.get("repo_name", repo_id)
                if name == "Unknown":
                    name = repo_id
                
                # Enhanced metadata
                source_location = metadata.get("source_location")
                source_type = metadata.get("source_type", "local")
                
                logger.info(f"Found repo: {name} (ID: {repo_id}, Source: {source_location})")
                
                repos.append({
                    "repo_id": repo_id,
                    "name": name,
                    "path": repo_path,
                    "source_location": source_location,
                    "source_type": source_type
                })
        
        logger.info(f"Total repositories found: {len(repos)}")
        return repos

    def save_runbook(self, repo_id: str, content: str):
        """Save runbook content for a repository."""
        repo_path = os.path.join(self.repos_base_path, repo_id)
        if not os.path.exists(repo_path):
            logger.warning(f"Cannot save runbook: Repository {repo_id} not found")
            return
            
        runbook_path = os.path.join(repo_path, "runbook.md")
        try:
            with open(runbook_path, 'w') as f:
                f.write(content)
            logger.info(f"Saved runbook for {repo_id}")
        except Exception as e:
            logger.error(f"Failed to save runbook for {repo_id}: {e}")

    def get_runbook(self, repo_id: str) -> Optional[str]:
        """Retrieve runbook content for a repository."""
        repo_path = os.path.join(self.repos_base_path, repo_id)
        runbook_path = os.path.join(repo_path, "runbook.md")
        
        if os.path.exists(runbook_path):
            try:
                with open(runbook_path, 'r') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Failed to read runbook for {repo_id}: {e}")
        
        return None


# Global instance
repo_manager = RepositoryManager()
