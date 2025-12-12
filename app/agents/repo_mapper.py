"""
Agent 1: Repo Mapper
Builds a structural map of the repository, identifying entry points and dependencies.
"""
import os
import json
import logging
from typing import Dict, Any, List, Optional

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

import ast

def extract_imports_from_file(filepath: str) -> List[str]:
    """Extract top-level imports from a Python file using AST."""
    imports = set()
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            tree = ast.parse(f.read())
            
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Get top-level package name
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception:
        # Ignore parsing errors
        pass
    return list(imports)

def scan_codebase_imports(root_path: str) -> List[str]:
    """Scan all python files in the repo for imports."""
    all_imports = set()
    for root, dirs, files in os.walk(root_path):
        # Skip ignore dirs
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                file_imports = extract_imports_from_file(filepath)
                all_imports.update(file_imports)
                
    return list(all_imports)

def parse_dependencies(root_path: str) -> List[str]:
    """Extract dependencies from requirements.txt and code imports."""
    dependencies = set()
    
    # 1. Check requirements.txt
    req_path = os.path.join(root_path, 'requirements.txt')
    if os.path.exists(req_path):
        try:
            with open(req_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dependencies.add(line)
        except Exception as e:
            logger.warning(f"Failed to parse requirements.txt: {e}")

    # 2. Check pyproject.toml (basic parsing)
    toml_path = os.path.join(root_path, 'pyproject.toml')
    if os.path.exists(toml_path):
        dependencies.add("pyproject.toml found (parsing not fully implemented)")

    # 3. Dynamic Scan
    code_imports = scan_codebase_imports(root_path)
    
    # Filter stdlib (approximate list)
    stdlib = {
        'os', 'sys', 'json', 'logging', 'ast', 'typing', 'math', 'time', 'datetime', 
        're', 'random', 'collections', 'itertools', 'functools', 'pathlib', 'subprocess',
        'platform', 'shutil', 'hashlib', 'unittest', 'copy', 'io', 'contextlib'
    }
    
    for imp in code_imports:
        if imp not in stdlib and not imp.startswith('.'):
            # If it's already in requirements (e.g. "numpy>=1.0"), don't add just "numpy"
            # But checking that is hard without parsing. 
            # For now, we just add it. The ContextManager can handle duplicates or we let the LLM see both.
            # Actually, let's check if the package name exists in any requirement string
            is_covered = any(d.split('==')[0].split('>=')[0].split('<')[0].strip() == imp for d in dependencies)
            if not is_covered:
                dependencies.add(imp)

    return list(dependencies)

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

import subprocess
import datetime

def get_last_updated_date(root_path: str) -> str:
    """Get the last updated date of the repository."""
    # Try git first
    try:
        if os.path.exists(os.path.join(root_path, '.git')):
            result = subprocess.run(
                ['git', 'log', '-1', '--format=%cd', '--date=short'],
                cwd=root_path,
                capture_output=True,
                text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
    except Exception:
        pass
        
    # Fallback to filesystem
    try:
        latest_mtime = 0
        for root, _, files in os.walk(root_path):
            if any(d in root for d in IGNORE_DIRS):
                continue
            for file in files:
                if file in IGNORE_FILES:
                    continue
                try:
                    mtime = os.path.getmtime(os.path.join(root, file))
                    if mtime > latest_mtime:
                        latest_mtime = mtime
                except OSError:
                    pass
        
        if latest_mtime > 0:
            return datetime.datetime.fromtimestamp(latest_mtime).strftime('%Y-%m-%d')
    except Exception:
        pass
        
    return "Unknown"

def detect_repo_name_and_readme(root_path: str) -> tuple[Optional[str], bool]:
    """
    Detect repository name from README or configuration files.
    Returns (detected_name, readme_exists).
    """
    readme_exists = False
    detected_name = None
    
    # Check for README
    readme_candidates = ['README.md', 'README.rst', 'README.txt', 'README']
    readme_path = None
    
    for candidate in readme_candidates:
        # Case insensitive check
        for file in os.listdir(root_path):
            if file.lower() == candidate.lower():
                readme_path = os.path.join(root_path, file)
                readme_exists = True
                break
        if readme_exists:
            break
            
    # Try to extract name from README (First H1)
    if readme_path:
        try:
            with open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('# '):
                        detected_name = line[2:].strip()
                        break
                    if line.startswith('='): # RST style
                        # Previous line was title
                        pass 
        except Exception:
            pass

    # If no name from README, try pyproject.toml
    if not detected_name:
        toml_path = os.path.join(root_path, 'pyproject.toml')
        if os.path.exists(toml_path):
            try:
                with open(toml_path, 'r') as f:
                    for line in f:
                        if line.strip().startswith('name ='):
                            detected_name = line.split('=')[1].strip().strip('"').strip("'")
                            break
            except Exception:
                pass

    # Try package.json
    if not detected_name:
        pkg_path = os.path.join(root_path, 'package.json')
        if os.path.exists(pkg_path):
            try:
                with open(pkg_path, 'r') as f:
                    data = json.load(f)
                    detected_name = data.get('name')
            except Exception:
                pass

    return detected_name, readme_exists

def detect_repo_type(root_path: str) -> str:
    """
    Detect if the repo is a library (pip installable) or an application (git clone & run).
    Returns: 'library', 'application', 'script', or 'unknown'
    """
    # 1. Check for explicit package definition (Strong signal for Library)
    if os.path.exists(os.path.join(root_path, 'setup.py')):
        return "library"
        
    if os.path.exists(os.path.join(root_path, 'pyproject.toml')):
        try:
            with open(os.path.join(root_path, 'pyproject.toml'), 'r') as f:
                content = f.read()
                if '[build-system]' in content or '[tool.poetry]' in content:
                    return "library"
        except Exception:
            pass

    # 2. Check for Application signals
    app_indicators = ['Dockerfile', 'docker-compose.yml', 'manage.py', 'wsgi.py', 'asgi.py']
    for indicator in app_indicators:
        if os.path.exists(os.path.join(root_path, indicator)):
            return "application"
            
    # 3. Check for Node.js App vs Lib
    if os.path.exists(os.path.join(root_path, 'package.json')):
        try:
            with open(os.path.join(root_path, 'package.json'), 'r') as f:
                data = json.load(f)
                if 'scripts' in data and 'start' in data['scripts']:
                    return "application"
                if 'main' in data: # likely a lib if it has main but no start? ambiguous.
                    pass 
        except Exception:
            pass

    # 4. Fallback: Check structure
    # If it has requirements.txt but no setup.py, it's likely an app
    if os.path.exists(os.path.join(root_path, 'requirements.txt')):
        return "application"

    return "unknown"


def map_repo(repo_path: str, repo_name: str) -> RepoMap:
    """Generate a full repository map."""
    logger.info(f"Mapping repository: {repo_name} at {repo_path}")
    
    structure = build_directory_tree(repo_path)
    entry_points = find_entry_points(repo_path)
    dependencies = parse_dependencies(repo_path)
    last_updated = get_last_updated_date(repo_path)
    detected_name, readme_exists = detect_repo_name_and_readme(repo_path)
    repo_type = detect_repo_type(repo_path)
    
    logger.info(f"Repo Map Complete: {len(entry_points)} entry points found. Type: {repo_type}. Last updated: {last_updated}. README: {readme_exists}")
    
    return RepoMap(
        repo_name=repo_name,
        root_path=repo_path,
        structure=structure,
        entry_points=entry_points,
        dependencies=dependencies,
        last_updated=last_updated,
        readme_exists=readme_exists,
        detected_name=detected_name,
        repo_type=repo_type
    )
