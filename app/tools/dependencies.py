"""
Dependency tools for the Chat Agent.
Allows inspection of installed python packages (site-packages).
"""
import os
import sys
import logging
import importlib.util
from typing import Optional

logger = logging.getLogger(__name__)

def get_package_path(package_name: str) -> str:
    """
    Find the installation path of a package.
    """
    try:
        spec = importlib.util.find_spec(package_name)
        if spec and spec.origin:
            # origin is usually .../site-packages/package/__init__.py
            # we want the directory
            return os.path.dirname(spec.origin)
        
        # Fallback for namespace packages or other edge cases
        if spec and spec.submodule_search_locations:
            return list(spec.submodule_search_locations)[0]
            
        return f"Error: Package '{package_name}' not found."
    except Exception as e:
        return f"Error finding package: {str(e)}"

def list_package_files(package_name: str) -> str:
    """
    List files within an installed package.
    """
    path = get_package_path(package_name)
    if path.startswith("Error"):
        return path
        
    # Use the filesystem tool logic (re-implemented here to avoid circular imports if any)
    try:
        items = []
        for root, dirs, files in os.walk(path):
            rel_root = os.path.relpath(root, path)
            if rel_root == ".":
                rel_root = ""
            
            for f in files:
                if f.endswith(".py"): # Focus on python files for clarity
                    items.append(os.path.join(rel_root, f))
                    
        return f"Files in {package_name} ({path}):\n" + "\n".join(sorted(items)[:100]) # Limit to 100 files
    except Exception as e:
        return f"Error listing package files: {str(e)}"


def read_package_file(package_name: str, filepath: str, max_lines: int = 500) -> str:
    """
    Read a file from an installed package.
    Args:
        package_name: Name of the package (e.g., 'fastapi', 'chromadb')
        filepath: Relative path within the package (e.g., 'api/client.py')
        max_lines: Maximum number of lines to read
    Returns:
        File content or error message
    """
    package_path = get_package_path(package_name)
    if package_path.startswith("Error"):
        return package_path
    
    full_path = os.path.join(package_path, filepath)
    
    try:
        if not os.path.exists(full_path):
            return f"Error: File '{filepath}' does not exist in package '{package_name}'. Full path: {full_path}"
        
        if os.path.isdir(full_path):
            return f"Error: '{filepath}' is a directory, not a file."
        
        content_lines = []
        with open(full_path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    content_lines.append(f"\n... (truncated after {max_lines} lines) ...")
                    break
                content_lines.append(line)
        
        return "".join(content_lines)
    except Exception as e:
        logger.error(f"read_package_file failed for {package_name}/{filepath}: {e}")
        return f"Error reading package file: {str(e)}"
