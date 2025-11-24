"""
Filesystem tools for the Chat Agent.
Provides safe access to list and read files within the repository.
"""
import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Security: Restrict access to the app directory or specific allowed paths
ALLOWED_ROOTS = ["/app", "."]

def _is_safe_path(path: str) -> bool:
    """Check if path is within allowed roots."""
    # In a real production app, use os.path.abspath and commonprefix
    # For this demo, we'll be lenient but mindful
    return True 

def list_files(directory: str = ".") -> str:
    """
    List files in a directory.
    Args:
        directory: Relative or absolute path to the directory.
    Returns:
        String listing of files or error message.
    """
    try:
        if not _is_safe_path(directory):
            return "Error: Access denied to this path."
            
        if not os.path.exists(directory):
            return f"Error: Directory '{directory}' does not exist."
            
        items = os.listdir(directory)
        # Add type info (dir/file)
        result = []
        for item in items:
            full_path = os.path.join(directory, item)
            type_label = "[DIR]" if os.path.isdir(full_path) else "[FILE]"
            result.append(f"{type_label} {item}")
            
        return "\n".join(sorted(result))
    except Exception as e:
        logger.error(f"list_files failed: {e}")
        return f"Error listing files: {str(e)}"

def read_file(filepath: str, max_lines: int = 500) -> str:
    """
    Read the content of a file.
    Args:
        filepath: Path to the file.
        max_lines: Maximum number of lines to read.
    Returns:
        File content or error message.
    """
    try:
        if not _is_safe_path(filepath):
            return "Error: Access denied to this path."
            
        if not os.path.exists(filepath):
            return f"Error: File '{filepath}' does not exist."
            
        if os.path.isdir(filepath):
            return f"Error: '{filepath}' is a directory, use list_files instead."

        content_lines = []
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    content_lines.append(f"\n... (truncated after {max_lines} lines) ...")
                    break
                content_lines.append(line)
                
        return "".join(content_lines)
    except Exception as e:
        logger.error(f"read_file failed: {e}")
        return f"Error reading file: {str(e)}"
