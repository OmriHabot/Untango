"""
Tools Module
Provides agent tools for filesystem, dependencies, shell execution, testing, git, and code quality.
"""

from .filesystem import read_file, list_files
from .dependencies import get_package_path, list_package_files, read_package_file
from .shell_execution import execute_command, ensure_venv, is_command_allowed, install_requirements
from .test_runner import discover_tests, run_tests, run_single_test, analyze_test_failure
from .git_tools import get_git_status, get_git_diff, get_git_log, get_current_branch, get_remote_url
from .code_quality import run_linter, run_type_checker, detect_available_linters
from .ast_tools import find_function_usages, get_function_details, get_class_hierarchy, get_file_symbols

__all__ = [
    # Filesystem
    "read_file",
    "list_files",
    
    # Dependencies
    "get_package_path",
    "list_package_files", 
    "read_package_file",
    
    # Shell Execution
    "execute_command",
    "ensure_venv",
    "is_command_allowed",
    "install_requirements",
    
    # Test Runner
    "discover_tests",
    "run_tests",
    "run_single_test",
    "analyze_test_failure",
    
    # Git Tools
    "get_git_status",
    "get_git_diff",
    "get_git_log",
    "get_current_branch",
    "get_remote_url",
    
    # Code Quality
    "run_linter",
    "run_type_checker",
    "detect_available_linters",
    
    # AST Tools
    "find_function_usages",
    "get_function_details",
    "get_class_hierarchy",
    "get_file_symbols",
]
