"""
MCP Server for Untango Code Assistant
Exposes all code exploration, execution, and analysis tools via Model Context Protocol.
"""
import os
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP

# Import existing tool implementations
from .tools.filesystem import read_file as read_file_impl, list_files as list_files_impl
from .tools.shell_execution import execute_command as shell_execute, ensure_venv
from .tools.test_runner import discover_tests as test_discover, run_tests as test_run
from .tools.git_tools import get_git_status, get_git_diff, get_git_log
from .tools.code_quality import run_linter as quality_lint
from .tools.ast_tools import find_function_usages as ast_find_usages
from .search import perform_hybrid_search
from .active_repo_state import active_repo_state
from .repo_manager import repo_manager

logger = logging.getLogger(__name__)

# Create the MCP server instance
mcp = FastMCP(
    name="Untango Code Assistant",
    json_response=True  # For HTTP transport
)


def _get_active_repo_path() -> str:
    """Get the path of the active repository."""
    repo_id = active_repo_state.get_active_repo_id()
    if repo_id and repo_id != "default":
        return os.path.join(repo_manager.repos_base_path, repo_id)
    return "."


# ============= Tools =============

@mcp.tool()
def rag_search(query: str) -> str:
    """
    Search the codebase using RAG (Retrieval-Augmented Generation).
    Uses hybrid vector + keyword search to find relevant code snippets.
    
    Args:
        query: The search query describing what you're looking for
        
    Returns:
        Relevant code snippets with file paths and line numbers
    """
    try:
        repo_id = active_repo_state.get_active_repo_id()
        results = perform_hybrid_search(query, n_results=5, repo_id=repo_id)
        if not results:
            return "No relevant code found."
        
        context = []
        for r in results:
            meta = r['metadata']
            context.append(
                f"File: {meta.get('filepath')}\n"
                f"Lines: {meta.get('start_line')}-{meta.get('end_line')}\n"
                f"Content:\n{r['content']}\n"
            )
        return "\n---\n".join(context)
    except Exception as e:
        return f"RAG search failed: {e}"


@mcp.tool()
def read_file(filepath: str, max_lines: int = 500) -> str:
    """
    Read the content of a file from the repository.
    
    Args:
        filepath: Relative path to the file from repository root
        max_lines: Maximum number of lines to read (default 500)
        
    Returns:
        The file content or an error message
    """
    try:
        base_path = _get_active_repo_path()
        full_path = os.path.join(base_path, filepath)
        return read_file_impl(full_path, max_lines)
    except Exception as e:
        return f"Error reading file: {e}"


@mcp.tool()
def list_files(directory: str = ".") -> str:
    """
    List all files and subdirectories in a directory.
    
    Args:
        directory: Relative path to directory (default is repository root)
        
    Returns:
        List of files and directories with type indicators
    """
    try:
        base_path = _get_active_repo_path()
        full_path = os.path.join(base_path, directory)
        return list_files_impl(full_path)
    except Exception as e:
        return f"Error listing files: {e}"


# @mcp.tool()
# def execute_command(command: str, timeout: int = 60) -> str:
#     """
#     Suggest a shell command for the user to run manually in their terminal.
#     This tool does NOT execute commands - it returns the command for the user to copy and run.
    
#     Args:
#         command: The command to suggest (e.g., "pip list", "pytest tests/")
#         timeout: Unused, kept for API compatibility
        
#     Returns:
#         Formatted suggested command for the user to run manually
#     """
#     repo_path = _get_active_repo_path()
    
#     # Return a formatted message for the frontend to display
#     output = "📋 **Suggested Command**\n\n"
#     output += "Please run this command in your terminal:\n\n"
#     output += f"```bash\ncd {repo_path} && {command}\n```\n\n"
#     output += "_This command was not executed automatically. Copy and run it in your terminal._"
    
#     return output


@mcp.tool()
def discover_tests() -> str:
    """
    Discover all pytest tests in the repository.
    
    Returns:
        List of test files and test functions found
    """
    try:
        repo_path = _get_active_repo_path()
        result = test_discover(repo_path)
        
        output = "Test Discovery Results:\n"
        output += f"- Test files: {len(result.get('test_files', []))}\n"
        output += f"- Test functions: {result.get('test_count', 0)}\n\n"
        
        if result.get('test_files'):
            output += "Test Files:\n"
            for f in result['test_files'][:20]:
                output += f"  - {f}\n"
            if len(result['test_files']) > 20:
                output += f"  ... and {len(result['test_files']) - 20} more\n"
        
        return output
    except Exception as e:
        return f"Error discovering tests: {e}"


@mcp.tool()
def run_tests(test_path: str = "", verbose: bool = True) -> str:
    """
    Run pytest tests in the repository.
    
    Args:
        test_path: Specific test file or pattern (empty runs all tests)
        verbose: Include verbose output (default True)
        
    Returns:
        Test results with pass/fail counts and output
    """
    try:
        repo_path = _get_active_repo_path()
        result = test_run(repo_path, test_path=test_path, verbose=verbose)
        
        output = "Test Results:\n"
        output += f"- Passed: {result.get('passed', 0)}\n"
        output += f"- Failed: {result.get('failed', 0)}\n"
        output += f"- Errors: {result.get('errors', 0)}\n"
        output += f"- Skipped: {result.get('skipped', 0)}\n"
        output += f"- Duration: {result.get('duration_seconds', 0):.2f}s\n\n"
        output += "--- Output ---\n" + result.get('output', '')
        return output
    except Exception as e:
        return f"Error running tests: {e}"


@mcp.tool()
def git_status() -> str:
    """
    Get the git status of the repository.
    
    Returns:
        Current branch, modified/staged/untracked files
    """
    try:
        repo_path = _get_active_repo_path()
        result = get_git_status(repo_path)
        
        if not result.get('success'):
            return f"Git status failed: {result.get('output', 'Unknown error')}"
        
        output = f"Branch: {result.get('branch', 'unknown')}\n"
        output += f"Status: {'Clean' if result.get('clean') else 'Modified'}\n"
        
        if result.get('modified'):
            output += f"\nModified files ({len(result['modified'])}):\n"
            for f in result['modified'][:10]:
                output += f"  M {f}\n"
        
        if result.get('staged'):
            output += f"\nStaged files ({len(result['staged'])}):\n"
            for f in result['staged'][:10]:
                output += f"  + {f}\n"
        
        if result.get('untracked'):
            output += f"\nUntracked files ({len(result['untracked'])}):\n"
            for f in result['untracked'][:10]:
                output += f"  ? {f}\n"
        
        return output
    except Exception as e:
        return f"Error getting git status: {e}"


@mcp.tool()
def git_diff(filepath: str = "") -> str:
    """
    Get git diff showing changes in the repository.
    
    Args:
        filepath: Specific file to diff (empty shows all changes)
        
    Returns:
        Diff output with statistics
    """
    try:
        repo_path = _get_active_repo_path()
        result = get_git_diff(repo_path, filepath=filepath)
        
        if not result.get('success') and not result.get('diff'):
            return "No changes to show."
        
        output = f"Files changed: {result.get('files_changed', 0)}\n"
        output += f"Insertions: +{result.get('insertions', 0)}\n"
        output += f"Deletions: -{result.get('deletions', 0)}\n\n"
        output += result.get('diff', '')
        return output
    except Exception as e:
        return f"Error getting git diff: {e}"


@mcp.tool()
def git_log(filepath: str = "", max_commits: int = 10) -> str:
    """
    Get recent git commit history.
    
    Args:
        filepath: Specific file (empty shows entire repo history)
        max_commits: Maximum number of commits to return (default 10)
        
    Returns:
        Commit history with hashes, authors, dates, and messages
    """
    try:
        repo_path = _get_active_repo_path()
        result = get_git_log(repo_path, filepath=filepath, max_commits=max_commits)
        
        if not result.get('success'):
            return f"Git log failed: {result.get('output', 'Unknown error')}"
        
        output = f"Recent commits ({len(result.get('commits', []))}):\n\n"
        for commit in result.get('commits', []):
            output += f"{commit.get('short_hash', '')} - {commit.get('message', '')}\n"
            output += f"  Author: {commit.get('author', '')} | {commit.get('date', '')}\n\n"
        
        return output
    except Exception as e:
        return f"Error getting git log: {e}"


@mcp.tool()
def run_linter(filepath: str = "") -> str:
    """
    Run code linter on the repository.
    Auto-detects installed linter (ruff, flake8, or pylint).
    
    Args:
        filepath: Specific file to lint (empty lints all files)
        
    Returns:
        Linting issues found with file, line, and message
    """
    try:
        repo_path = _get_active_repo_path()
        result = quality_lint(repo_path, filepath=filepath)
        
        output = f"Linter: {result.get('linter_used', 'unknown')}\n"
        output += f"Issues found: {result.get('issue_count', 0)}\n\n"
        
        if result.get('issues'):
            for issue in result['issues'][:20]:
                output += f"{issue.get('file')}:{issue.get('line')}: {issue.get('message')}\n"
            if len(result['issues']) > 20:
                output += f"\n... and {len(result['issues']) - 20} more issues\n"
        
        return output
    except Exception as e:
        return f"Error running linter: {e}"


@mcp.tool()
def find_function_usages(function_name: str) -> str:
    """
    Find all places where a function is defined and called in the codebase.
    Uses AST analysis for accurate results.
    
    Args:
        function_name: Name of the function to find usages of
        
    Returns:
        List of definitions and call sites with file and line numbers
    """
    try:
        repo_path = _get_active_repo_path()
        result = ast_find_usages(repo_path, function_name, include_definitions=True)
        
        output = f"Searching for usages of '{function_name}':\n\n"
        
        if result.get('definitions'):
            output += f"Definitions ({len(result['definitions'])}):\n"
            for defn in result['definitions']:
                output += f"  {defn.get('file')}:{defn.get('line')}\n"
            output += "\n"
        
        output += f"Usages ({result.get('usage_count', 0)}):\n"
        for usage in result.get('usages', [])[:30]:
            output += f"  {usage.get('file')}:{usage.get('line')} ({usage.get('type')})\n"
        
        if result.get('usage_count', 0) > 30:
            output += f"\n... and {result['usage_count'] - 30} more usages\n"
        
        return output
    except Exception as e:
        return f"Error finding function usages: {e}"


@mcp.tool()
def get_active_repo_path() -> str:
    """
    Get the absolute filesystem path to the currently active repository.
    
    Returns:
        Absolute path to the repository
    """
    return _get_active_repo_path()


# ============= Resources =============

@mcp.resource("repo://info")
def get_repo_info() -> str:
    """Get information about the currently active repository."""
    from .context_manager import context_manager
    
    report = context_manager.get_context_report()
    if report:
        return report.to_string()
    return "No repository context available. Please ensure a repository is active."


@mcp.resource("repo://structure")
def get_repo_structure() -> str:
    """Get the directory structure of the active repository."""
    repo_path = _get_active_repo_path()
    return list_files(repo_path)


@mcp.resource("repo://dependencies")
def get_repo_dependencies() -> str:
    """Get the dependencies of the active repository."""
    from .context_manager import context_manager
    
    report = context_manager.get_context_report()
    if report and report.dependency_analysis:
        output = "Dependency Analysis:\n"
        for dep in report.dependency_analysis:
            status_icon = "✓" if dep.status == "OK" else "✗" if dep.status == "MISSING" else "!"
            output += f"  {status_icon} {dep.package}: {dep.status}\n"
            if dep.required_version:
                output += f"      Required: {dep.required_version}\n"
            if dep.installed_version:
                output += f"      Installed: {dep.installed_version}\n"
        return output
    return "No dependency information available."


# Export the mcp server instance for mounting
def get_mcp_server() -> FastMCP:
    """Get the MCP server instance for mounting in FastAPI."""
    return mcp
