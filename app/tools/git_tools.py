"""
Git Tools
Provides git integration for understanding code changes, history, and status.
"""
import os
import re
import subprocess
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def is_git_repo(repo_path: str) -> bool:
    """Check if the path is inside a git repository."""
    git_dir = os.path.join(repo_path, ".git")
    return os.path.isdir(git_dir)


def run_git_command(repo_path: str, args: List[str], timeout: int = 30) -> Dict[str, Any]:
    """
    Run a git command safely.
    
    Args:
        repo_path: Path to the repository
        args: Git command arguments (e.g., ["status", "--porcelain"])
        timeout: Command timeout in seconds
        
    Returns:
        Dict with success, stdout, stderr, exit_code
    """
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "exit_code": -1,
    }
    
    if not is_git_repo(repo_path):
        result["stderr"] = "Not a git repository"
        return result
    
    try:
        proc = subprocess.run(
            ["git"] + args,
            cwd=repo_path,
            capture_output=True,
            timeout=timeout,
        )
        
        result["stdout"] = proc.stdout.decode("utf-8", errors="replace")
        result["stderr"] = proc.stderr.decode("utf-8", errors="replace")
        result["exit_code"] = proc.returncode
        result["success"] = proc.returncode == 0
        
    except subprocess.TimeoutExpired:
        result["stderr"] = f"Git command timed out after {timeout} seconds"
    except Exception as e:
        result["stderr"] = f"Git command failed: {str(e)}"
    
    return result


def get_git_status(repo_path: str) -> Dict[str, Any]:
    """
    Get the current git status of the repository.
    
    Returns:
        Dict with branch, modified, staged, untracked files, and ahead/behind info
    """
    result = {
        "success": False,
        "branch": "",
        "modified": [],
        "staged": [],
        "untracked": [],
        "ahead": 0,
        "behind": 0,
        "clean": False,
        "output": "",
    }
    
    # Get branch name
    branch_result = run_git_command(repo_path, ["branch", "--show-current"])
    if branch_result["success"]:
        result["branch"] = branch_result["stdout"].strip()
    
    # Get porcelain status for easy parsing
    status_result = run_git_command(repo_path, ["status", "--porcelain"])
    if not status_result["success"]:
        result["output"] = status_result["stderr"]
        return result
    
    result["success"] = True
    result["output"] = status_result["stdout"]
    
    # Parse status output
    # Format: XY filename
    # X = index status, Y = worktree status
    for line in status_result["stdout"].strip().split('\n'):
        if not line:
            continue
        
        status_code = line[:2]
        filepath = line[3:]
        
        # Untracked files
        if status_code == "??":
            result["untracked"].append(filepath)
        # Staged files (added, modified, deleted in index)
        elif status_code[0] in "AMDR":
            result["staged"].append(filepath)
        # Modified in worktree
        if status_code[1] in "MD":
            result["modified"].append(filepath)
    
    result["clean"] = (
        len(result["modified"]) == 0 and
        len(result["staged"]) == 0 and
        len(result["untracked"]) == 0
    )
    
    # Get ahead/behind info
    ahead_behind = run_git_command(repo_path, [
        "rev-list", "--left-right", "--count", f"{result['branch']}...origin/{result['branch']}"
    ])
    if ahead_behind["success"]:
        parts = ahead_behind["stdout"].strip().split('\t')
        if len(parts) == 2:
            result["ahead"] = int(parts[0])
            result["behind"] = int(parts[1])
    
    return result


def get_git_diff(
    repo_path: str,
    filepath: str = "",
    staged: bool = False,
    commit: str = ""
) -> Dict[str, Any]:
    """
    Get git diff for the repository or a specific file.
    
    Args:
        repo_path: Path to the repository
        filepath: Specific file to diff (empty = all files)
        staged: If True, show staged changes (--cached)
        commit: Compare against specific commit (default: HEAD)
        
    Returns:
        Dict with diff output and statistics
    """
    result = {
        "success": False,
        "diff": "",
        "files_changed": 0,
        "insertions": 0,
        "deletions": 0,
    }
    
    # Build diff command
    args = ["diff", "--stat"]
    
    if staged:
        args.append("--cached")
    
    if commit:
        args.append(commit)
    
    if filepath:
        args.extend(["--", filepath])
    
    # Get stat first
    stat_result = run_git_command(repo_path, args)
    if stat_result["success"]:
        # Parse stats from last line
        stat_lines = stat_result["stdout"].strip().split('\n')
        if stat_lines:
            last_line = stat_lines[-1]
            # Parse "2 files changed, 10 insertions(+), 5 deletions(-)"
            files_match = re.search(r'(\d+)\s+file', last_line)
            ins_match = re.search(r'(\d+)\s+insertion', last_line)
            del_match = re.search(r'(\d+)\s+deletion', last_line)
            
            if files_match:
                result["files_changed"] = int(files_match.group(1))
            if ins_match:
                result["insertions"] = int(ins_match.group(1))
            if del_match:
                result["deletions"] = int(del_match.group(1))
    
    # Get actual diff content
    args = ["diff"]
    if staged:
        args.append("--cached")
    if commit:
        args.append(commit)
    if filepath:
        args.extend(["--", filepath])
    
    diff_result = run_git_command(repo_path, args, timeout=60)
    if diff_result["success"]:
        result["success"] = True
        result["diff"] = diff_result["stdout"]
        
        # Truncate if too long
        max_length = 15000
        if len(result["diff"]) > max_length:
            result["diff"] = result["diff"][:max_length] + "\n\n... (diff truncated)"
    else:
        result["diff"] = diff_result["stderr"]
    
    return result


def get_git_log(
    repo_path: str,
    filepath: str = "",
    max_commits: int = 10,
    format_type: str = "medium"
) -> Dict[str, Any]:
    """
    Get git log for the repository or a specific file.
    
    Args:
        repo_path: Path to the repository
        filepath: Specific file to get log for
        max_commits: Maximum number of commits to return
        format_type: Log format (short, medium, full)
        
    Returns:
        Dict with commits list
    """
    result = {
        "success": False,
        "commits": [],
        "output": "",
    }
    
    # Build log command with structured output
    args = [
        "log",
        f"-n{max_commits}",
        "--format=%H|%h|%an|%ae|%at|%s",  # hash|short_hash|author|email|timestamp|subject
    ]
    
    if filepath:
        args.extend(["--", filepath])
    
    log_result = run_git_command(repo_path, args, timeout=30)
    
    if not log_result["success"]:
        result["output"] = log_result["stderr"]
        return result
    
    result["success"] = True
    result["output"] = log_result["stdout"]
    
    # Parse commits
    for line in log_result["stdout"].strip().split('\n'):
        if not line:
            continue
        
        parts = line.split('|')
        if len(parts) >= 6:
            try:
                timestamp = int(parts[4])
                date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            except:
                date_str = parts[4]
            
            result["commits"].append({
                "hash": parts[0],
                "short_hash": parts[1],
                "author": parts[2],
                "email": parts[3],
                "date": date_str,
                "message": parts[5],
            })
    
    return result


def get_current_branch(repo_path: str) -> str:
    """Get the current branch name."""
    result = run_git_command(repo_path, ["branch", "--show-current"])
    return result["stdout"].strip() if result["success"] else ""


def get_remote_url(repo_path: str) -> str:
    """Get the origin remote URL."""
    result = run_git_command(repo_path, ["remote", "get-url", "origin"])
    return result["stdout"].strip() if result["success"] else ""
