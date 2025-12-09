"""
Code Quality Tools
Provides linting and type checking capabilities.
Auto-detects installed linters (ruff, flake8, pylint) and uses the best available.
"""
import os
import shutil
import logging
from typing import Dict, Any, List, Optional

from .shell_execution import execute_command, ensure_venv

logger = logging.getLogger(__name__)


def detect_available_linters(repo_path: str) -> List[str]:
    """
    Detect which linters are available in the environment.
    Checks both venv and system installations.
    
    Returns:
        List of available linter names in preference order
    """
    available = []
    
    # Ensure venv exists
    try:
        venv_bin = ensure_venv(repo_path)
    except Exception:
        venv_bin = None
    
    # Check for linters in order of preference
    linters = ["ruff", "flake8", "pylint"]
    
    for linter in linters:
        # Check venv first
        if venv_bin:
            venv_linter = os.path.join(venv_bin, linter)
            if os.path.exists(venv_linter):
                available.append(linter)
                continue
        
        # Check system installation
        if shutil.which(linter):
            available.append(linter)
    
    return available


def run_linter(
    repo_path: str,
    filepath: str = "",
    linter: Optional[str] = None,
    fix: bool = False
) -> Dict[str, Any]:
    """
    Run a linter on the repository or specific file.
    
    Args:
        repo_path: Path to the repository
        filepath: Specific file to lint (empty = lint all)
        linter: Specific linter to use (None = auto-detect)
        fix: If True and linter supports it, apply fixes
        
    Returns:
        Dict with issues found, linter used, and output
    """
    result = {
        "success": False,
        "linter_used": "",
        "issues": [],
        "issue_count": 0,
        "output": "",
        "fixed_count": 0,
    }
    
    # Auto-detect linter if not specified
    if not linter:
        available = detect_available_linters(repo_path)
        if not available:
            # Try to install ruff
            logger.info("No linter found, attempting to install ruff...")
            install_result = execute_command(
                "pip install ruff",
                repo_path=repo_path,
                timeout=120,
                use_venv=True
            )
            if install_result["success"]:
                linter = "ruff"
            else:
                result["output"] = "No linter available. Install ruff, flake8, or pylint."
                return result
        else:
            linter = available[0]
    
    result["linter_used"] = linter
    
    # Build command based on linter
    target = filepath if filepath else "."
    
    if linter == "ruff":
        cmd = f"ruff check {target}"
        if fix:
            cmd += " --fix"
    elif linter == "flake8":
        cmd = f"flake8 {target}"
        # flake8 doesn't have auto-fix
    elif linter == "pylint":
        cmd = f"pylint {target} --output-format=text"
        # pylint doesn't have auto-fix
    else:
        result["output"] = f"Unknown linter: {linter}"
        return result
    
    # Run linter
    cmd_result = execute_command(
        cmd,
        repo_path=repo_path,
        timeout=120,
        use_venv=True
    )
    
    result["output"] = cmd_result["stdout"] + cmd_result["stderr"]
    
    # Parse issues - format varies by linter
    # Common format: filepath:line:col: message
    import re
    issue_pattern = re.compile(r'^(.+?):(\d+):(\d+)?:?\s*(.+)$', re.MULTILINE)
    
    for match in issue_pattern.finditer(result["output"]):
        result["issues"].append({
            "file": match.group(1),
            "line": int(match.group(2)),
            "column": int(match.group(3)) if match.group(3) else 0,
            "message": match.group(4).strip(),
        })
    
    result["issue_count"] = len(result["issues"])
    
    # For ruff with --fix, count fixed issues
    if fix and linter == "ruff":
        fixed_match = re.search(r'(\d+)\s+fix', result["output"], re.IGNORECASE)
        if fixed_match:
            result["fixed_count"] = int(fixed_match.group(1))
    
    # Success = command ran without error (issues are expected)
    result["success"] = not cmd_result["timed_out"]
    
    return result


def run_type_checker(
    repo_path: str,
    filepath: str = "",
    strict: bool = False
) -> Dict[str, Any]:
    """
    Run mypy type checker on the repository or specific file.
    
    Args:
        repo_path: Path to the repository
        filepath: Specific file to check (empty = check all)
        strict: If True, use strict mode
        
    Returns:
        Dict with type errors and output
    """
    result = {
        "success": False,
        "type_errors": [],
        "error_count": 0,
        "output": "",
    }
    
    # Check if mypy is available
    try:
        venv_bin = ensure_venv(repo_path)
        mypy_path = os.path.join(venv_bin, "mypy")
        if not os.path.exists(mypy_path) and not shutil.which("mypy"):
            # Try to install mypy
            logger.info("mypy not found, attempting to install...")
            install_result = execute_command(
                "pip install mypy",
                repo_path=repo_path,
                timeout=120,
                use_venv=True
            )
            if not install_result["success"]:
                result["output"] = "mypy not available and could not be installed"
                return result
    except Exception as e:
        result["output"] = f"Failed to set up type checker: {e}"
        return result
    
    # Build command
    target = filepath if filepath else "."
    cmd = f"mypy {target}"
    
    if strict:
        cmd += " --strict"
    else:
        # Use reasonable defaults
        cmd += " --ignore-missing-imports"
    
    # Run mypy
    cmd_result = execute_command(
        cmd,
        repo_path=repo_path,
        timeout=180,  # Type checking can be slow
        use_venv=True
    )
    
    result["output"] = cmd_result["stdout"] + cmd_result["stderr"]
    
    # Parse errors
    # Format: filepath:line: error: message
    import re
    error_pattern = re.compile(r'^(.+?):(\d+):\s*(error|warning):\s*(.+)$', re.MULTILINE)
    
    for match in error_pattern.finditer(result["output"]):
        result["type_errors"].append({
            "file": match.group(1),
            "line": int(match.group(2)),
            "severity": match.group(3),
            "message": match.group(4).strip(),
        })
    
    result["error_count"] = len(result["type_errors"])
    result["success"] = not cmd_result["timed_out"]
    
    return result


def get_code_complexity(
    repo_path: str,
    filepath: str = ""
) -> Dict[str, Any]:
    """
    Analyze code complexity using radon if available.
    Falls back to basic metrics if radon not installed.
    
    Args:
        repo_path: Path to the repository
        filepath: Specific file to analyze
        
    Returns:
        Dict with complexity metrics
    """
    result = {
        "success": False,
        "complexity": {},
        "output": "",
    }
    
    target = filepath if filepath else "."
    
    # Try using radon
    cmd_result = execute_command(
        f"radon cc {target} -a -s",
        repo_path=repo_path,
        timeout=60,
        use_venv=True
    )
    
    if "No module named radon" in cmd_result["stderr"]:
        # Radon not installed - provide basic line count instead
        result["output"] = "radon not installed. Install with: pip install radon"
        
        # Basic analysis: just count lines
        try:
            if os.path.isfile(os.path.join(repo_path, target)):
                with open(os.path.join(repo_path, target), 'r') as f:
                    lines = f.readlines()
                result["complexity"]["line_count"] = len(lines)
                result["complexity"]["code_lines"] = len([l for l in lines if l.strip() and not l.strip().startswith('#')])
                result["success"] = True
        except Exception as e:
            result["output"] = f"Failed to analyze: {e}"
        
        return result
    
    result["output"] = cmd_result["stdout"] + cmd_result["stderr"]
    result["success"] = cmd_result["success"]
    
    return result
