"""
Test Runner Tool
Provides pytest test discovery and execution capabilities.
Uses shell_execution for sandboxed command running.
"""
import os
import re
import json
import logging
from typing import Dict, Any, List, Optional

from .shell_execution import execute_command, ensure_venv

logger = logging.getLogger(__name__)


def discover_tests(repo_path: str) -> Dict[str, Any]:
    """
    Discover all pytest tests in the repository.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Dict with keys: success, test_files, test_count, output
    """
    result = {
        "success": False,
        "test_files": [],
        "test_functions": [],
        "test_count": 0,
        "output": "",
    }
    
    # Use pytest --collect-only to discover tests
    cmd_result = execute_command(
        "pytest --collect-only -q",
        repo_path=repo_path,
        timeout=60,
        use_venv=True
    )
    
    if cmd_result["timed_out"]:
        result["output"] = "Test discovery timed out"
        return result
    
    # Parse the output to extract test information
    output = cmd_result["stdout"] + cmd_result["stderr"]
    result["output"] = output
    
    # If pytest is not installed, try to install it
    if "No module named pytest" in output or "pytest: not found" in output:
        logger.info("pytest not found, attempting to install...")
        install_result = execute_command(
            "pip install pytest",
            repo_path=repo_path,
            timeout=120,
            use_venv=True
        )
        if install_result["success"]:
            # Retry discovery
            cmd_result = execute_command(
                "pytest --collect-only -q",
                repo_path=repo_path,
                timeout=60,
                use_venv=True
            )
            output = cmd_result["stdout"] + cmd_result["stderr"]
            result["output"] = output
    
    # Parse test files and functions from output
    # pytest --collect-only -q output looks like:
    # tests/test_foo.py::test_bar
    # tests/test_foo.py::TestClass::test_method
    test_pattern = re.compile(r'^([\w\-_./]+\.py)::([\w:]+)$', re.MULTILINE)
    matches = test_pattern.findall(output)
    
    test_files = set()
    test_functions = []
    
    for filepath, test_name in matches:
        test_files.add(filepath)
        test_functions.append({
            "file": filepath,
            "name": test_name,
            "full_name": f"{filepath}::{test_name}"
        })
    
    result["test_files"] = sorted(list(test_files))
    result["test_functions"] = test_functions
    result["test_count"] = len(test_functions)
    result["success"] = len(test_functions) > 0 or cmd_result["success"]
    
    # Also do a simple file scan for common test locations
    if result["test_count"] == 0:
        # Fallback: find test files by pattern
        for root, dirs, files in os.walk(repo_path):
            # Skip venv and cache directories
            dirs[:] = [d for d in dirs if d not in {'.venv', 'venv', '__pycache__', '.git', 'node_modules'}]
            
            for f in files:
                if f.startswith('test_') and f.endswith('.py'):
                    rel_path = os.path.relpath(os.path.join(root, f), repo_path)
                    test_files.add(rel_path)
                elif f.endswith('_test.py'):
                    rel_path = os.path.relpath(os.path.join(root, f), repo_path)
                    test_files.add(rel_path)
        
        result["test_files"] = sorted(list(test_files))
        if test_files:
            result["output"] += f"\n\nFound {len(test_files)} test files via file scan."
            result["success"] = True
    
    return result


def run_tests(
    repo_path: str,
    test_path: str = "",
    verbose: bool = True,
    timeout: int = 300,
    markers: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run pytest tests in the repository.
    
    Args:
        repo_path: Path to the repository
        test_path: Specific test file or pattern (default: run all tests)
        verbose: Include verbose output
        timeout: Maximum execution time in seconds
        markers: Pytest markers to filter tests (e.g., "not slow")
        
    Returns:
        Dict with keys: success, passed, failed, errors, skipped, output, duration
    """
    result = {
        "success": False,
        "passed": 0,
        "failed": 0,
        "errors": 0,
        "skipped": 0,
        "total": 0,
        "output": "",
        "duration_seconds": 0.0,
    }
    
    # Build pytest command
    cmd_parts = ["pytest"]
    
    if verbose:
        cmd_parts.append("-v")
    
    if markers:
        cmd_parts.extend(["-m", markers])
    
    if test_path:
        cmd_parts.append(test_path)
    
    # Add output formatting for easier parsing
    cmd_parts.append("--tb=short")  # Short traceback
    
    command = " ".join(cmd_parts)
    
    # Execute tests
    import time
    start_time = time.time()
    
    cmd_result = execute_command(
        command,
        repo_path=repo_path,
        timeout=timeout,
        use_venv=True
    )
    
    result["duration_seconds"] = round(time.time() - start_time, 2)
    result["output"] = cmd_result["stdout"] + cmd_result["stderr"]
    
    if cmd_result["timed_out"]:
        result["output"] = f"Tests timed out after {timeout} seconds\n" + result["output"]
        return result
    
    # Parse test results from output
    # Look for summary line like: "5 passed, 2 failed, 1 error in 10.5s"
    summary_pattern = re.compile(
        r'(\d+)\s+passed|(\d+)\s+failed|(\d+)\s+error|(\d+)\s+skipped',
        re.IGNORECASE
    )
    
    for match in summary_pattern.finditer(result["output"]):
        groups = match.groups()
        if groups[0]:
            result["passed"] = int(groups[0])
        elif groups[1]:
            result["failed"] = int(groups[1])
        elif groups[2]:
            result["errors"] = int(groups[2])
        elif groups[3]:
            result["skipped"] = int(groups[3])
    
    result["total"] = result["passed"] + result["failed"] + result["errors"] + result["skipped"]
    result["success"] = (result["failed"] == 0 and result["errors"] == 0)
    
    return result


def run_single_test(
    repo_path: str,
    test_name: str,
    timeout: int = 60
) -> Dict[str, Any]:
    """
    Run a single specific test.
    
    Args:
        repo_path: Path to the repository
        test_name: Full test name (e.g., "tests/test_foo.py::test_bar")
        timeout: Maximum execution time
        
    Returns:
        Test execution result
    """
    return run_tests(
        repo_path=repo_path,
        test_path=test_name,
        verbose=True,
        timeout=timeout
    )


def analyze_test_failure(output: str) -> Dict[str, Any]:
    """
    Analyze test failure output to extract useful information.
    
    Args:
        output: Raw pytest output
        
    Returns:
        Dict with failure analysis
    """
    result = {
        "failed_tests": [],
        "error_summary": "",
        "likely_cause": "",
    }
    
    # Find FAILED lines
    failed_pattern = re.compile(r'FAILED\s+([\w\-_./]+::\w+)')
    failed_matches = failed_pattern.findall(output)
    result["failed_tests"] = failed_matches
    
    # Extract assertion errors
    if "AssertionError" in output:
        result["likely_cause"] = "Assertion failed - expected values don't match actual values"
    elif "ModuleNotFoundError" in output or "ImportError" in output:
        result["likely_cause"] = "Missing dependency - a required module is not installed"
    elif "TypeError" in output:
        result["likely_cause"] = "Type error - wrong argument types or count"
    elif "AttributeError" in output:
        result["likely_cause"] = "Attribute error - object doesn't have the expected attribute"
    elif "FileNotFoundError" in output:
        result["likely_cause"] = "File not found - test expects a file that doesn't exist"
    
    # Get error summary (last few lines before the summary)
    lines = output.split('\n')
    error_lines = []
    in_error = False
    for line in lines:
        if 'FAILED' in line or 'ERROR' in line or in_error:
            in_error = True
            error_lines.append(line)
            if len(error_lines) > 20:
                break
    
    result["error_summary"] = '\n'.join(error_lines[-10:])
    
    return result
