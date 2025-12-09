"""
Shell Execution Tool
Provides sandboxed shell command execution with:
- Allowlist-based command validation
- Automatic virtual environment management
- Timeout enforcement
"""
import os
import sys
import re
import subprocess
import logging
import shutil
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Allowlist of safe command patterns (regex)
ALLOWED_COMMAND_PATTERNS = [
    r'^pytest(\s|$)',
    r'^python\s+-m\s+pytest(\s|$)',
    r'^pip\s+(list|freeze|show|install)(\s|$)',
    r'^git\s+(status|diff|log|branch|remote)(\s|$)',
    r'^(ruff|flake8|mypy|pylint)(\s|$)',
    r'^python\s+[\w\-_./]+\.py(\s|$)',  # python script.py
    r'^ls(\s|$)',
    r'^cat\s+[\w\-_./]+(\s|$)',  # cat file
    r'^head(\s|$)',
    r'^tail(\s|$)',
    r'^grep(\s|$)',
    r'^find(\s|$)',
    r'^wc(\s|$)',
]

# Explicitly blocked patterns (even if they match allowlist)
BLOCKED_PATTERNS = [
    r'rm\s+-rf',
    r'rm\s+-r',
    r'rmdir',
    r'sudo',
    r'chmod',
    r'chown',
    r'curl.*\|.*sh',
    r'wget.*\|.*sh',
    r'eval\s',
    r'\$\(',  # Command substitution
    r'`',     # Backtick command substitution
    r';\s*rm',
    r'&&\s*rm',
    r'\|\s*sh',
    r'\|\s*bash',
]


def is_command_allowed(command: str) -> Tuple[bool, str]:
    """
    Check if a command is allowed based on allowlist and blocklist.
    
    Returns:
        Tuple of (is_allowed, reason)
    """
    command = command.strip()
    
    # Check blocklist first
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return False, f"Command matches blocked pattern: {pattern}"
    
    # Check allowlist
    for pattern in ALLOWED_COMMAND_PATTERNS:
        if re.match(pattern, command, re.IGNORECASE):
            return True, "Command matches allowlist"
    
    return False, "Command does not match any allowed pattern"


def ensure_venv(repo_path: str) -> str:
    """
    Ensure a virtual environment exists for the repository.
    Creates one if it doesn't exist.
    
    Args:
        repo_path: Path to the repository
        
    Returns:
        Path to the venv's bin directory (e.g., /repo/.venv/bin)
    """
    venv_path = os.path.join(repo_path, ".venv")
    venv_bin = os.path.join(venv_path, "bin")
    venv_python = os.path.join(venv_bin, "python")
    
    # Check if venv already exists
    if os.path.exists(venv_python):
        logger.info(f"Using existing venv at {venv_path}")
        return venv_bin
    
    # Also check for common alternative venv locations
    for alt_name in ["venv", "env", ".env"]:
        alt_path = os.path.join(repo_path, alt_name, "bin", "python")
        if os.path.exists(alt_path):
            logger.info(f"Using existing venv at {os.path.dirname(os.path.dirname(alt_path))}")
            return os.path.dirname(alt_path)
    
    # Create new venv
    logger.info(f"Creating new virtual environment at {venv_path}")
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", venv_path],
            check=True,
            capture_output=True,
            timeout=60
        )
        logger.info(f"Virtual environment created at {venv_path}")
        
        # Upgrade pip in the new venv
        pip_path = os.path.join(venv_bin, "pip")
        subprocess.run(
            [pip_path, "install", "--upgrade", "pip"],
            capture_output=True,
            timeout=60
        )
        
        return venv_bin
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create venv: {e.stderr.decode() if e.stderr else str(e)}")
        raise RuntimeError(f"Failed to create virtual environment: {e}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Timeout creating virtual environment")


def translate_command_for_venv(command: str, venv_bin: str) -> str:
    """
    Translate a command to use the venv's binaries.
    
    Examples:
        - "pytest tests/" -> "/path/.venv/bin/pytest tests/"
        - "pip install requests" -> "/path/.venv/bin/pip install requests"
        - "python script.py" -> "/path/.venv/bin/python script.py"
    """
    command = command.strip()
    
    # Commands that should use venv binaries
    venv_commands = {
        "pytest": "pytest",
        "python": "python",
        "pip": "pip",
        "ruff": "ruff",
        "flake8": "flake8",
        "mypy": "mypy",
        "pylint": "pylint",
    }
    
    # Check if command starts with a venv-able command
    for cmd_name, bin_name in venv_commands.items():
        if command.startswith(cmd_name + " ") or command == cmd_name:
            venv_binary = os.path.join(venv_bin, bin_name)
            if os.path.exists(venv_binary):
                return venv_binary + command[len(cmd_name):]
            else:
                # Binary not in venv, use system one
                logger.warning(f"{bin_name} not found in venv, using system version")
                return command
    
    # Handle "python -m pytest" -> use venv python
    if command.startswith("python -m "):
        venv_python = os.path.join(venv_bin, "python")
        if os.path.exists(venv_python):
            return venv_python + command[6:]  # Replace "python" with venv python
    
    return command


def execute_command(
    command: str,
    repo_path: str,
    timeout: int = 60,
    use_venv: bool = True,
    require_permission: bool = False
) -> Dict[str, Any]:
    """
    Execute a shell command safely within a repository context.
    
    Args:
        command: The command to execute
        repo_path: Path to the repository (used as cwd and for venv)
        timeout: Maximum execution time in seconds
        use_venv: Whether to use/create virtual environment
        require_permission: If True, command requires user approval (for local repos)
        
    Returns:
        Dict with keys: success, stdout, stderr, exit_code, timed_out, command_executed
    """
    result = {
        "success": False,
        "stdout": "",
        "stderr": "",
        "exit_code": -1,
        "timed_out": False,
        "command_executed": command,
        "venv_used": False,
    }
    
    # Validate command against allowlist
    is_allowed, reason = is_command_allowed(command)
    if not is_allowed:
        result["stderr"] = f"Command not allowed: {reason}"
        logger.warning(f"Blocked command: {command} - {reason}")
        return result
    
    # Validate repo_path exists
    if not os.path.isdir(repo_path):
        result["stderr"] = f"Repository path does not exist: {repo_path}"
        return result
    
    # Set up virtual environment if needed
    actual_command = command
    if use_venv:
        try:
            venv_bin = ensure_venv(repo_path)
            actual_command = translate_command_for_venv(command, venv_bin)
            result["venv_used"] = True
            result["command_executed"] = actual_command
            logger.info(f"Translated command: {command} -> {actual_command}")
        except Exception as e:
            logger.error(f"Venv setup failed: {e}")
            result["stderr"] = f"Failed to set up virtual environment: {e}"
            return result
    
    # Execute the command
    try:
        logger.info(f"Executing: {actual_command} in {repo_path}")
        
        # Set up environment with venv activated
        env = os.environ.copy()
        if use_venv:
            venv_bin = ensure_venv(repo_path)
            env["PATH"] = venv_bin + ":" + env.get("PATH", "")
            env["VIRTUAL_ENV"] = os.path.dirname(venv_bin)
        
        proc = subprocess.run(
            actual_command,
            shell=True,
            cwd=repo_path,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        
        result["stdout"] = proc.stdout.decode("utf-8", errors="replace")
        result["stderr"] = proc.stderr.decode("utf-8", errors="replace")
        result["exit_code"] = proc.returncode
        result["success"] = proc.returncode == 0
        
        # Truncate output if too long
        max_output = 10000  # chars
        if len(result["stdout"]) > max_output:
            result["stdout"] = result["stdout"][:max_output] + "\n... (output truncated)"
        if len(result["stderr"]) > max_output:
            result["stderr"] = result["stderr"][:max_output] + "\n... (output truncated)"
            
    except subprocess.TimeoutExpired:
        result["timed_out"] = True
        result["stderr"] = f"Command timed out after {timeout} seconds"
        logger.warning(f"Command timed out: {command}")
    except Exception as e:
        result["stderr"] = f"Execution error: {str(e)}"
        logger.error(f"Command execution failed: {e}")
    
    return result


def install_requirements(repo_path: str, requirements_file: str = "requirements.txt") -> Dict[str, Any]:
    """
    Install dependencies from requirements file into the repo's venv.
    
    Args:
        repo_path: Path to the repository
        requirements_file: Name of the requirements file
        
    Returns:
        Execution result dict
    """
    req_path = os.path.join(repo_path, requirements_file)
    
    if not os.path.exists(req_path):
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Requirements file not found: {requirements_file}",
            "exit_code": -1,
            "timed_out": False,
        }
    
    return execute_command(
        f"pip install -r {requirements_file}",
        repo_path=repo_path,
        timeout=300,  # Longer timeout for pip install
        use_venv=True
    )
