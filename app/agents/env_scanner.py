"""
Agent 0: Environment Scanner
Scans the local environment to detect OS, Python version, GPU availability, and installed packages.
"""
import platform
import sys
import subprocess
import logging
from typing import List

from ..models import EnvInfo

logger = logging.getLogger(__name__)

def get_os_info() -> str:
    """Get operating system information."""
    return f"{platform.system()} {platform.release()} ({platform.machine()})"

def get_python_version() -> str:
    """Get Python version."""
    return sys.version.split()[0]

def check_cuda_availability() -> tuple[bool, str]:
    """Check if CUDA/GPU is available."""
    cuda_available = False
    gpu_info = "No GPU detected or drivers missing"
    
    # Try checking via nvidia-smi
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            cuda_available = True
            gpu_info = result.stdout.strip()
    except FileNotFoundError:
        pass

    # Fallback: check via torch if available (but we don't want to depend on torch being installed in the agent env if possible, 
    # but since this is a python app, maybe we can try importing it if it exists)
    if not cuda_available:
        try:
            import torch
            if torch.cuda.is_available():
                cuda_available = True
                gpu_info = f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
        except ImportError:
            pass
            
    return cuda_available, gpu_info

def get_installed_packages() -> List[str]:
    """Get list of installed pip packages."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'freeze'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
    except Exception as e:
        logger.error(f"Failed to list packages: {e}")
    return []

def scan_environment() -> EnvInfo:
    """Run full environment scan."""
    logger.info("Starting environment scan...")
    
    os_info = get_os_info()
    python_version = get_python_version()
    cuda_available, gpu_info = check_cuda_availability()
    packages = get_installed_packages()
    
    logger.info(f"Env Scan Complete: OS={os_info}, Python={python_version}, CUDA={cuda_available}")
    
    return EnvInfo(
        os_info=os_info,
        python_version=python_version,
        cuda_available=cuda_available,
        gpu_info=gpu_info,
        installed_packages=packages
    )
