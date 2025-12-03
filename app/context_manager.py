"""
Context Manager
Automates the collection of environment and repository context, and performs dependency comparison.
"""
import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .agents.env_scanner import scan_environment
from .agents.repo_mapper import map_repo
from .models import EnvInfo, RepoMap

logger = logging.getLogger(__name__)

@dataclass
class DependencyStatus:
    package: str
    required_version: Optional[str]
    installed_version: Optional[str]
    status: str  # "OK", "MISSING", "MISMATCH", "UNKNOWN"

@dataclass
class ContextReport:
    env_info: EnvInfo
    repo_map: RepoMap
    dependency_analysis: List[DependencyStatus]
    
    def to_string(self) -> str:
        """Convert report to a human/LLM readable string."""
        report = []
        report.append("=== AUTOMATED CONTEXT REPORT ===")
        
        # Environment Section
        report.append("\n[Environment]")
        report.append(f"OS: {self.env_info.os_info}")
        report.append(f"Python: {self.env_info.python_version}")
        report.append(f"GPU: {self.env_info.gpu_info}")
        
        # Repo Section
        report.append("\n[Repository]")
        report.append(f"Name: {self.repo_map.repo_name}")
        report.append(f"Last Updated: {self.repo_map.last_updated}")
        report.append(f"Entry Points: {', '.join(self.repo_map.entry_points[:5])}" + 
                     (f" (+{len(self.repo_map.entry_points)-5} more)" if len(self.repo_map.entry_points) > 5 else ""))
        
        # Dependency Analysis
        report.append("\n[Dependency Check]")
        issues = [d for d in self.dependency_analysis if d.status != "OK"]
        if not issues:
            report.append("All dependencies appear to be satisfied.")
        else:
            for issue in issues:
                req = issue.required_version or "any"
                inst = issue.installed_version or "none"
                report.append(f"- {issue.package}: Status={issue.status} (Required: {req}, Installed: {inst})")
                
        return "\n".join(report)

class ContextManager:
    def __init__(self):
        self._current_report: Optional[ContextReport] = None
        self._current_repo_id: Optional[str] = None

    def _parse_version_constraint(self, dep_string: str) -> tuple[str, Optional[str]]:
        """
        Parse a dependency string like 'numpy>=1.20' into ('numpy', '>=1.20').
        Very basic parsing.
        """
        # Split by common operators
        parts = re.split(r'(==|>=|<=|>|<|~=)', dep_string, 1)
        package = parts[0].strip()
        version = "".join(parts[1:]).strip() if len(parts) > 1 else None
        return package, version

    def _get_installed_version(self, package: str, installed_packages: List[str]) -> Optional[str]:
        """Find installed version of a package."""
        # installed_packages is list of "pkg==ver"
        package = package.lower().replace('-', '_') # Normalize slightly
        for pkg_str in installed_packages:
            p, v = pkg_str.split('==')
            if p.lower().replace('-', '_') == package:
                return v
        return None

    def analyze_dependencies(self, env: EnvInfo, repo: RepoMap) -> List[DependencyStatus]:
        """Compare repo dependencies with installed packages."""
        results = []
        
        for dep in repo.dependencies:
            # Skip non-package lines (like comments or file paths if any slipped through)
            if dep.startswith('#') or ' ' in dep.strip():
                continue
                
            pkg_name, req_version = self._parse_version_constraint(dep)
            installed_ver = self._get_installed_version(pkg_name, env.installed_packages)
            
            status = "OK"
            if not installed_ver:
                status = "MISSING"
            elif req_version:
                # Basic check: if we have a constraint, we just flag it for the LLM to verify
                # strictly implementing semver in python without 'packaging' lib is painful.
                # We'll trust the LLM to understand "Required: >=1.20, Installed: 1.19" is bad.
                # But we can mark it as "CHECK_VERSION" if they differ significantly?
                # For now, let's just say OK if installed, and let the LLM see the versions.
                # Actually, let's try a simple string check if it's exact match '=='
                if req_version.startswith('==') and req_version[2:] != installed_ver:
                    status = "MISMATCH"
            
            results.append(DependencyStatus(
                package=pkg_name,
                required_version=req_version,
                installed_version=installed_ver,
                status=status
            ))
            
        return results

    def initialize_context(self, repo_path: str, repo_name: str, repo_id: str) -> ContextReport:
        """Run full analysis and cache the report."""
        logger.info(f"Initializing context for {repo_name} (ID: {repo_id})...")
        
        # 1. Scan Env
        env = scan_environment()
        
        # 2. Map Repo
        repo = map_repo(repo_path, repo_name)
        
        # 3. Analyze
        analysis = self.analyze_dependencies(env, repo)
        
        self._current_report = ContextReport(
            env_info=env,
            repo_map=repo,
            dependency_analysis=analysis
        )
        self._current_repo_id = repo_id
        
        return self._current_report

    def get_context_report(self) -> Optional[ContextReport]:
        """
        Get the current context report.
        Automatically re-initializes if the active repo has changed or if no report exists.
        """
        from .active_repo_state import active_repo_state
        from .repo_manager import repo_manager

        active_repo_id = active_repo_state.get_active_repo_id()
        
        # If no active repo is set (or default), we might return None or existing report
        # But if we have an active repo ID, we must ensure the report matches it.
        
        if active_repo_id and active_repo_id != "default":
            # Check if we need to re-initialize
            if self._current_repo_id != active_repo_id or self._current_report is None:
                logger.info(f"Context report stale or missing for {active_repo_id}. Re-initializing...")
                
                # Find repo details
                repos = repo_manager.list_repositories()
                repo = next((r for r in repos if r['repo_id'] == active_repo_id), None)
                
                if repo:
                    return self.initialize_context(repo['path'], repo['name'], active_repo_id)
                else:
                    logger.warning(f"Active repo {active_repo_id} not found in manager. Cannot initialize context.")
                    return None
                    
        return self._current_report

# Global instance
context_manager = ContextManager()
