#!/usr/bin/env python3
"""
Untango Local CLI Tool

Upload and sync a local repository to a hosted Untango server.
Supports both one-time upload and continuous file watching.

Usage:
    python untango_local.py /path/to/project --server https://untango.example.com
    python untango_local.py /path/to/project --server https://untango.example.com --watch
"""

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List, Set

try:
    import requests
except ImportError:
    print("Error: 'requests' package is required. Install with: pip install requests")
    sys.exit(1)


# File extensions to include
INCLUDE_EXTENSIONS = {'.py', '.js', '.ts', '.tsx', '.jsx', '.md', '.txt', '.json', '.yaml', '.yml', '.html', '.css', '.ipynb'}

# Directories to skip
# Note: .venv and venv are NOT skipped - we want to include virtual environments
SKIP_DIRS = {'.git', '__pycache__', 'node_modules', '.repos', 'dist', 'build', '.next'}


def find_venv_python(repo_path: str) -> Optional[str]:
    """Find virtual environment Python executable."""
    venv_patterns = [
        os.path.join(repo_path, "venv", "bin", "python"),
        os.path.join(repo_path, ".venv", "bin", "python"),
        os.path.join(repo_path, "env", "bin", "python"),
        os.path.join(repo_path, ".env", "bin", "python"),
    ]
    
    for pattern in venv_patterns:
        if os.path.exists(pattern):
            return pattern
    
    return None


def get_file_hashes(repo_path: str) -> Dict[str, str]:
    """Get MD5 hashes of all relevant files."""
    hashes = {}
    
    for root, dirs, files in os.walk(repo_path):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in INCLUDE_EXTENSIONS:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, repo_path)
                try:
                    with open(filepath, 'rb') as f:
                        hashes[rel_path] = hashlib.md5(f.read()).hexdigest()
                except Exception:
                    pass
    
    return hashes


def get_repo_files(repo_path: str) -> List[str]:
    """Get list of files to include."""
    files = []
    
    for root, dirs, filenames in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        
        for file in filenames:
            ext = os.path.splitext(file)[1].lower()
            if ext in INCLUDE_EXTENSIONS:
                filepath = os.path.join(root, file)
                rel_path = os.path.relpath(filepath, repo_path)
                files.append(rel_path)
    
    return files


def create_zip_bundle(repo_path: str, files: List[str]) -> BytesIO:
    """Create a zip file containing specified files."""
    buffer = BytesIO()
    
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for rel_path in files:
            full_path = os.path.join(repo_path, rel_path)
            if os.path.exists(full_path):
                zf.write(full_path, rel_path)
    
    buffer.seek(0)
    return buffer


def upload_repository(server: str, repo_path: str) -> dict:
    """Upload repository to server."""
    abs_path = os.path.abspath(repo_path)
    repo_name = os.path.basename(abs_path)
    
    print(f"📦 Preparing {repo_name}...")
    
    # Get files and create bundle
    files = get_repo_files(abs_path)
    print(f"   Found {len(files)} files")
    
    venv_python = find_venv_python(abs_path)
    if venv_python:
        print(f"   🐍 Virtual environment detected")
    
    # Create zip bundle
    zip_buffer = create_zip_bundle(abs_path, files)
    
    print(f"📤 Uploading to {server}...")
    
    # Upload to server
    try:
        response = requests.post(
            f"{server.rstrip('/')}/api/ingest-local-upload",
            files={'bundle': ('repo.zip', zip_buffer, 'application/zip')},
            data={
                'repo_name': repo_name,
                'source_path': abs_path,
                'venv_python': venv_python or '',
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"❌ Failed to connect to {server}")
        print("   Make sure the server is running and accessible.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)


def watch_and_sync(server: str, repo_path: str, interval: float = 2.0):
    """Watch for changes and sync incrementally."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        print("Error: 'watchdog' package is required for watch mode.")
        print("Install with: pip install watchdog")
        sys.exit(1)
    
    abs_path = os.path.abspath(repo_path)
    repo_name = os.path.basename(abs_path)
    
    # Initial upload
    result = upload_repository(server, repo_path)
    repo_id = result.get('repo_id')
    print(f"✅ Repository uploaded (ID: {repo_id})")
    
    # Track current hashes
    current_hashes = get_file_hashes(abs_path)
    pending_changes: Set[str] = set()
    last_sync = time.time()
    
    class ChangeHandler(FileSystemEventHandler):
        def on_any_event(self, event):
            if event.is_directory:
                return
            
            # Check if file should be tracked
            src_path = event.src_path
            ext = os.path.splitext(src_path)[1].lower()
            
            if ext not in INCLUDE_EXTENSIONS:
                return
            
            # Skip ignored directories
            for skip_dir in SKIP_DIRS:
                if f"/{skip_dir}/" in src_path or src_path.endswith(f"/{skip_dir}"):
                    return
            
            rel_path = os.path.relpath(src_path, abs_path)
            pending_changes.add(rel_path)
    
    observer = Observer()
    observer.schedule(ChangeHandler(), abs_path, recursive=True)
    observer.start()
    
    print(f"\n👀 Watching {repo_name} for changes...")
    print("   Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(interval)
            
            # Check if there are pending changes to sync
            if pending_changes and (time.time() - last_sync) >= interval:
                changed_files = list(pending_changes)
                pending_changes.clear()
                
                # Filter to actual changes
                new_hashes = get_file_hashes(abs_path)
                actual_changes = []
                
                for rel_path in changed_files:
                    old_hash = current_hashes.get(rel_path)
                    new_hash = new_hashes.get(rel_path)
                    
                    if old_hash != new_hash:
                        actual_changes.append(rel_path)
                
                if actual_changes:
                    print(f"🔄 Syncing {len(actual_changes)} changed file(s)...")
                    
                    # Create bundle of just changed files
                    zip_buffer = create_zip_bundle(abs_path, actual_changes)
                    
                    try:
                        response = requests.post(
                            f"{server.rstrip('/')}/api/sync-repository",
                            files={'bundle': ('changes.zip', zip_buffer, 'application/zip')},
                            data={'repo_id': repo_id},
                            timeout=60
                        )
                        response.raise_for_status()
                        print(f"   ✅ Synced: {', '.join(actual_changes[:3])}{'...' if len(actual_changes) > 3 else ''}")
                    except Exception as e:
                        print(f"   ⚠️ Sync failed: {e}")
                    
                    current_hashes = new_hashes
                    last_sync = time.time()
                    
    except KeyboardInterrupt:
        observer.stop()
        print("\n👋 Stopped watching")
    
    observer.join()


def main():
    parser = argparse.ArgumentParser(
        description="Upload and sync a local repository to Untango server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/project --server https://untango.example.com
  %(prog)s . --server http://localhost:8001 --watch
        """
    )
    
    parser.add_argument('path', help='Path to the local repository')
    parser.add_argument('--server', '-s', required=True, help='Untango server URL')
    parser.add_argument('--watch', '-w', action='store_true', help='Watch for changes and sync continuously')
    
    args = parser.parse_args()
    
    # Validate path
    repo_path = os.path.abspath(args.path)
    if not os.path.isdir(repo_path):
        print(f"❌ Error: '{args.path}' is not a valid directory")
        sys.exit(1)
    
    print(f"🚀 Untango Local Sync")
    print(f"   Repository: {repo_path}")
    print(f"   Server: {args.server}")
    print()
    
    if args.watch:
        watch_and_sync(args.server, repo_path)
    else:
        result = upload_repository(args.server, repo_path)
        print(f"✅ Repository uploaded successfully!")
        print(f"   ID: {result.get('repo_id')}")
        print(f"   Name: {result.get('repo_name')}")


if __name__ == '__main__':
    main()
