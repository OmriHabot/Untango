"""
Ingest Manager
Handles incremental ingestion of the repository into ChromaDB.
Tracks file modification times to only re-ingest changed files.
"""
import os
import json
import logging
import asyncio
from typing import Dict, Set

from .chunker import chunk_python_code
from .database import get_collection, get_collection_name, delete_file_chunks

logger = logging.getLogger(__name__)

CACHE_FILE = ".ingest_cache.json"
IGNORE_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.idea', '.vscode', 'data'}
EXTENSIONS = {'.py', '.md'}

class IngestManager:
    def __init__(self, repo_path: str = "."):
        self.repo_path = repo_path
        self.cache: Dict[str, float] = self._load_cache()
        
    def _load_cache(self) -> Dict[str, float]:
        """Load cache from disk."""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return {}
        
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(CACHE_FILE, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    async def sync_repo(self):
        """
        Synchronize the repository with the vector database.
        Only processes files that have changed since the last sync.
        """
        logger.info("Starting smart ingestion sync...")
        changes_detected = 0
        
        # 1. Walk repo and identify changes
        current_files: Set[str] = set()
        
        for root, dirs, files in os.walk(self.repo_path):
            # Skip ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            
            for file in files:
                if not any(file.endswith(ext) for ext in EXTENSIONS):
                    continue
                    
                filepath = os.path.join(root, file)
                # Normalize path
                rel_path = os.path.relpath(filepath, self.repo_path)
                if rel_path.startswith("./"):
                    rel_path = rel_path[2:]
                    
                current_files.add(rel_path)
                
                try:
                    mtime = os.path.getmtime(filepath)
                    
                    # Check if new or modified
                    if rel_path not in self.cache or self.cache[rel_path] < mtime:
                        logger.info(f"Syncing file: {rel_path}")
                        await self._ingest_file(filepath, rel_path)
                        self.cache[rel_path] = mtime
                        changes_detected += 1
                        
                except Exception as e:
                    logger.error(f"Error checking file {rel_path}: {e}")

        # 2. Handle deletions (files in cache but not on disk)
        # Note: For simplicity, we won't actively delete from DB here to avoid complex logic,
        # but we will remove them from cache. A full reset/re-ingest handles cleanups.
        # Ideally, we should delete chunks for missing files too.
        
        for cached_file in list(self.cache.keys()):
            if cached_file not in current_files:
                logger.info(f"File removed: {cached_file}")
                # Optional: delete from DB
                try:
                    delete_file_chunks(cached_file)
                except Exception as e:
                    logger.warning(f"Failed to delete chunks for {cached_file}: {e}")
                del self.cache[cached_file]
                changes_detected += 1

        if changes_detected > 0:
            self._save_cache()
            logger.info(f"Smart ingestion complete. {changes_detected} files updated.")
        else:
            logger.debug("No changes detected.")

    async def _ingest_file(self, filepath: str, rel_path: str):
        """Ingest a single file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # 1. Delete old chunks
            delete_file_chunks(rel_path)
            
            # 2. Chunk and add new
            # We assume repo_name is "Untango" or derived from path
            repo_name = "Untango" 
            
            if rel_path.endswith('.py'):
                chunks = chunk_python_code(code, rel_path, repo_name)
            else:
                # Simple chunking for non-python files (e.g. Markdown)
                # Treat the whole file as one chunk for now, or split by paragraphs if needed.
                # For simplicity, we'll just index the whole file content if it's not too huge.
                # Ideally we should split by headers for MD.
                chunks = [{
                    "id": f"{rel_path}::text::0",
                    "content": code,
                    "metadata": {
                        "filepath": rel_path,
                        "repo_name": repo_name,
                        "chunk_type": "text",
                        "start_line": 1,
                        "end_line": len(code.splitlines()),
                        "imports": ""
                    }
                }]
            
            if not chunks:
                return

            ids = [chunk["id"] for chunk in chunks]
            documents = [chunk["content"] for chunk in chunks]
            metadatas = []
            for chunk in chunks:
                filtered_metadata = {}
                for key, value in chunk["metadata"].items():
                    if value is not None:
                        filtered_metadata[key] = value
                metadatas.append(filtered_metadata)
            
            collection = get_collection()
            collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            
        except Exception as e:
            logger.error(f"Failed to ingest {rel_path}: {e}")

# Global instance
ingest_manager = IngestManager()
