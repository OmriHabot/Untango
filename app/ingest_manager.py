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

CACHE_DIR = ".ingest_cache"
IGNORE_DIRS = {
    '.git', '__pycache__', '.venv', 'venv', 'node_modules', '.idea', '.vscode', 'data',
    'tests', 'test', 'docs', 'doc', 'benchmarks', 'examples', 'samples', 'build', 'dist', 'site-packages', 'repos'
}
EXTENSIONS = {'.py', '.md'}

class IngestManager:
    def __init__(self, repo_path: str = ".", repo_id: str = "default", repo_name: str = "Untango"):
        self.repo_path = repo_path
        self.repo_id = repo_id
        self.repo_name = repo_name
        self.cache_file = os.path.join(CACHE_DIR, f"{repo_id}.json")
        os.makedirs(CACHE_DIR, exist_ok=True)
        self.cache: Dict[str, float] = self._load_cache()
        
    def _load_cache(self) -> Dict[str, float]:
        """Load cache from disk (per-repository cache file)."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {self.repo_id}: {e}")
        return {}
        
    def _save_cache(self):
        """Save cache to disk (per-repository cache file)."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache for {self.repo_id}: {e}")

    async def sync_repo(self):
        """
        Synchronize the repository with the vector database.
        Only processes files that have changed since the last sync.
        Uses batch processing for better performance.
        """
        logger.info("Starting smart ingestion sync...")
        
        # 1. Identify files to process
        files_to_process = []
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
                        files_to_process.append((filepath, rel_path, mtime))
                        
                except Exception as e:
                    logger.error(f"Error checking file {rel_path}: {e}")

        # 2. Handle deletions
        changes_detected = 0
        for cached_file in list(self.cache.keys()):
            if cached_file not in current_files:
                logger.info(f"File removed: {cached_file}")
                try:
                    delete_file_chunks(cached_file)
                except Exception as e:
                    logger.warning(f"Failed to delete chunks for {cached_file}: {e}")
                del self.cache[cached_file]
                changes_detected += 1

        # 3. Process updates in batches
        if files_to_process:
            logger.info(f"Found {len(files_to_process)} files to ingest.")
            
            BATCH_SIZE = 20  # Number of files to process concurrently
            
            for i in range(0, len(files_to_process), BATCH_SIZE):
                batch = files_to_process[i:i + BATCH_SIZE]
                logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(files_to_process) + BATCH_SIZE - 1)//BATCH_SIZE} ({len(batch)} files)")
                
                # Process batch concurrently
                tasks = [self._process_file(fp, rp) for fp, rp, _ in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect valid chunks
                all_ids = []
                all_docs = []
                all_metas = []
                processed_files = []
                
                for j, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process {batch[j][1]}: {result}")
                        continue
                        
                    if result:
                        ids, docs, metas = result
                        if ids:
                            all_ids.extend(ids)
                            all_docs.extend(docs)
                            all_metas.extend(metas)
                        processed_files.append(batch[j])

                # Batch insert to ChromaDB
                if all_ids:
                    try:
                        collection = get_collection()
                        # ChromaDB recommends batches < 40k tokens. 
                        # We might need to sub-batch if really huge, but 20 files should be safe.
                        collection.add(
                            ids=all_ids,
                            documents=all_docs,
                            metadatas=all_metas
                        )
                        changes_detected += len(processed_files)
                        
                        # Update cache only after successful insertion
                        for _, rp, mtime in processed_files:
                            self.cache[rp] = mtime
                            
                    except Exception as e:
                        logger.error(f"Failed to insert batch to ChromaDB: {e}")
                else:
                    # Even if no chunks (empty files), update cache
                    for _, rp, mtime in processed_files:
                        self.cache[rp] = mtime
                        changes_detected += 1

        if changes_detected > 0:
            self._save_cache()
            logger.info(f"Smart ingestion complete. {changes_detected} files updated.")
        else:
            logger.debug("No changes detected.")

    async def _process_file(self, filepath: str, rel_path: str):
        """
        Read and chunk a single file. 
        Returns (ids, documents, metadatas) or None.
        """
        try:
            # Run file I/O in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            with open(filepath, 'r', encoding='utf-8') as f:
                code = await loop.run_in_executor(None, f.read)
            
            # Delete old chunks first
            # Note: This is still synchronous and individual. 
            # Optimization: Could batch delete, but delete_file_chunks takes one file.
            # For now, keep it simple.
            delete_file_chunks(rel_path)
            
            # Chunking (CPU bound)
            if rel_path.endswith('.py'):
                # Run CPU-bound chunking in executor
                chunks = await loop.run_in_executor(None, chunk_python_code, code, rel_path, self.repo_name)
            else:
                chunks = [{
                    "id": f"{rel_path}::text::0",
                    "content": code,
                    "metadata": {
                        "filepath": rel_path,
                        "repo_id": self.repo_id,
                        "repo_name": self.repo_name,
                        "chunk_type": "text",
                        "start_line": 1,
                        "end_line": len(code.splitlines()),
                        "imports": ""
                    }
                }]
            
            if not chunks:
                return [], [], []

            ids = [chunk["id"] for chunk in chunks]
            documents = [chunk["content"] for chunk in chunks]
            metadatas = []
            
            for chunk in chunks:
                filtered_metadata = {}
                for key, value in chunk["metadata"].items():
                    if value is not None:
                        filtered_metadata[key] = value
                # Ensure repo_id is always present
                filtered_metadata["repo_id"] = self.repo_id
                metadatas.append(filtered_metadata)
                
            return ids, documents, metadatas
            
        except Exception as e:
            logger.error(f"Error processing {rel_path}: {e}")
            raise e

# Global instance for the default (local) repository
ingest_manager = IngestManager(repo_path=".", repo_id="default", repo_name="Untango")
