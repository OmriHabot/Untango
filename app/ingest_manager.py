"""
Ingest Manager
Handles incremental ingestion of the repository into ChromaDB.
Tracks file modification times to only re-ingest changed files.

Optimized batching based on ChromaDB research:
- ChromaDB default max batch: 41,666
- SQLite rebalancing overhead scales with collection size
- Conservative batch size of 30K prevents exponential slowdown
"""
import os
import json
import logging
import asyncio
import time
from typing import Dict, Set

from .chunker import chunk_python_code
from .database import get_collection, get_collection_name, delete_file_chunks, get_embedding_function

logger = logging.getLogger(__name__)

# Optimized batch sizing for ChromaDB ingestion
CHROMADB_MAX_BATCH = 41_666  # ChromaDB default limit
OPTIMAL_BATCH_SIZE = int(CHROMADB_MAX_BATCH * 0.9)  # ~37,500 with safety margin
CONSERVATIVE_BATCH_SIZE = 30_000  # Proven fastest, avoids SQLite rebalancing overhead
FILE_BATCH_SIZE = 50  # Concurrent file I/O batch size

# Performance monitoring thresholds
MIN_EXPECTED_RATE = 500  # docs/sec - warn if rate drops below this
LOG_BATCH_INTERVAL = 5  # Log performance every N batches

CACHE_DIR = ".ingest_cache"
IGNORE_DIRS = {
    '.git', '__pycache__', '.venv', 'venv', 'node_modules', '.idea', '.vscode', 'data',
    'tests', 'test', 'docs', 'doc', 'benchmarks', 'examples', 'samples', 'build', 'dist', 'site-packages', 'repos'
}
EXTENSIONS = {'.py', '.md', '.ipynb'}

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
            
            # Get embedding function for explicit embedding generation
            embedding_function = get_embedding_function()
            
            # Use conservative batch size to avoid SQLite rebalancing overhead
            # Research shows 30K is optimal, respects ChromaDB limits with margin
            target_batch_size = CONSERVATIVE_BATCH_SIZE
            
            # Accumulators for bulk insertion
            pending_ids = []
            pending_docs = []
            pending_metas = []
            pending_files = []  # (filepath, rel_path, mtime)
            
            # Performance monitoring
            total_docs_processed = 0
            batch_count = 0
            ingestion_start_time = time.time()
            
            async def flush_batch():
                nonlocal pending_ids, pending_docs, pending_metas, pending_files, changes_detected
                nonlocal total_docs_processed, batch_count
                if not pending_ids:
                    return

                batch_count += 1
                batch_start = time.time()
                batch_size = len(pending_docs)
                
                try:
                    collection = get_collection()
                    
                    # Generate embeddings explicitly using the batched function
                    embed_start = time.time()
                    embeddings = embedding_function(pending_docs)
                    embed_time = time.time() - embed_start
                    
                    # Insert to ChromaDB
                    insert_start = time.time()
                    collection.add(
                        ids=pending_ids,
                        documents=pending_docs,
                        embeddings=embeddings,
                        metadatas=pending_metas
                    )
                    insert_time = time.time() - insert_start
                    
                    batch_time = time.time() - batch_start
                    total_docs_processed += batch_size
                    
                    # Update cache for all successfully processed files
                    processed_paths = set()
                    for _, rp, mtime in pending_files:
                        self.cache[rp] = mtime
                        processed_paths.add(rp)
                        
                    changes_detected += len(processed_paths)
                    
                    # Performance monitoring - log every N batches or on large batches
                    if batch_count % LOG_BATCH_INTERVAL == 0 or batch_size > 1000:
                        elapsed = time.time() - ingestion_start_time
                        rate = total_docs_processed / elapsed if elapsed > 0 else 0
                        
                        logger.info(
                            f"Batch {batch_count}: {batch_size} chunks in {batch_time:.2f}s "
                            f"(embed: {embed_time:.2f}s, insert: {insert_time:.2f}s) | "
                            f"Total: {total_docs_processed} docs, Rate: {rate:.1f} docs/sec"
                        )
                        
                        # Warn if rate drops below expected threshold
                        if rate < MIN_EXPECTED_RATE and total_docs_processed > 100:
                            logger.warning(
                                f"Ingestion rate ({rate:.1f} docs/sec) below threshold ({MIN_EXPECTED_RATE}). "
                                "Check: SQLite locks, memory pressure, or embedding model bottleneck."
                            )
                    
                except Exception as e:
                    logger.error(f"Failed to insert batch {batch_count} ({batch_size} chunks) to ChromaDB: {e}")
                    # In a real system, we might want to retry or handle partial failures
                
                # Reset accumulators
                pending_ids = []
                pending_docs = []
                pending_metas = []
                pending_files = []

            # Process files in batches for I/O efficiency
            total_file_batches = (len(files_to_process) + FILE_BATCH_SIZE - 1) // FILE_BATCH_SIZE
            
            for i in range(0, len(files_to_process), FILE_BATCH_SIZE):
                batch = files_to_process[i:i + FILE_BATCH_SIZE]
                file_batch_num = i // FILE_BATCH_SIZE + 1
                
                if file_batch_num % LOG_BATCH_INTERVAL == 0 or file_batch_num == 1:
                    logger.info(f"Processing file batch {file_batch_num}/{total_file_batches}")
                
                # Process files concurrently
                tasks = [self._process_file(fp, rp) for fp, rp, _ in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Collect results
                for j, result in enumerate(results):
                    file_info = batch[j]  # (filepath, rel_path, mtime)
                    
                    if isinstance(result, Exception):
                        logger.error(f"Failed to process {file_info[1]}: {result}")
                        continue
                        
                    if result:
                        ids, docs, metas = result
                        if ids:
                            pending_ids.extend(ids)
                            pending_docs.extend(docs)
                            pending_metas.extend(metas)
                        
                        # Track file as pending (even if empty, we want to update cache)
                        pending_files.append(file_info)

                # If we have accumulated enough chunks, flush to DB
                # Using conservative 30K limit to prevent exponential slowdown
                if len(pending_docs) >= target_batch_size:
                    await flush_batch()

            # Final flush of any remaining items
            if pending_files:
                await flush_batch()
            
            # Log final performance summary
            total_time = time.time() - ingestion_start_time
            final_rate = total_docs_processed / total_time if total_time > 0 else 0
            logger.info(
                f"Ingestion complete: {total_docs_processed} chunks from {len(files_to_process)} files "
                f"in {total_time:.2f}s ({final_rate:.1f} docs/sec avg)"
            )

        if changes_detected > 0:
            self._save_cache()
            logger.info(f"Cache saved. {changes_detected} files updated.")
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
            # Chunking (CPU bound)
            if rel_path.endswith('.py'):
                # Run CPU-bound chunking in executor
                chunks = await loop.run_in_executor(None, chunk_python_code, code, rel_path, self.repo_name)
            elif rel_path.endswith('.ipynb'):
                # Handle notebooks by converting to simple text rep
                try:
                    nb = json.loads(code)
                    cells = []
                    for cell in nb.get('cells', []):
                        cell_type = cell.get('cell_type', '')
                        source = ''.join(cell.get('source', []))
                        if source.strip():
                            if cell_type == 'code':
                                cells.append(f"```python\n{source}\n```")
                            elif cell_type == 'markdown':
                                cells.append(source)
                    
                    nb_content = "\n\n".join(cells)
                    chunks = [{
                        "id": f"{rel_path}::notebook::0",
                        "content": nb_content,
                        "metadata": {
                            "filepath": rel_path,
                            "repo_id": self.repo_id,
                            "repo_name": self.repo_name,
                            "chunk_type": "notebook",
                            "start_line": 1,
                            "end_line": len(nb_content.splitlines()),
                            "imports": ""
                        }
                    }]
                except Exception as e:
                    logger.warning(f"Failed to parse notebook {rel_path}: {e}")
                    # Fallback to treating as text
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
