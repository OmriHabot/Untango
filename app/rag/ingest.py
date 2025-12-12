"""
Ingest Manager
Handles ingestion of the repository into ChromaDB using upsert-based ingestion.
Uses ChromaDB's upsert() method which handles both inserts and updates atomically,
eliminating the need for separate delete operations.

Optimized batching based on ChromaDB research:
- ChromaDB default max batch: 41,666
- Conservative batch size of 30K prevents SQLite overhead
- Upsert combines add/update in a single operation
"""
import os
import json
import logging
import asyncio
import time
from typing import Dict, Set, List, Any, Tuple

from .chunker import chunk_python_code
from .database import get_collection, delete_file_chunks

logger = logging.getLogger(__name__)

# Optimized batch sizing for ChromaDB ingestion
CHROMADB_MAX_BATCH = 41_666  # ChromaDB default limit [web:19]
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
        Uses ChromaDB's upsert() method for efficient updates:
        1. Collect all files that need processing
        2. Generate ALL chunks upfront from all files
        3. Batch upsert - ChromaDB handles embeddings and updates atomically
        """
        logger.info("Starting smart ingestion sync (ChromaDB upsert-based ingestion)...")
        
        # ============================================================
        # PHASE 1: Identify files to process
        # ============================================================
        files_to_process: List[Tuple[str, str, float]] = []  # (filepath, rel_path, mtime)
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

        # ============================================================
        # PHASE 2: Handle deletions
        # ============================================================
        changes_detected = 0
        files_deleted = []
        for cached_file in list(self.cache.keys()):
            if cached_file not in current_files:
                logger.info(f"File removed: {cached_file}")
                files_deleted.append(cached_file)
                del self.cache[cached_file]
                changes_detected += 1
        
        # Batch delete chunks for removed files
        collection = get_collection()
        for filepath in files_deleted:
            try:
                # Query for chunks belonging to this file
                results = collection.get(
                    where={"filepath": filepath}
                )
                if results['ids']:
                    collection.delete(ids=results['ids'])
                    logger.info(f"Deleted {len(results['ids'])} chunks for {filepath}")
            except Exception as e:
                logger.warning(f"Failed to delete chunks for {filepath}: {e}")

        if not files_to_process:
            if changes_detected > 0:
                self._save_cache()
                logger.info(f"Cache saved. {changes_detected} files deleted, no new files to ingest.")
            else:
                logger.debug("No changes detected.")
            return

        logger.info(f"Found {len(files_to_process)} files to ingest.")
        ingestion_start_time = time.time()

        # ============================================================
        # PHASE 3: Generate ALL chunks upfront from all files
        # ============================================================
        logger.info("Phase 3: Generating all chunks from files...")
        
        all_ids: List[str] = []
        all_docs: List[str] = []
        all_metas: List[Dict[str, Any]] = []
        successfully_processed_files: List[Tuple[str, str, float]] = []
        
        # Process files in batches for I/O efficiency
        total_file_batches = (len(files_to_process) + FILE_BATCH_SIZE - 1) // FILE_BATCH_SIZE
        
        for i in range(0, len(files_to_process), FILE_BATCH_SIZE):
            batch = files_to_process[i:i + FILE_BATCH_SIZE]
            file_batch_num = i // FILE_BATCH_SIZE + 1
            
            if file_batch_num % LOG_BATCH_INTERVAL == 0 or file_batch_num == 1:
                logger.info(f"Chunking file batch {file_batch_num}/{total_file_batches}")
            
            # Process files concurrently
            tasks = [self._generate_chunks_for_file(fp, rp) for fp, rp, _ in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for j, result in enumerate(results):
                file_info = batch[j]  # (filepath, rel_path, mtime)
                
                if isinstance(result, Exception):
                    logger.error(f"Failed to chunk {file_info[1]}: {result}")
                    continue
                    
                if result:
                    ids, docs, metas = result
                    if ids:
                        all_ids.extend(ids)
                        all_docs.extend(docs)
                        all_metas.extend(metas)
                    
                    # Track file as successfully processed
                    successfully_processed_files.append(file_info)

        total_chunks = len(all_ids)
        chunk_time = time.time() - ingestion_start_time
        logger.info(f"Generated {total_chunks} chunks from {len(successfully_processed_files)} files in {chunk_time:.2f}s")

        if total_chunks == 0:
            logger.info("No chunks generated, nothing to upsert.")
            # Still update cache for processed files (they might be empty)
            for filepath, rel_path, mtime in successfully_processed_files:
                self.cache[rel_path] = mtime
                changes_detected += 1
            if changes_detected > 0:
                self._save_cache()
            return

        # ============================================================
        # PHASE 4: Batch upsert ALL chunks using ChromaDB's upsert()
        # ============================================================
        logger.info(f"Phase 4: Upserting {total_chunks} chunks using ChromaDB upsert()...")
        
        insert_start_time = time.time()
        total_upserted = 0
        batch_count = 0
        
        for i in range(0, total_chunks, CONSERVATIVE_BATCH_SIZE):
            batch_ids = all_ids[i:i + CONSERVATIVE_BATCH_SIZE]
            batch_docs = all_docs[i:i + CONSERVATIVE_BATCH_SIZE]
            batch_metas = all_metas[i:i + CONSERVATIVE_BATCH_SIZE]
            
            batch_count += 1
            batch_size = len(batch_ids)
            batch_start = time.time()
            
            try:
                # Use upsert() - atomically handles both inserts and updates
                # If ID exists, updates it; if not, creates it [web:7][web:14]
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas
                    # NO embeddings parameter - ChromaDB generates them automatically
                )
                
                batch_time = time.time() - batch_start
                total_upserted += batch_size
                
                # Performance monitoring
                if batch_count % LOG_BATCH_INTERVAL == 0 or batch_size > 1000:
                    elapsed = time.time() - insert_start_time
                    rate = total_upserted / elapsed if elapsed > 0 else 0
                    
                    logger.info(
                        f"Batch {batch_count}: {batch_size} chunks upserted in {batch_time:.2f}s | "
                        f"Total: {total_upserted}/{total_chunks} docs, Rate: {rate:.1f} docs/sec"
                    )
                    
                    # Warn if rate drops below expected threshold
                    if rate < MIN_EXPECTED_RATE and total_upserted > 100:
                        logger.warning(
                            f"Ingestion rate ({rate:.1f} docs/sec) below threshold ({MIN_EXPECTED_RATE}). "
                            "Check: SQLite locks, memory pressure, or embedding model bottleneck."
                        )
                        
            except Exception as e:
                logger.error(f"Failed to upsert batch {batch_count} ({batch_size} chunks) to ChromaDB: {e}")
                # Continue with next batch

        # ============================================================
        # PHASE 5: Update cache for successfully processed files
        # ============================================================
        for filepath, rel_path, mtime in successfully_processed_files:
            self.cache[rel_path] = mtime
            changes_detected += 1

        # Log final performance summary
        total_time = time.time() - ingestion_start_time
        insert_time = time.time() - insert_start_time
        final_rate = total_upserted / total_time if total_time > 0 else 0
        
        logger.info(
            f"Ingestion complete: {total_upserted} chunks from {len(successfully_processed_files)} files "
            f"in {total_time:.2f}s (chunking: {chunk_time:.2f}s, upsert: {insert_time:.2f}s) | "
            f"Rate: {final_rate:.1f} docs/sec avg"
        )

        if changes_detected > 0:
            self._save_cache()
            logger.info(f"Cache saved. {changes_detected} files updated.")

    async def _generate_chunks_for_file(self, filepath: str, rel_path: str) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
        """
        Read and chunk a single file.
        Returns (ids, documents, metadatas).
        Does NOT interact with the database - only generates chunks.
        """
        try:
            # Run file I/O in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            with open(filepath, 'r', encoding='utf-8') as f:
                code = await loop.run_in_executor(None, f.read)
            
            # Generate chunks based on file type
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
                                cells.append(f"``````")
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
                # Markdown and other text files
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
