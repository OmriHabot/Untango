"""
RAG (Retrieval-Augmented Generation) pipeline module.
Contains database, embeddings, chunking, search, and ingestion components.
"""
from .database import get_client, get_collection, get_collection_name, get_embedding_function, reset_collection, delete_file_chunks
from .embeddings import SentenceTransformerEmbeddingFunction
from .chunker import chunk_python_code
from .search import perform_vector_search, perform_hybrid_search, tokenize_code, distance_to_similarity
from .ingest import IngestManager, ingest_manager

__all__ = [
    # Database
    "get_client",
    "get_collection",
    "get_collection_name",
    "get_embedding_function",
    "reset_collection",
    "delete_file_chunks",
    # Embeddings
    "SentenceTransformerEmbeddingFunction",
    # Chunker
    "chunk_python_code",
    # Search
    "perform_vector_search",
    "perform_hybrid_search",
    "tokenize_code",
    "distance_to_similarity",
    # Ingest
    "IngestManager",
    "ingest_manager",
]
