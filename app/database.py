"""
ChromaDB client initialization and collection management.
"""
import os
import time
from typing import Optional
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api import ClientAPI
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.config import Settings


# lazy initialization - clients are created on first access
_chroma_client: Optional[ClientAPI] = None
_collection: Optional[Collection] = None
_embedding_func: Optional[embedding_functions.DefaultEmbeddingFunction] = None


def get_collection_name():
    """get the collection name"""
    return os.getenv("CHROMA_COLLECTION_NAME", "python_code_chunks")


def get_embedding_function() -> embedding_functions.DefaultEmbeddingFunction:
    """get or create the embedding function"""
    global _embedding_func
    if _embedding_func is None:
        _embedding_func = embedding_functions.DefaultEmbeddingFunction()
    return _embedding_func


def get_client() -> ClientAPI:
    """get or create the chromadb client instance with retry logic"""
    global _chroma_client
    if _chroma_client is None:
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                _chroma_client = chromadb.HttpClient(
                    host=os.getenv("CHROMA_HOST", "localhost"),
                    port=int(os.getenv("CHROMA_PORT", 8000)),
                    settings=Settings(
                        chroma_client_auth_provider=None,
                        chroma_client_auth_credentials=None,
                    )
                )
                # test the connection
                _chroma_client.heartbeat()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    retry_delay *= 2  # exponential backoff
                else:
                    raise Exception(f"Failed to connect to ChromaDB after {max_retries} attempts: {e}")
    
    # type guard - this will never be None here
    assert _chroma_client is not None, "Client should be initialized"
    return _chroma_client


def get_collection() -> Collection:
    """get or create the current collection instance"""
    global _collection
    if _collection is None:
        client = get_client()
        embedding_func = get_embedding_function()
        collection_name = get_collection_name()
        _collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_func,  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"}
        )
    
    # type guard - this will never be None here
    assert _collection is not None, "Collection should be initialized"
    return _collection


def reset_collection() -> Collection:
    """delete and recreate the collection"""
    global _collection
    client = get_client()
    embedding_func = get_embedding_function()
    collection_name = get_collection_name()
    
    client.delete_collection(name=collection_name)
    _collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_func,  # type: ignore[arg-type]
        metadata={"hnsw:space": "cosine"}
    )
    
    # type guard - this will never be None here
    assert _collection is not None, "Collection should be initialized"
    return _collection


def delete_file_chunks(filepath: str):
    """
    Delete all chunks associated with a specific file.
    Uses metadata filtering: where={"filepath": filepath}
    """
    collection = get_collection()
    try:
        # Note: ChromaDB delete uses 'where' filter for metadata
        collection.delete(where={"filepath": filepath})
    except Exception as e:
        # Log but don't crash if delete fails (e.g. if collection empty)
        print(f"Warning: Failed to delete chunks for {filepath}: {e}")
