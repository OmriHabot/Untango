"""
ChromaDB client initialization and collection management.
"""
import os
import time
from typing import Optional
import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.api import ClientAPI
from chromadb.config import Settings
from app.embeddings import SentenceTransformerEmbeddingFunction
import google.auth
import google.auth.transport.requests
from google.oauth2 import id_token

# lazy initialization - clients are created on first access
_chroma_client: Optional[ClientAPI] = None
_collection: Optional[Collection] = None
_embedding_func: Optional[SentenceTransformerEmbeddingFunction] = None


def get_collection_name():
    """get the collection name"""
    return os.getenv("CHROMA_COLLECTION_NAME", "python_code_chunks")


def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """get or create the embedding function"""
    global _embedding_func
    if _embedding_func is None:
        # Use a larger batch size for better throughput on GPU/MPS
        # 128 provides good GPU utilization for large ingestion batches
        _embedding_func = SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            batch_size=128
        )
    return _embedding_func


def get_client() -> ClientAPI:
    """get or create the chromadb client instance with retry logic"""
    global _chroma_client
    if _chroma_client is None:
        max_retries = 5
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                host = os.getenv("CHROMA_HOST", "localhost")
                port = int(os.getenv("CHROMA_PORT", 8000))
                ssl = os.getenv("CHROMA_SSL", "FALSE").upper() == "TRUE"
                auth_provider = os.getenv("CHROMA_AUTH_PROVIDER", "none")

                settings = Settings(
                    chroma_client_auth_provider=None,
                    chroma_client_auth_credentials=None,
                )

                if auth_provider == "google_iam":
                    # Generate ID token for the target audience (Chroma host)
                    target_audience = f"https://{host}" if ssl and not host.startswith("http") else host
                    
                    # If host is just a domain, 'https://' should be prepended for audience, 
                    # but check if user provided full URL in CHROMA_HOST
                    if "://" not in target_audience:
                         target_audience = f"https://{target_audience}"

                    try:
                        auth_req = google.auth.transport.requests.Request()
                        # Verify we can get credentials (implicitly uses GOOGLE_APPLICATION_CREDENTIALS)
                        creds, project = google.auth.default()
                        
                        # Note: For Cloud Run, we need an ID token, not an access token.
                        # However, google.auth.default() returns credentials that might be for a service account.
                        # We specifically need to generate an ID token.
                        # If we are using a service account key file:
                        token = id_token.fetch_id_token(auth_req, target_audience)
                        
                        settings = Settings(
                            chroma_client_auth_provider=None,
                            chroma_client_auth_credentials=None,
                            chroma_server_headers={"Authorization": f"Bearer {token}"}
                        )
                    except Exception as e:
                        print(f"Failed to generate Google ID token: {e}")
                        raise

                _chroma_client = chromadb.HttpClient(
                    host=host,
                    port=port,
                    ssl=ssl,
                    settings=settings
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
