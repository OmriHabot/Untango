"""
Custom embedding function using SentenceTransformer with hardware acceleration and batching.
"""
import logging
from typing import Optional, List, Union
import torch
from chromadb import EmbeddingFunction, Documents, Embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function that uses SentenceTransformer with:
    - Automatic device selection (CUDA > MPS > CPU)
    - Batch processing
    - Configurable model and batch size
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 32
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        
        logger.info(f"Initializing SentenceTransformer model '{model_name}' on device: {device}")
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as e:
            logger.error(f"Failed to initialize SentenceTransformer: {e}")
            raise

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for a list of documents.
        """
        try:
            # SentenceTransformer handles batching internally via the batch_size parameter
            embeddings = self.model.encode(
                input,
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Convert numpy array to list of lists as expected by ChromaDB
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
