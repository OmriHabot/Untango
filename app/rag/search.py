"""
Search utilities including tokenization and hybrid search.
"""
import re
from typing import List, Dict, Any, Optional
import numpy as np
from rank_bm25 import BM25Okapi

from .database import get_collection
from ..core.logger import get_logger

logger = get_logger(__name__)


def distance_to_similarity(distance: float, metric: str = "cosine") -> float:
    """
    Convert distance metric to similarity score.
    
    Args:
        distance: Distance value from ChromaDB
        metric: Distance metric used ("cosine", "l2", "ip")
    
    Returns:
        Similarity score (higher = more similar)
    """
    if metric == "cosine":
        # Cosine distance ranges from 0 to 2, similarity = 1 - distance
        # This gives us similarity in range [0, 1] where 1 = identical
        return 1.0 - distance
    elif metric == "l2":
        # L2 distance: convert to similarity using inverse
        # Adding 1 to avoid division by zero
        return 1.0 / (1.0 + distance)
    elif metric == "ip":
        # Inner product: already a similarity (higher = more similar)
        return distance
    else:
        # Default: assume cosine
        return 1.0 - distance


def tokenize_code(text: str) -> List[str]:
    """
    tokenize code text for bm25 search.
    splits on non-alphanumeric characters and converts to lowercase.
    this helps match partial words like 'auth' in 'authenticate_user'.
    """
    # split on any non-alphanumeric character
    tokens = re.findall(r'\w+', text.lower())
    # filter out very short tokens (like single characters) and python keywords
    return [token for token in tokens if len(token) > 1]


def perform_vector_search(
    query: str,
    n_results: int,
    similarity_threshold: Optional[float] = None,
    distance_metric: str = "cosine",
    repo_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform vector similarity search using chromadb with optional threshold filtering.
    
    Args:
        query: Search query text
        n_results: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0-1) to include result.
                            Results below this threshold are filtered out.
        distance_metric: Distance metric used by ChromaDB ("cosine", "l2", "ip")
        repo_id: Optional repository ID to filter results
    
    Returns:
        Dictionary with filtered results
    """
    collection = get_collection()
    
    # Fetch more results if we're filtering, to ensure we get n_results after filtering
    fetch_count = n_results * 3 if similarity_threshold is not None else n_results
    
    # Build where clause for repo filtering
    where_clause = {"repo_id": repo_id} if repo_id else None
    
    vector_results = collection.query(
        query_texts=[query],
        n_results=fetch_count,
        include=["documents", "metadatas", "distances"],
        where=where_clause
    )
    
    # Apply similarity threshold if specified
    if similarity_threshold is not None and vector_results.get("ids") and vector_results["ids"][0]:
        filtered_ids = []
        filtered_docs = []
        filtered_metas = []
        filtered_distances = []
        filtered_count = 0
        
        ids_list = vector_results["ids"][0]
        docs_list = vector_results["documents"][0]  # type: ignore
        metas_list = vector_results["metadatas"][0]  # type: ignore
        distances_list = vector_results["distances"][0]  # type: ignore
        
        for i, distance in enumerate(distances_list):
            similarity = distance_to_similarity(distance, distance_metric)
            
            if similarity >= similarity_threshold:
                filtered_ids.append(ids_list[i])
                filtered_docs.append(docs_list[i])
                filtered_metas.append(metas_list[i])
                filtered_distances.append(distance)
                filtered_count += 1
                
                if filtered_count >= n_results:
                    break
        
        if filtered_count < len(ids_list):
            logger.info(
                "Vector search filtered: %d/%d results passed similarity threshold %.2f",
                filtered_count,
                len(ids_list),
                similarity_threshold
            )
        
        # Reconstruct results with filtered data
        vector_results = {
            "ids": [filtered_ids],
            "documents": [filtered_docs],
            "metadatas": [filtered_metas],
            "distances": [filtered_distances]
        }
    
    return vector_results  # type: ignore


def perform_hybrid_search(
    query: str,
    n_results: int,
    vector_similarity_threshold: Optional[float] = None,
    bm25_score_threshold: Optional[float] = None,
    distance_metric: str = "cosine",
    repo_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Hybrid search combining vector similarity and BM25 keyword search,
    using Reciprocal Rank Fusion (RRF) to merge rankings as described
    in Chroma's hybrid search docs.
    
    Args:
        query: Search query text
        n_results: Maximum number of results to return
        vector_similarity_threshold: Minimum vector similarity score (0-1) to include in fusion.
                                     Documents below this are excluded from vector ranking.
        bm25_score_threshold: Minimum BM25 score to include in fusion.
                             Documents below this are excluded from BM25 ranking.
        distance_metric: Distance metric used by ChromaDB ("cosine", "l2", "ip")
        repo_id: Optional repository ID to filter results
    
    Returns:
        List of fused results sorted by combined RRF score
    """
    collection = get_collection()

    # Build where clause for repo filtering
    where_clause = {"repo_id": repo_id} if repo_id else None

    # get a reasonably large candidate set from the dense index
    vector_k = max(n_results * 3, n_results)
    vector_results = collection.query(
        query_texts=[query],
        n_results=vector_k,
        include=["documents", "metadatas", "distances"],  # removed "ids" - they're always returned
        where=where_clause
    )

    # ids are returned by default, so we can access them directly
    vector_ids = vector_results["ids"][0]
    vector_docs = vector_results["documents"][0]  # type: ignore
    vector_metas = vector_results["metadatas"][0]  # type: ignore
    vector_distances = vector_results["distances"][0]  # type: ignore
    
    # Filter by vector similarity threshold if specified
    if vector_similarity_threshold is not None:
        filtered_indices = []
        for i, distance in enumerate(vector_distances):
            similarity = distance_to_similarity(distance, distance_metric)
            if similarity >= vector_similarity_threshold:
                filtered_indices.append(i)
        
        if len(filtered_indices) < len(vector_ids):
            logger.info(
                "Hybrid search: %d/%d vector results passed similarity threshold %.2f",
                len(filtered_indices),
                len(vector_ids),
                vector_similarity_threshold
            )
        
        vector_ids = [vector_ids[i] for i in filtered_indices]
        vector_docs = [vector_docs[i] for i in filtered_indices]
        vector_metas = [vector_metas[i] for i in filtered_indices]
        vector_distances = [vector_distances[i] for i in filtered_indices]

    # for BM25 we need the full document set (or a large subset) as in the docs' BM25 examples
    all_docs = collection.get(include=["documents", "metadatas"], where=where_clause)  # Apply repo filter
    if not all_docs["documents"]:
        return []

    corpus_docs = all_docs["documents"]
    corpus_ids = all_docs["ids"]  # ids are always returned
    corpus_metas = all_docs["metadatas"]

    tokenized_docs = [tokenize_code(doc) for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = tokenize_code(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # Store BM25 scores for each document ID
    bm25_score_map: Dict[str, float] = {}
    for doc_idx, score in enumerate(bm25_scores):
        bm25_score_map[corpus_ids[doc_idx]] = score

    # Filter by BM25 score threshold if specified
    if bm25_score_threshold is not None:
        filtered_doc_ids = set()
        for doc_id, score in bm25_score_map.items():
            if score >= bm25_score_threshold:
                filtered_doc_ids.add(doc_id)
        
        original_count = len(bm25_score_map)
        if len(filtered_doc_ids) < original_count:
            logger.info(
                "Hybrid search: %d/%d BM25 results passed score threshold %.2f",
                len(filtered_doc_ids),
                original_count,
                bm25_score_threshold
            )
        
        # Only keep scores for documents that passed the threshold
        bm25_score_map = {k: v for k, v in bm25_score_map.items() if k in filtered_doc_ids}

    # get BM25 ranking as an ordering of document indices (highest score first)
    # Only rank documents that passed the threshold (if any)
    ranked_bm25_items = sorted(bm25_score_map.items(), key=lambda x: x[1], reverse=True)
    
    # map doc_id -> rank (1-based) for BM25
    bm25_rank: Dict[str, int] = {}
    for rank_idx, (doc_id, _) in enumerate(ranked_bm25_items, start=1):
        bm25_rank[doc_id] = rank_idx

    # for vectors, rank is just the index in vector_ids (1-based, already sorted by distance)
    vector_rank: Dict[str, int] = {}
    vector_distance_map: Dict[str, float] = {}
    for rank_idx, doc_id in enumerate(vector_ids, start=1):
        vector_rank[doc_id] = rank_idx
        vector_distance_map[doc_id] = vector_distances[rank_idx - 1]

    # following the hybrid search with RRF pattern from the Chroma docs.
    k_rrf = 60  # standard constant; can be tuned

    # union of doc_ids that appear in either ranking (only those that passed thresholds)
    all_candidate_ids = set(bm25_rank.keys()) | set(vector_rank.keys())
    
    # Log if threshold filtering resulted in no candidates
    if not all_candidate_ids:
        logger.warning(
            "Hybrid search: No documents passed the specified thresholds "
            "(vector_threshold=%.2f, bm25_threshold=%.2f)",
            vector_similarity_threshold or 0.0,
            bm25_score_threshold or 0.0
        )
        return []

    fused_results: List[Dict[str, Any]] = []

    for doc_id in all_candidate_ids:
        # dense side
        r_dense = vector_rank.get(doc_id)
        dense_rrf = 1.0 / (k_rrf + r_dense) if r_dense is not None else 0.0

        # sparse side (BM25)
        r_bm25 = bm25_rank.get(doc_id)
        bm25_rrf = 1.0 / (k_rrf + r_bm25) if r_bm25 is not None else 0.0

        combined_score = dense_rrf + bm25_rrf

        # Get distance and calculate similarity for vector results
        distance = vector_distance_map.get(doc_id)
        similarity = distance_to_similarity(distance, distance_metric) if distance is not None else None
        
        # Get BM25 score
        bm25_score = bm25_score_map.get(doc_id)

        # pull content/metadata from whichever source we have handy
        content = ""
        metadata = {}
        if doc_id in corpus_ids:
            idx = corpus_ids.index(doc_id)
            content = corpus_docs[idx]
            metadata = corpus_metas[idx]  # type: ignore
        else:
            # fallback: find in vector results
            if doc_id in vector_ids:
                idx = vector_ids.index(doc_id)
                content = vector_docs[idx]
                metadata = vector_metas[idx]

        result = {
            "id": doc_id,
            "content": content,
            "metadata": metadata,
            "vector_rank": r_dense,
            "bm25_rank": r_bm25,
            "rrf_dense": dense_rrf,
            "rrf_bm25": bm25_rrf,
            "combined_score": combined_score,
        }
        
        # Add optional fields if available
        if distance is not None:
            result["distance"] = distance
        if similarity is not None:
            result["similarity"] = similarity
        if bm25_score is not None:
            result["bm25_score"] = bm25_score
        
        fused_results.append(result)

    fused_results.sort(key=lambda x: x["combined_score"], reverse=True)
    return fused_results[:n_results]
