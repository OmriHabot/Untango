"""
Search utilities including tokenization and hybrid search.
"""
import re
from typing import List, Dict, Any
import numpy as np
from rank_bm25 import BM25Okapi

from .database import get_collection


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


def perform_vector_search(query: str, n_results: int) -> Dict[str, Any]:
    """
    perform vector similarity search using chromadb.
    """
    collection = get_collection()
    vector_results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return vector_results


def perform_hybrid_search(query: str, n_results: int) -> List[Dict[str, Any]]:
    """
    hybrid search combining vector similarity and BM25 keyword search,
    using Reciprocal Rank Fusion (RRF) to merge rankings as described
    in Chroma's hybrid search docs.
    """
    collection = get_collection()

    # get a reasonably large candidate set from the dense index
    vector_k = max(n_results * 2, n_results)
    vector_results = collection.query(
        query_texts=[query],
        n_results=vector_k,
        include=["documents", "metadatas", "distances"],  # removed "ids" - they're always returned
    )

    # ids are returned by default, so we can access them directly
    vector_ids = vector_results["ids"][0]
    vector_docs = vector_results["documents"][0]
    vector_metas = vector_results["metadatas"][0]
    vector_distances = vector_results["distances"][0]

    # for BM25 we need the full document set (or a large subset) as in the docs' BM25 examples
    all_docs = collection.get(include=["documents", "metadatas"])  # removed "ids" here too
    if not all_docs["documents"]:
        return []

    corpus_docs = all_docs["documents"]
    corpus_ids = all_docs["ids"]  # ids are always returned
    corpus_metas = all_docs["metadatas"]

    tokenized_docs = [tokenize_code(doc) for doc in corpus_docs]
    bm25 = BM25Okapi(tokenized_docs)

    tokenized_query = tokenize_code(query)
    bm25_scores = bm25.get_scores(tokenized_query)

    # get BM25 ranking as an ordering of document indices (highest score first)
    bm25_order = np.argsort(-np.array(bm25_scores))  # descending
    # map doc_id -> rank (1-based) for BM25
    bm25_rank: Dict[str, int] = {}
    for rank_idx, doc_idx in enumerate(bm25_order, start=1):
        doc_id = corpus_ids[doc_idx]
        bm25_rank[doc_id] = rank_idx

    # for vectors, rank is just the index in vector_ids (1-based, already sorted by distance)
    vector_rank: Dict[str, int] = {}
    for rank_idx, doc_id in enumerate(vector_ids, start=1):
        vector_rank[doc_id] = rank_idx

    # following the hybrid search with RRF pattern from the Chroma docs.
    k_rrf = 60  # standard constant; can be tuned

    # union of doc_ids that appear in either ranking
    all_candidate_ids = set(corpus_ids) | set(vector_ids)

    fused_results: List[Dict[str, Any]] = []

    for doc_id in all_candidate_ids:
        # dense side
        r_dense = vector_rank.get(doc_id)
        dense_rrf = 1.0 / (k_rrf + r_dense) if r_dense is not None else 0.0

        # sparse side (BM25)
        r_bm25 = bm25_rank.get(doc_id)
        bm25_rrf = 1.0 / (k_rrf + r_bm25) if r_bm25 is not None else 0.0

        combined_score = dense_rrf + bm25_rrf

        # pull content/metadata from whichever source we have handy
        if doc_id in corpus_ids:
            idx = corpus_ids.index(doc_id)
            content = corpus_docs[idx]
            metadata = corpus_metas[idx]
        else:
            # fallback: find in vector results
            if doc_id in vector_ids:
                idx = vector_ids.index(doc_id)
                content = vector_docs[idx]
                metadata = vector_metas[idx]
            else:
                # should not happen, but guard anyway
                content = ""
                metadata = {}

        fused_results.append(
            {
                "id": doc_id,
                "content": content,
                "metadata": metadata,
                "vector_rank": r_dense,
                "bm25_rank": r_bm25,
                "rrf_dense": dense_rrf,
                "rrf_bm25": bm25_rrf,
                "combined_score": combined_score,
            }
        )

    fused_results.sort(key=lambda x: x["combined_score"], reverse=True)
    return fused_results[:n_results]
