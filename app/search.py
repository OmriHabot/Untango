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
    hybrid search combining vector similarity and bm25 keyword search.
    this provides better results by leveraging both semantic and lexical matching.
    """
    collection = get_collection()
    
    # step 1: vector search
    vector_results = collection.query(
        query_texts=[query],
        n_results=n_results * 2,  # get more candidates
        include=["documents", "metadatas", "distances"]
    )
    
    # step 2: bm25 keyword search on the same candidates
    # extract all documents from collection for bm25 indexing
    all_docs = collection.get(
        include=["documents", "metadatas"]
    )
    
    if not all_docs["documents"]:
        return []
    
    # tokenize documents for bm25 using code-aware tokenization
    tokenized_docs = [tokenize_code(doc) for doc in all_docs["documents"]]
    bm25 = BM25Okapi(tokenized_docs)
    
    # score with bm25
    tokenized_query = tokenize_code(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # normalize scores
    bm25_scores = np.array(bm25_scores)
    if bm25_scores.max() > 0:
        bm25_scores = bm25_scores / bm25_scores.max()
    
    # combine scores (reciprocal rank fusion or simple weighted average)
    combined_results = {}
    
    # add vector search scores (convert distances to similarities)
    for i, doc_id in enumerate(vector_results["ids"][0]):
        distance = vector_results["distances"][0][i]
        # convert cosine distance to similarity (0 to 1)
        similarity = 1 - distance
        combined_results[doc_id] = {
            "content": vector_results["documents"][0][i],
            "metadata": vector_results["metadatas"][0][i],
            "vector_score": similarity,
            "bm25_score": 0.0,
            "combined_score": similarity * 0.5  # weight vector search 50%
        }
    
    # add bm25 scores
    for i, doc_id in enumerate(all_docs["ids"]):
        if doc_id in combined_results:
            combined_results[doc_id]["bm25_score"] = float(bm25_scores[i])
            # update combined score (simple average)
            combined_results[doc_id]["combined_score"] = (
                combined_results[doc_id]["vector_score"] * 0.5 +
                bm25_scores[i] * 0.5
            )
    
    # sort by combined score and return top-k
    sorted_results = sorted(
        combined_results.values(),
        key=lambda x: x["combined_score"],
        reverse=True
    )[:n_results]
    
    return sorted_results

