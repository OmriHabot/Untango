"""
Script to compare /query (vector search) and /query-hybrid (hybrid search) endpoints.
Calculates and displays relevant metrics for both approaches.
"""
import time
import requests
import statistics
from typing import List, Dict, Any, Optional
import json
from collections import Counter


# Configuration
API_BASE_URL = "http://localhost:8001"
N_RESULTS = 10  # Number of results to retrieve per query

# Threshold settings
# Set to None to disable, or use values like 0.5, 0.7, etc.
VECTOR_SIMILARITY_THRESHOLD = None  # Range: 0.0 to 1.0 (higher = more strict)
BM25_SCORE_THRESHOLD = None  # Range: 0.0+ (higher = more strict, typical good scores are 5-20)

# For testing with thresholds, uncomment these:
# VECTOR_SIMILARITY_THRESHOLD = 0.5  # Only keep results with >50% similarity
# BM25_SCORE_THRESHOLD = 2.0  # Only keep results with BM25 score > 2.0


# Test queries - customize these based on your codebase
TEST_QUERIES = [
    "authentication and user login",
    "database connection",
    "error handling",
    "logging functionality",
    "search implementation",
    "code chunking",
    "API endpoints",
    "vector embeddings",
]


def call_vector_search(
    query: str,
    n_results: int = N_RESULTS,
    vector_similarity_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Call the /query endpoint (vector search only)"""
    start_time = time.time()
    
    payload = {"query": query, "n_results": n_results}
    if vector_similarity_threshold is not None:
        payload["vector_similarity_threshold"] = vector_similarity_threshold
    
    response = requests.post(
        f"{API_BASE_URL}/query",
        json=payload,
        timeout=30
    )
    response_time = time.time() - start_time
    
    response.raise_for_status()
    result = response.json()
    result["response_time"] = response_time
    return result


def call_hybrid_search(
    query: str,
    n_results: int = N_RESULTS,
    vector_similarity_threshold: Optional[float] = None,
    bm25_score_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Call the /query-hybrid endpoint (vector + BM25)"""
    start_time = time.time()
    
    payload = {"query": query, "n_results": n_results}
    if vector_similarity_threshold is not None:
        payload["vector_similarity_threshold"] = vector_similarity_threshold
    if bm25_score_threshold is not None:
        payload["bm25_score_threshold"] = bm25_score_threshold
    
    response = requests.post(
        f"{API_BASE_URL}/query-hybrid",
        json=payload,
        timeout=30
    )
    response_time = time.time() - start_time
    
    response.raise_for_status()
    result = response.json()
    result["response_time"] = response_time
    return result


def calculate_result_overlap(vector_results: List[Dict], hybrid_results: List[Dict]) -> Dict[str, Any]:
    """Calculate overlap between vector and hybrid search results"""
    vector_ids = {r["id"] for r in vector_results}
    hybrid_ids = {r["id"] for r in hybrid_results}
    
    intersection = vector_ids & hybrid_ids
    union = vector_ids | hybrid_ids
    
    jaccard_similarity = len(intersection) / len(union) if union else 0
    overlap_count = len(intersection)
    
    # Calculate rank correlation for overlapping items
    rank_differences = []
    for doc_id in intersection:
        vector_rank = next(i for i, r in enumerate(vector_results) if r["id"] == doc_id)
        hybrid_rank = next(i for i, r in enumerate(hybrid_results) if r["id"] == doc_id)
        rank_differences.append(abs(vector_rank - hybrid_rank))
    
    avg_rank_difference = statistics.mean(rank_differences) if rank_differences else 0
    
    return {
        "overlap_count": overlap_count,
        "overlap_percentage": (overlap_count / len(vector_results)) * 100 if vector_results else 0,
        "jaccard_similarity": jaccard_similarity,
        "avg_rank_difference": avg_rank_difference,
        "vector_only_count": len(vector_ids - hybrid_ids),
        "hybrid_only_count": len(hybrid_ids - vector_ids)
    }


def calculate_diversity(results: List[Dict]) -> Dict[str, Any]:
    """Calculate diversity metrics for search results"""
    if not results:
        return {"unique_files": 0, "unique_types": 0, "unique_repos": 0}
    
    filepaths = [r["metadata"].get("filepath", "unknown") for r in results]
    chunk_types = [r["metadata"].get("chunk_type", "unknown") for r in results]
    repos = [r["metadata"].get("repo_name", "unknown") for r in results]
    
    return {
        "unique_files": len(set(filepaths)),
        "unique_types": len(set(chunk_types)),
        "unique_repos": len(set(repos)),
        "type_distribution": dict(Counter(chunk_types))
    }


def calculate_score_stats(results: List[Dict], score_field: str) -> Dict[str, float]:
    """Calculate statistics for score distributions"""
    if not results or score_field not in results[0]:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "std_dev": 0}
    
    scores = [r[score_field] for r in results]
    
    return {
        "min": min(scores),
        "max": max(scores),
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "std_dev": statistics.stdev(scores) if len(scores) > 1 else 0
    }


def print_section_header(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subsection(title: str):
    """Print a formatted subsection"""
    print(f"\n--- {title} ---")


def run_comparison():
    """Main comparison function"""
    print_section_header("SEARCH ENDPOINT COMPARISON")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Number of test queries: {len(TEST_QUERIES)}")
    print(f"Results per query: {N_RESULTS}")
    print(f"\n📊 Threshold Settings:")
    print(f"   • Vector Similarity Threshold: {VECTOR_SIMILARITY_THRESHOLD if VECTOR_SIMILARITY_THRESHOLD is not None else 'Disabled'}")
    print(f"   • BM25 Score Threshold: {BM25_SCORE_THRESHOLD if BM25_SCORE_THRESHOLD is not None else 'Disabled'}")
    
    if VECTOR_SIMILARITY_THRESHOLD is not None or BM25_SCORE_THRESHOLD is not None:
        print(f"\n⚖️  Absolute Scoring Thresholds ENABLED")
        print(f"   Results below these thresholds will be filtered before RRF fusion.")
    else:
        print(f"\n⚖️  Absolute Scoring Thresholds DISABLED")
        print(f"   All results participate in RRF fusion (standard hybrid search).")
    
    # Collect results for all queries
    all_vector_results = []
    all_hybrid_results = []
    all_response_times = {"vector": [], "hybrid": []}
    all_overlaps = []
    
    for i, query in enumerate(TEST_QUERIES, 1):
        print_section_header(f"Query {i}/{len(TEST_QUERIES)}: '{query}'")
        
        try:
            # Call vector search
            print("\n📊 Calling vector search endpoint...")
            vector_result = call_vector_search(
                query,
                vector_similarity_threshold=VECTOR_SIMILARITY_THRESHOLD
            )
            all_vector_results.append(vector_result)
            all_response_times["vector"].append(vector_result["response_time"])
            
            print(f"   ✓ Vector search completed in {vector_result['response_time']:.3f}s")
            print(f"   ✓ Retrieved {vector_result['count']} results")
            
            # Call hybrid search
            print("\n📊 Calling hybrid search endpoint...")
            hybrid_result = call_hybrid_search(
                query,
                vector_similarity_threshold=VECTOR_SIMILARITY_THRESHOLD,
                bm25_score_threshold=BM25_SCORE_THRESHOLD
            )
            all_hybrid_results.append(hybrid_result)
            all_response_times["hybrid"].append(hybrid_result["response_time"])
            
            print(f"   ✓ Hybrid search completed in {hybrid_result['response_time']:.3f}s")
            print(f"   ✓ Retrieved {hybrid_result['count']} results")
            
            # Calculate overlap
            overlap = calculate_result_overlap(
                vector_result["results"],
                hybrid_result["results"]
            )
            all_overlaps.append(overlap)
            
            print_subsection("Result Overlap")
            print(f"   • Overlap: {overlap['overlap_count']}/{N_RESULTS} ({overlap['overlap_percentage']:.1f}%)")
            print(f"   • Jaccard similarity: {overlap['jaccard_similarity']:.3f}")
            print(f"   • Avg rank difference: {overlap['avg_rank_difference']:.1f} positions")
            print(f"   • Vector-only results: {overlap['vector_only_count']}")
            print(f"   • Hybrid-only results: {overlap['hybrid_only_count']}")
            
            # Show top 3 results for each method
            print_subsection("Top 3 Vector Search Results")
            for j, result in enumerate(vector_result["results"][:3], 1):
                meta = result["metadata"]
                print(f"   {j}. [{meta.get('chunk_type', 'unknown')}] {meta.get('filepath', 'unknown')}")
                print(f"      Distance: {result['distance']:.4f}")
            
            print_subsection("Top 3 Hybrid Search Results")
            for j, result in enumerate(hybrid_result["results"][:3], 1):
                meta = result["metadata"]
                print(f"   {j}. [{meta.get('chunk_type', 'unknown')}] {meta.get('filepath', 'unknown')}")
                print(f"      Combined Score: {result['combined_score']:.4f} "
                      f"(Vector RRF: {result.get('rrf_dense', 0):.4f}, BM25 RRF: {result.get('rrf_bm25', 0):.4f})")
                
                # Show raw scores if available (when thresholds are enabled)
                if 'similarity' in result or 'bm25_score' in result:
                    raw_scores = []
                    if 'similarity' in result:
                        raw_scores.append(f"Similarity: {result['similarity']:.4f}")
                    if 'bm25_score' in result:
                        raw_scores.append(f"BM25: {result['bm25_score']:.2f}")
                    if raw_scores:
                        print(f"      Raw Scores: {', '.join(raw_scores)}")
            
        except requests.exceptions.RequestException as e:
            print(f"\n❌ Error querying '{query}': {e}")
            continue
    
    # Overall statistics
    print_section_header("OVERALL COMPARISON STATISTICS")
    
    print_subsection("Response Time Comparison")
    if all_response_times["vector"]:
        print(f"   Vector Search:")
        print(f"      • Mean: {statistics.mean(all_response_times['vector']):.3f}s")
        print(f"      • Median: {statistics.median(all_response_times['vector']):.3f}s")
        print(f"      • Min: {min(all_response_times['vector']):.3f}s")
        print(f"      • Max: {max(all_response_times['vector']):.3f}s")
    
    if all_response_times["hybrid"]:
        print(f"\n   Hybrid Search:")
        print(f"      • Mean: {statistics.mean(all_response_times['hybrid']):.3f}s")
        print(f"      • Median: {statistics.median(all_response_times['hybrid']):.3f}s")
        print(f"      • Min: {min(all_response_times['hybrid']):.3f}s")
        print(f"      • Max: {max(all_response_times['hybrid']):.3f}s")
    
    if all_response_times["vector"] and all_response_times["hybrid"]:
        avg_vector = statistics.mean(all_response_times['vector'])
        avg_hybrid = statistics.mean(all_response_times['hybrid'])
        diff = avg_hybrid - avg_vector
        diff_pct = (diff / avg_vector) * 100
        print(f"\n   Δ Hybrid is {abs(diff):.3f}s ({'slower' if diff > 0 else 'faster'}, {abs(diff_pct):.1f}%)")
    
    print_subsection("Result Overlap Statistics (Across All Queries)")
    if all_overlaps:
        avg_overlap_pct = statistics.mean([o["overlap_percentage"] for o in all_overlaps])
        avg_jaccard = statistics.mean([o["jaccard_similarity"] for o in all_overlaps])
        avg_rank_diff = statistics.mean([o["avg_rank_difference"] for o in all_overlaps])
        
        print(f"   • Average overlap: {avg_overlap_pct:.1f}%")
        print(f"   • Average Jaccard similarity: {avg_jaccard:.3f}")
        print(f"   • Average rank difference: {avg_rank_diff:.1f} positions")
        print(f"   • Total queries with 100% overlap: {sum(1 for o in all_overlaps if o['overlap_percentage'] == 100)}")
        print(f"   • Total queries with <50% overlap: {sum(1 for o in all_overlaps if o['overlap_percentage'] < 50)}")
    
    # Diversity analysis
    print_subsection("Result Diversity Analysis")
    for method_name, results_list in [("Vector Search", all_vector_results), 
                                       ("Hybrid Search", all_hybrid_results)]:
        print(f"\n   {method_name}:")
        all_results = []
        for query_result in results_list:
            all_results.extend(query_result["results"])
        
        if all_results:
            diversity = calculate_diversity(all_results)
            print(f"      • Unique files retrieved: {diversity['unique_files']}")
            print(f"      • Unique chunk types: {diversity['unique_types']}")
            print(f"      • Unique repositories: {diversity['unique_repos']}")
            print(f"      • Type distribution: {diversity['type_distribution']}")
    
    # Score distribution analysis
    print_subsection("Score Distribution Analysis")
    
    # Vector search uses distance (lower is better)
    all_vector_distances = []
    for result in all_vector_results:
        all_vector_distances.extend(result["results"])
    if all_vector_distances:
        distance_stats = calculate_score_stats(all_vector_distances, "distance")
        print(f"\n   Vector Search Distance (lower = more similar):")
        print(f"      • Min: {distance_stats['min']:.4f}")
        print(f"      • Max: {distance_stats['max']:.4f}")
        print(f"      • Mean: {distance_stats['mean']:.4f}")
        print(f"      • Median: {distance_stats['median']:.4f}")
        print(f"      • Std Dev: {distance_stats['std_dev']:.4f}")
    
    # Hybrid search uses combined_score (higher is better)
    all_hybrid_scores = []
    for result in all_hybrid_results:
        all_hybrid_scores.extend(result["results"])
    if all_hybrid_scores:
        combined_stats = calculate_score_stats(all_hybrid_scores, "combined_score")
        rrf_dense_stats = calculate_score_stats(all_hybrid_scores, "rrf_dense")
        rrf_bm25_stats = calculate_score_stats(all_hybrid_scores, "rrf_bm25")
        
        print(f"\n   Hybrid Search Combined Score (higher = more relevant):")
        print(f"      • Min: {combined_stats['min']:.4f}")
        print(f"      • Max: {combined_stats['max']:.4f}")
        print(f"      • Mean: {combined_stats['mean']:.4f}")
        print(f"      • Median: {combined_stats['median']:.4f}")
        print(f"      • Std Dev: {combined_stats['std_dev']:.4f}")
        
        print(f"\n   RRF Dense Component:")
        print(f"      • Mean: {rrf_dense_stats['mean']:.4f}")
        print(f"      • Std Dev: {rrf_dense_stats['std_dev']:.4f}")
        
        print(f"\n   RRF BM25 Component:")
        print(f"      • Mean: {rrf_bm25_stats['mean']:.4f}")
        print(f"      • Std Dev: {rrf_bm25_stats['std_dev']:.4f}")
    
    # Recommendations
    print_section_header("RECOMMENDATIONS")
    
    if all_response_times["vector"] and all_response_times["hybrid"]:
        avg_vector = statistics.mean(all_response_times['vector'])
        avg_hybrid = statistics.mean(all_response_times['hybrid'])
        
        print("\n📌 Performance:")
        if avg_vector < avg_hybrid:
            print(f"   Vector search is faster by {(avg_hybrid - avg_vector):.3f}s on average.")
            print(f"   Consider vector search for latency-critical applications.")
        else:
            print(f"   Hybrid search is faster by {(avg_vector - avg_hybrid):.3f}s on average.")
    
    if all_overlaps:
        avg_overlap = statistics.mean([o["overlap_percentage"] for o in all_overlaps])
        print(f"\n📌 Result Agreement:")
        if avg_overlap > 70:
            print(f"   High overlap ({avg_overlap:.1f}%) suggests both methods agree on most results.")
        elif avg_overlap > 40:
            print(f"   Moderate overlap ({avg_overlap:.1f}%) - hybrid search provides different perspective.")
            print(f"   Consider hybrid search for more diverse results.")
        else:
            print(f"   Low overlap ({avg_overlap:.1f}%) - methods differ significantly.")
            print(f"   Hybrid search may catch lexical matches that vector search misses.")
    
    print(f"\n📌 Use Cases:")
    print(f"   • Vector Search: Best for semantic/conceptual queries")
    print(f"   • Hybrid Search: Best for queries with specific keywords or technical terms")
    print(f"   • Hybrid Search: Recommended when you need both semantic + lexical matching")
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        # Check if API is available
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        print(f"✓ API is healthy and ready at {API_BASE_URL}")
        
        run_comparison()
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: Cannot connect to API at {API_BASE_URL}")
        print(f"   Make sure the server is running with: python -m app.main")
        print(f"   Error details: {e}")

