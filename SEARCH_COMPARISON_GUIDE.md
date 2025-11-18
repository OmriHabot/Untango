# Search Endpoint Comparison Guide

This guide explains how to use the `compare_search_endpoints.py` script to evaluate and compare the performance of vector search vs. hybrid search endpoints.

## Overview

The script compares two search endpoints:
- **`/query`**: Pure vector similarity search using embeddings
- **`/query-hybrid`**: Hybrid search combining vector similarity (dense) and BM25 (sparse) keyword search using Reciprocal Rank Fusion (RRF)

## Prerequisites

1. Ensure the RAG backend server is running:
   ```bash
   python -m app.main
   ```

2. Ensure you have data ingested into the ChromaDB collection. If not, ingest some code first using the `/ingest` endpoint.

3. Install dependencies (if not already done):
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Simply run the script:
```bash
python compare_search_endpoints.py
```

### Customizing Test Queries

Edit the `TEST_QUERIES` list in the script to test with your own queries:

```python
TEST_QUERIES = [
    "authentication and user login",
    "database connection",
    "your custom query here",
]
```

### Changing Configuration

You can modify these constants at the top of the script:

```python
API_BASE_URL = "http://localhost:8001"  # Change if running on different host/port
N_RESULTS = 10  # Number of results to retrieve per query
```

## Metrics Calculated

### Per-Query Metrics

For each test query, the script calculates:

1. **Response Time**: Time taken for each endpoint to respond
2. **Result Count**: Number of results returned
3. **Result Overlap**: 
   - Overlap count and percentage
   - Jaccard similarity (intersection over union)
   - Average rank difference for overlapping results
   - Unique results from each method

### Overall Statistics

Aggregated across all queries:

1. **Response Time Comparison**:
   - Mean, median, min, max response times
   - Performance difference between methods

2. **Result Overlap Statistics**:
   - Average overlap percentage
   - Average Jaccard similarity
   - Average rank difference
   - Queries with 100% overlap
   - Queries with <50% overlap

3. **Result Diversity Analysis**:
   - Unique files retrieved
   - Unique chunk types
   - Unique repositories
   - Type distribution

4. **Score Distribution Analysis**:
   - Vector search distance statistics (lower = more similar)
   - Hybrid search combined score statistics (higher = more relevant)
   - RRF component breakdowns (dense vs. BM25 contributions)

## Understanding the Output

### Response Time

```
Response Time Comparison
   Vector Search:
      • Mean: 0.145s
      • Median: 0.142s
   
   Hybrid Search:
      • Mean: 0.312s
      • Median: 0.305s
   
   Δ Hybrid is 0.167s (slower, 115.2%)
```

This shows hybrid search takes longer due to the additional BM25 computation.

### Result Overlap

```
Result Overlap
   • Overlap: 7/10 (70.0%)
   • Jaccard similarity: 0.583
   • Avg rank difference: 2.3 positions
```

- **Overlap**: 7 out of 10 results are the same between both methods
- **Jaccard similarity**: 0.583 means moderate agreement considering all unique results
- **Avg rank difference**: Overlapping results differ by ~2 positions on average

### Score Distributions

**Vector Search** uses **distance** (Euclidean distance in embedding space):
- Lower values = more similar
- Typical range: 0.1 to 2.0

**Hybrid Search** uses **combined_score** (RRF fusion):
- Higher values = more relevant
- Sum of dense RRF and BM25 RRF scores
- Typical range: 0.01 to 0.06

## Interpreting Results

### When Vector Search Performs Better

- Queries are conceptual/semantic ("what handles authentication?")
- Looking for functionality described in different words
- Need to find semantically similar code even if keywords differ

### When Hybrid Search Performs Better

- Queries contain specific technical terms or identifiers
- Looking for exact function/class/variable names
- Need both semantic understanding AND keyword matching
- Want to catch lexical matches that embeddings might miss

### High Overlap (>70%)

Both methods agree on most results. Consider:
- Using vector search for speed
- Using hybrid search for slightly better ranking

### Low Overlap (<40%)

Methods find significantly different results. Consider:
- Using hybrid search for more comprehensive coverage
- Hybrid search catches keyword matches that vector search misses
- May indicate that semantic and lexical signals provide complementary information

## Example Output Interpretation

```
📌 Result Agreement:
   Moderate overlap (55.2%) - hybrid search provides different perspective.
   Consider hybrid search for more diverse results.

📌 Use Cases:
   • Vector Search: Best for semantic/conceptual queries
   • Hybrid Search: Best for queries with specific keywords or technical terms
   • Hybrid Search: Recommended when you need both semantic + lexical matching
```

## Customization Ideas

### Add Your Own Metrics

Extend the script with custom metrics:

```python
def calculate_custom_metric(results):
    # Your custom metric logic
    pass
```

### Test with Ground Truth

If you have known relevant results for queries:

```python
GROUND_TRUTH = {
    "authentication": ["auth.py", "login.py"],
    # ...
}

def calculate_precision_recall(results, query):
    relevant = GROUND_TRUTH.get(query, [])
    # Calculate precision and recall
    pass
```

### Export Results to JSON/CSV

Add export functionality:

```python
import json

# After collecting results
with open("comparison_results.json", "w") as f:
    json.dump({
        "vector_results": all_vector_results,
        "hybrid_results": all_hybrid_results,
        "statistics": {...}
    }, f, indent=2)
```

## Troubleshooting

### API Connection Error

```
❌ Error: Cannot connect to API at http://localhost:8001
```

**Solution**: Make sure the server is running:
```bash
python -m app.main
```

### No Results Returned

If queries return 0 results:
1. Check that data is ingested: `GET http://localhost:8001/health`
2. Verify collection has documents
3. Try broader queries

### Timeout Errors

If requests timeout:
1. Increase timeout in script: `timeout=60`
2. Check server logs for performance issues
3. Consider reducing `N_RESULTS` or `TEST_QUERIES`

## Best Practices

1. **Test with Representative Queries**: Use real queries from your use case
2. **Vary Query Types**: Mix semantic and keyword-based queries
3. **Sufficient Data**: Ensure collection has enough documents for meaningful comparison
4. **Multiple Runs**: Run multiple times to account for variance
5. **Monitor Server**: Watch server logs during comparison for insights

## Further Reading

- [Reciprocal Rank Fusion (RRF)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [ChromaDB Hybrid Search Documentation](https://docs.trychroma.com/)
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25)

