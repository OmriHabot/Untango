# Changelog

## v1.2.0 - Absolute Scoring Thresholds ⚖️

### 🎯 New Feature: Quality-Based Result Filtering

**Absolute Scoring Thresholds**
- ✅ Vector similarity threshold filtering (0.0-1.0 range)
- ✅ BM25 score threshold filtering (0.0+ range)
- ✅ Pre-RRF filtering to prevent irrelevant results
- ✅ Prevents "best of the worst" ranking problem
- ✅ Optional per-query configuration

**How It Works**
1. Documents are scored by both vector and BM25 methods
2. Documents below absolute thresholds are filtered out **before** RRF fusion
3. Only high-quality documents participate in final ranking
4. Empty results signal no relevant documents found (better than misleading results)

**API Changes**
- Enhanced `/query` endpoint with `vector_similarity_threshold` parameter
- Enhanced `/query-hybrid` endpoint with both vector and BM25 thresholds
- Enhanced `/query-db` (RAG) endpoint with threshold support
- All threshold parameters are optional (backwards compatible)

**New Request Fields**
```json
{
  "query": "authentication system",
  "n_results": 10,
  "vector_similarity_threshold": 0.5,  // Optional: 0.0-1.0
  "bm25_score_threshold": 5.0          // Optional: 0.0+
}
```

**New Response Fields**
Hybrid search results now include raw scores for debugging:
- `similarity`: Raw vector similarity score (0.0-1.0)
- `bm25_score`: Raw BM25 relevance score
- `distance`: Vector distance (1 - similarity)

**Enhanced Search Functions**
- `perform_vector_search()`: Added similarity threshold filtering
- `perform_hybrid_search()`: Added dual threshold filtering with logging
- `distance_to_similarity()`: New helper for distance-to-similarity conversion

**Documentation**
- 📄 New: `THRESHOLD_FILTERING.md` - Comprehensive threshold guide
- 📄 Updated: `SEARCH_COMPARISON_GUIDE.md` - Threshold testing instructions
- 📄 Updated: `compare_search_endpoints.py` - Threshold support added

**Logging Improvements**
- Threshold filtering statistics logged at INFO level
- Shows how many results passed each threshold
- Warns when no results pass thresholds

**Configuration Guidelines**
- Vector: 0.5-0.7 recommended for moderate filtering
- BM25: 3.0-10.0 recommended for moderate filtering
- Start conservative, adjust based on results
- Different thresholds for different query types

**Use Cases**
- ✅ Large, diverse codebases (prevent irrelevant results)
- ✅ High-precision requirements (only relevant results)
- ✅ RAG pipelines (better context quality)
- ✅ Production systems (consistent quality)

**Benefits**
- Higher precision (more relevant results)
- Clear signal when no relevant results exist
- Better RAG answer quality
- Reduced noise in search results

### 🔧 Technical Details

**Threshold Filtering Architecture**
```
Query → Vector Search → Threshold Filter ─┐
                                           ├─→ RRF Fusion → Results
Query → BM25 Search → Threshold Filter ───┘
```

**Score Conversion**
- Cosine distance → similarity: `1 - distance`
- L2 distance → similarity: `1 / (1 + distance)`
- Inner product: Already a similarity score

**Backwards Compatibility**
- ✅ All threshold parameters are optional
- ✅ Default behavior unchanged (no filtering)
- ✅ Existing API calls continue to work

### 📊 Testing & Comparison

**Updated Comparison Script**
- Configurable thresholds at script level
- Shows raw scores in output (when available)
- Threshold status displayed in summary
- Easy A/B testing (with vs. without thresholds)

**Example Threshold Configuration**
```python
# In compare_search_endpoints.py
VECTOR_SIMILARITY_THRESHOLD = 0.5
BM25_SCORE_THRESHOLD = 5.0
```

---

## v1.1.0 - Complete RAG Pipeline

### 🎉 New Feature: End-to-End RAG Platform

**New Endpoint: `/query-db`**
- ✅ Complete RAG pipeline combining retrieval + generation
- ✅ Hybrid search (vector + BM25) for intelligent chunk retrieval
- ✅ Confidence-based filtering (default threshold: 0.2)
- ✅ Context-aware prompt construction
- ✅ GCP Vertex AI integration for answer generation
- ✅ Comprehensive response with chunks, scores, and token usage

**New Models Added**
- `RAGQueryRequest` - Complete RAG query with configurable parameters
- `RAGQueryResponse` - Detailed response with answer and retrieved chunks
- `RetrievedChunk` - Chunk metadata with vector, BM25, and combined scores

**Usage Example**
```bash
curl -X POST "http://localhost:8001/query-db" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does authentication work?",
    "n_results": 10,
    "confidence_threshold": 0.2,
    "model": "gemini-2.0-flash-exp"
  }'
```

**Response includes:**
- AI-generated answer based on retrieved context
- All chunks used (with vector, BM25, combined scores)
- Token usage and estimated cost
- Metadata for each chunk (file, type, location)

**Test Suite Added**
- `test_rag_pipeline.py` - Comprehensive test script
- Tests: health → ingest → hybrid search → RAG query
- Demonstrates complete end-to-end workflow

### 📚 Documentation Updates
- README updated with `/query-db` endpoint documentation
- Usage examples for complete RAG workflow
- Response structure documented

---

## v1.0.0 - Production Ready Release

### 🎉 Major Changes

**Codebase Cleanup**
- Removed 5 extra documentation files (consolidated into README)
- Removed 2 example/test files (functionality moved to API)
- Deleted `gcp_utils.py` (functionality inlined)
- Streamlined project structure

**Production-Ready API**
- ✅ Added proper Pydantic request/response models
- ✅ Type-safe endpoints with validation
- ✅ Renamed `/test-inference` → `/inference` (proper endpoint naming)
- ✅ Request body validation (no URL parameters for complex data)
- ✅ Comprehensive error handling
- ✅ API documentation with FastAPI auto-docs

**New Models Added**
- `InferenceRequest` - Validated inference requests
- `InferenceResponse` - Structured inference responses  
- `TokenUsage` - Token usage tracking
- `HealthResponse` - Health check responses
- `SearchResult` - Search result structure

**Updated Endpoints**
- `POST /inference` - Production-ready AI inference (was `/test-inference`)
- `GET /health` - Returns structured `HealthResponse`
- All endpoints now have proper type hints and validation

**Improved README**
- Concise, production-focused documentation
- Quick start guide
- Clear API endpoint table
- Usage examples with curl
- Troubleshooting section
- Security best practices

### 📦 Current Structure

```
Untango/
├── app/                        # Clean, organized app structure
│   ├── __init__.py            # Package init
│   ├── main.py                # FastAPI app (229 lines)
│   ├── models.py              # Pydantic models (68 lines)
│   ├── database.py            # ChromaDB client (100 lines)
│   ├── chunker.py             # Code chunking (122 lines)
│   └── search.py              # Hybrid search (108 lines)
├── docker-compose.yaml         # Docker configuration
├── Dockerfile                  # Container build
├── README.md                   # Main documentation
├── requirements.txt            # Python dependencies
└── service-account-key.json   # GCP credentials (gitignored)
```

### 🗑️ Removed Files

- `GCP_SETUP.md` (info moved to README)
- `QUICK_START_GCP.md` (consolidated)
- `RUN_TEST_INFERENCE.md` (consolidated)
- `example_vertex_ai.py` (use `/inference` endpoint)
- `test_inference.py` (use `/inference` endpoint)
- `app/gcp_utils.py` (inlined into main.py)

### 🔒 Security

- All credentials in gitignore
- Read-only volume mounts in Docker
- No hardcoded secrets
- Type-safe validation prevents injection

### ✅ Quality Checks

- ✅ No linter errors
- ✅ All imports working
- ✅ Type hints validated
- ✅ Production-ready error handling
- ✅ Comprehensive API documentation

### 📚 Documentation

Single source of truth: `README.md`
- API endpoints table
- Quick start guide
- Configuration reference
- Usage examples
- Troubleshooting

### 🚀 Usage

**Start services:**
```bash
docker-compose up --build -d
```

**Test inference:**
```bash
curl -X POST "http://localhost:8001/inference" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "model": "gemini-2.0-flash-exp"}'
```

**Interactive docs:**
http://localhost:8001/docs

