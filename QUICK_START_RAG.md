# Quick Start: RAG Query Pipeline

This guide shows you how to use the complete RAG platform with the new `/query-db` endpoint.

## Prerequisites

Make sure your services are running:

```bash
docker-compose up -d
```

Verify health:
```bash
curl http://localhost:8001/health
```

## Step-by-Step RAG Pipeline

### Step 1: Ingest Code

First, ingest some code into the ChromaDB vector store:

```bash
curl -X POST "http://localhost:8001/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def authenticate_user(username, password):\n    # Auth logic here\n    return True",
    "filepath": "auth/login.py",
    "repo_name": "my-app"
  }'
```

### Step 2: Query with RAG

Now use the complete RAG pipeline to get AI-powered answers:

```bash
curl -X POST "http://localhost:8001/query-db" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does user authentication work in this codebase?",
    "n_results": 10,
    "confidence_threshold": 0.2,
    "model": "gemini-2.0-flash-exp"
  }'
```

**What happens:**
1. ✅ Performs hybrid search (vector + BM25) to retrieve top 10 chunks
2. ✅ Filters chunks with combined confidence > 0.2
3. ✅ Builds a context-aware prompt with retrieved chunks
4. ✅ Sends prompt + question to Vertex AI
5. ✅ Returns AI answer + chunk details + token usage

### Step 3: Review the Response

```json
{
  "status": "success",
  "query": "How does user authentication work?",
  "retrieved_chunks": [
    {
      "id": "auth/login.py::func::authenticate_user::0",
      "content": "def authenticate_user(...)...",
      "metadata": {
        "filepath": "auth/login.py",
        "chunk_type": "function",
        "function_name": "authenticate_user"
      },
      "vector_score": 0.85,
      "bm25_score": 0.73,
      "combined_score": 0.79
    }
  ],
  "chunks_used": 3,
  "answer": "Based on the retrieved code, the authentication works by...",
  "model": "gemini-2.0-flash-exp",
  "usage": {
    "input_tokens": 450,
    "output_tokens": 120,
    "total_tokens": 570
  }
}
```

## Python Test Script

Run the comprehensive test suite:

```bash
python test_rag_pipeline.py
```

This script will:
- ✅ Check service health
- ✅ Ingest sample authentication code
- ✅ Test hybrid search
- ✅ Test complete RAG query
- ✅ Display results and token usage

## Parameter Reference

### RAGQueryRequest

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | string | **required** | Your question to answer |
| `n_results` | int | 10 | Number of chunks to retrieve |
| `confidence_threshold` | float | 0.2 | Minimum combined score (0.0-1.0) |
| `model` | string | "gemini-2.0-flash-exp" | Vertex AI model to use |

### Response Scores

- **vector_score**: Semantic similarity (0-1, higher = more similar)
- **bm25_score**: Keyword matching (0-1, higher = better match)
- **combined_score**: Average of both (0-1, higher = more relevant)

## Tips

### Adjust Confidence Threshold

Lower threshold (0.3) = more chunks, potentially less relevant:
```bash
curl ... -d '{"query": "...", "confidence_threshold": 0.3}'
```

Higher threshold (0.7) = fewer chunks, higher quality:
```bash
curl ... -d '{"query": "...", "confidence_threshold": 0.7}'
```

### Retrieve More Candidates

Get more chunks for better context:
```bash
curl ... -d '{"query": "...", "n_results": 20}'
```

### Different Models

Use different Vertex AI models:
```bash
curl ... -d '{"query": "...", "model": "gemini-1.5-pro"}'
```

## Common Patterns

### Q&A on Codebase
```bash
curl -X POST "http://localhost:8001/query-db" \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the login function do?"}'
```

### Code Understanding
```bash
curl -X POST "http://localhost:8001/query-db" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain how error handling works in this code"}'
```

### Security Analysis
```bash
curl -X POST "http://localhost:8001/query-db" \
  -H "Content-Type: application/json" \
  -d '{"query": "Are there any security vulnerabilities in the authentication?"}'
```

## Troubleshooting

### "No chunks found with confidence > X"

- Lower the `confidence_threshold`
- Ingest more relevant code
- Try a more general query

### GCP Not Configured

Make sure environment variables are set:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="./service-account-key.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

See `README.md` for full GCP setup instructions.

## Interactive Documentation

Visit http://localhost:8001/docs to try the API interactively with Swagger UI.

