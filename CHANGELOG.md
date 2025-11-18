# Changelog

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

