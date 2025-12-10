# RAG Platform Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         RAG Platform                             │
│                    (FastAPI Backend)                             │
└─────────────────────────────────────────────────────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        │                      │                      │
        ▼                      ▼                      ▼
┌──────────────┐     ┌──────────────┐      ┌─────────────────┐
│              │     │              │      │                 │
│  ChromaDB    │     │ Hybrid Search│      │  Vertex AI      │
│  Vector DB   │     │ (Vec + BM25) │      │  (Gemini)       │
│              │     │              │      │                 │
└──────────────┘     └──────────────┘      └─────────────────┘
```

## Data Flow: `/query-db` Endpoint

```
User Query
    │
    ├─► "How does authentication work?"
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Hybrid Search Retrieval                            │
│  ┌─────────────┐         ┌─────────────┐                   │
│  │ Vector      │         │ BM25        │                   │
│  │ Search      │         │ Keyword     │                   │
│  │ (Semantic)  │         │ Search      │                   │
│  └──────┬──────┘         └──────┬──────┘                   │
│         │                       │                           │
│         └───────────┬───────────┘                           │
│                     │                                       │
│              Combined Ranking                               │
│         (50% vector + 50% BM25)                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Confidence Filtering                               │
│                                                              │
│  Filter: combined_score > 0.2                               │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Chunk 1     │  │ Chunk 2     │  │ Chunk 3     │        │
│  │ Score: 0.79 │  │ Score: 0.65 │  │ Score: 0.52 │        │
│  │ ✓ Include   │  │ ✓ Include   │  │ ✓ Include   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐                          │
│  │ Chunk 4     │  │ Chunk 5     │                          │
│  │ Score: 0.43 │  │ Score: 0.18 │                          │
│  │ ✓ Include   │  │ ✗ Exclude   │                          │
│  └─────────────┘  └─────────────┘                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Context Construction                               │
│                                                              │
│  Build Prompt:                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ You are a helpful AI assistant.                      │  │
│  │                                                       │  │
│  │ RETRIEVED CONTEXT:                                   │  │
│  │ --- Chunk 1 (confidence: 0.79) ---                   │  │
│  │ File: auth/login.py                                  │  │
│  │ def authenticate_user(...):                          │  │
│  │     ...                                              │  │
│  │                                                       │  │
│  │ --- Chunk 2 (confidence: 0.65) ---                   │  │
│  │ File: auth/password.py                               │  │
│  │ def hash_password(...):                              │  │
│  │     ...                                              │  │
│  │                                                       │  │
│  │ USER QUESTION:                                       │  │
│  │ How does authentication work?                        │  │
│  │                                                       │  │
│  │ ANSWER:                                              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 4: Vertex AI Inference                                │
│                                                              │
│  Model: gemini-3.0-flash                                │
│  Input: Prompt with context + question                      │
│                                                              │
│  ┌────────────────────────────────────────┐                │
│  │  Google Cloud Vertex AI                │                │
│  │  ┌──────────────────────────────────┐  │                │
│  │  │   Gemini Model Processing        │  │                │
│  │  │   • Context understanding        │  │                │
│  │  │   • Code analysis                │  │                │
│  │  │   • Answer generation            │  │                │
│  │  └──────────────────────────────────┘  │                │
│  └────────────────────────────────────────┘                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  Step 5: Response Assembly                                  │
│                                                              │
│  {                                                           │
│    "status": "success",                                     │
│    "query": "How does authentication work?",                │
│    "retrieved_chunks": [...],                               │
│    "chunks_used": 3,                                        │
│    "answer": "Authentication in this codebase...",          │
│    "model": "gemini-3.0-flash",                         │
│    "usage": {                                               │
│      "input_tokens": 450,                                   │
│      "output_tokens": 120,                                  │
│      "total_tokens": 570                                    │
│    }                                                         │
│  }                                                           │
└─────────────────────────────────────────────────────────────┘
```

## API Endpoints

### 1. `/ingest` - Code Ingestion
- **Purpose**: Chunk and store Python code
- **Process**: AST parsing → Semantic chunks → ChromaDB storage
- **Used by**: Initial data loading

### 2. `/query` - Vector Search Only
- **Purpose**: Pure semantic search
- **Process**: Query → Vector similarity → Top-K results
- **Used by**: Simple semantic queries

### 3. `/query-hybrid` - Hybrid Search
- **Purpose**: Combined semantic + keyword search
- **Process**: Vector search + BM25 → Score fusion → Ranked results
- **Used by**: Complex queries needing both semantic and lexical matching

### 4. `/query-db` - Complete RAG Pipeline ⭐
- **Purpose**: End-to-end RAG with AI generation
- **Process**: Hybrid search → Filter → Context build → LLM → Answer
- **Used by**: Question answering on codebase

### 5. `/inference` - Direct AI Call
- **Purpose**: Direct Vertex AI inference
- **Process**: Prompt → Vertex AI → Response
- **Used by**: General AI queries without retrieval

### 6. `/health` - Health Check
- **Purpose**: Service monitoring
- **Process**: Check ChromaDB + GCP config
- **Used by**: DevOps, monitoring

## Component Breakdown

### `app/main.py`
- FastAPI application
- All endpoint definitions
- Request routing and error handling

### `app/models.py`
- Pydantic models for type safety
- Request/response schemas
- Data validation

### `app/database.py`
- ChromaDB client management
- Collection operations
- Connection pooling

### `app/chunker.py`
- AST-based code parsing
- Intelligent chunking (functions, classes, methods)
- Metadata extraction

### `app/search.py`
- Vector search implementation
- BM25 keyword search
- Hybrid search score fusion

## Key Features

### 1. Intelligent Code Chunking
- AST-based parsing (not naive splitting)
- Function-level granularity
- Class and method extraction
- Preserves code context

### 2. Hybrid Search
- **Vector Search**: Semantic understanding
- **BM25 Search**: Keyword matching
- **Score Fusion**: 50/50 weighted average
- **Result**: Best of both worlds

### 3. Confidence Filtering
- Configurable threshold (default: 0.5)
- Only high-quality chunks used
- Reduces noise in LLM context
- Improves answer quality

### 4. Context-Aware Prompts
- Structured prompt with chunk metadata
- Clear separation of context and question
- Instructs LLM to use provided context
- Handles "can't find answer" cases

### 5. Production Ready
- Type-safe with Pydantic
- Comprehensive error handling
- Token usage tracking
- Cost estimation
- Auto-generated API docs

## Scoring System

### Vector Score (Semantic)
- Range: 0.0 to 1.0
- Based on: Cosine similarity of embeddings
- Higher = semantically more similar
- Good for: Understanding intent

### BM25 Score (Keyword)
- Range: 0.0 to 1.0 (normalized)
- Based on: Term frequency and document frequency
- Higher = better keyword match
- Good for: Exact terms, names

### Combined Score
```
combined_score = (vector_score * 0.5) + (bm25_score * 0.5)
```
- Balanced approach
- Catches both semantic and lexical matches
- Default: 50/50 weighting

## Example Use Cases

### 1. Codebase Q&A
```
Query: "How does password hashing work?"
→ Retrieves relevant auth code
→ AI explains the implementation
```

### 2. Feature Discovery
```
Query: "What authentication methods are available?"
→ Finds all auth-related functions
→ AI summarizes the options
```

### 3. Security Audit
```
Query: "Are there any SQL injection vulnerabilities?"
→ Retrieves database query code
→ AI analyzes for security issues
```

### 4. Onboarding
```
Query: "Explain the user registration flow"
→ Retrieves relevant functions
→ AI provides step-by-step explanation
```

## Technology Stack

- **Backend**: FastAPI (Python 3.13)
- **Vector DB**: ChromaDB with cosine similarity
- **Embeddings**: ChromaDB default (sentence-transformers)
- **Search**: BM25Okapi (rank-bm25)
- **LLM**: Google Vertex AI (Gemini models)
- **Deployment**: Docker + Docker Compose

## Configuration

See `README.md` for:
- Environment variables
- GCP setup
- Docker configuration
- Local development setup

## Performance Considerations

### Retrieval Speed
- ChromaDB: Fast vector similarity search
- BM25: Lightweight keyword matching
- Combined: Typically < 100ms for small collections

### LLM Latency
- Vertex AI: 1-3 seconds typical
- Depends on: prompt length, model, region
- Can optimize: Use smaller models, reduce context

### Cost
- Vector DB: Free (ChromaDB)
- Embeddings: Free (local)
- LLM: Pay per token (Vertex AI)
- Typical: $0.001 - $0.01 per query

## Security

- ✅ No hardcoded credentials
- ✅ GCP credentials via service account
- ✅ Read-only Docker mounts
- ✅ Input validation (Pydantic)
- ✅ Type safety throughout
- ✅ Error handling without info leakage

## Future Enhancements

Possible improvements:
- [ ] Multi-language support (not just Python)
- [ ] Custom embedding models
- [ ] Re-ranking with cross-encoders
- [ ] Streaming LLM responses
- [ ] Query history and caching
- [ ] User authentication
- [ ] Rate limiting
- [ ] Monitoring and metrics

