# RAG Backend with ChromaDB & Vertex AI

A production-ready FastAPI backend for Retrieval-Augmented Generation (RAG) with intelligent Python code chunking, hybrid search, and Vertex AI integration.

## Features

- **Intelligent Code Chunking**: AST-based parsing for Python code (functions, classes, methods)
- **Hybrid Search**: Combines vector similarity with BM25 keyword matching
- **Code-Aware Tokenization**: Handles camelCase, snake_case, and special characters
- **Vertex AI Integration**: Generate AI responses using Google's Gemini models
- **Production Ready**: Type-safe models, proper error handling, comprehensive API docs

## Quick Start

### 1. Start the Services

```bash
docker-compose up --build -d
```

### 2. Verify Health

```bash
curl http://localhost:8001/health
```

### 3. Test the API

Visit: http://localhost:8001/docs for interactive API documentation

## Apple Silicon Support (Development)

To leverage **MPS (Metal Performance Shaders)** for hardware-accelerated embeddings on macOS, you must run the backend **natively** (outside Docker) while keeping the database containerized. Docker on macOS cannot access the GPU.

### 1. Start Dependencies (ChromaDB)
Use the dedicated MPS compose file to run only the database:
```bash
docker-compose -f docker-compose.mps.yaml up -d
```

### 2. Run Backend Natively
Use the helper script to start the backend with the correct environment variables:
```bash
# Ensure your virtual environment is active
source .venv/bin/activate

# Run the backend
./run_backend_mps.sh
```

The backend will automatically detect MPS availability and use it for embedding generation.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Ingest and chunk Python code |
| `/query` | POST | Vector similarity search |
| `/query-hybrid` | POST | Hybrid search (vector + BM25) |
| `/query-db` | POST | **Complete RAG pipeline**: Retrieve + Generate with Vertex AI |
| `/inference` | POST | Generate AI responses via Vertex AI |
| `/health` | GET | Service health check |
| `/collection` | DELETE | Reset the collection |

## Project Structure

```
app/
├── __init__.py       # Package initialization
├── main.py          # FastAPI application & routes
├── models.py        # Pydantic request/response models
├── database.py      # ChromaDB client & connection
├── chunker.py       # AST-based code chunking
└── search.py        # Hybrid search utilities
```

## Usage Examples

### Ingest Code

```bash
curl -X POST "http://localhost:8001/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def hello():\n    return \"world\"",
    "filepath": "example.py",
    "repo_name": "my-repo"
  }'
```

### Search Code

```bash
curl -X POST "http://localhost:8001/query-hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query": "authentication function", "n_results": 5}'
```

### RAG Query (Retrieve + Generate)

The `/query-db` endpoint is the **complete RAG pipeline** that:
1. Retrieves relevant chunks using hybrid search
2. Filters by confidence threshold (default: 0.2)
3. Builds a context-aware prompt
4. Generates an answer using Vertex AI

```bash
curl -X POST "http://localhost:8001/query-db" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does authentication work in this codebase?",
    "n_results": 10,
    "confidence_threshold": 0.2,
    "model": "gemini-2.5-flash"
  }'
```

Response includes:
- **answer**: AI-generated response
- **retrieved_chunks**: All chunks used for context
- **chunks_used**: Number of chunks above confidence threshold
- **usage**: Token usage and cost information

### Generate AI Response (Direct)

For direct inference without retrieval:

```bash
curl -X POST "http://localhost:8001/inference" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain vector databases",
    "model": "gemini-2.5-flash"
  }'
```

## Google Cloud / Vertex AI Setup

This project uses **Google Cloud Service Account key file** with **Application Default Credentials (ADC)** for authentication. This approach is recommended for workloads running outside of Google Cloud (local machine, other cloud providers, CI/CD pipelines).

### Step-by-Step: Get Your Service Account Key

#### 1. Create a Service Account 🔑

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Select your project (or create one)
3. Navigate to **IAM & Admin** → **Service Accounts**
4. Click **+ CREATE SERVICE ACCOUNT**
5. Enter a name (e.g., `vertex-ai-docker-sa`)
6. Click **CREATE AND CONTINUE**

#### 2. Grant Necessary Roles

Grant the service account the following roles:

- **Required:** `Vertex AI User` (`roles/aiplatform.user`) - For Vertex AI API calls
- **Optional:** `Storage Object Admin` (`roles/storage.objectAdmin`) - If accessing Cloud Storage for models/data

1. Click **ADD ANOTHER ROLE** to add multiple roles
2. Search for and select each role
3. Click **CONTINUE**, then **DONE**

#### 3. Create and Download the JSON Key

⚠️ **Important:** Treat this file as highly sensitive, like a password.

1. Click on the service account you just created
2. Go to the **KEYS** tab
3. Click **ADD KEY** → **Create new key**
4. Select **JSON** format
5. Click **CREATE**
6. The key file will download automatically (e.g., `your-project-xxxxx.json`)

#### 4. Setup the Key File

1. **Move** the downloaded file to your project root:
   ```bash
   mv ~/Downloads/your-project-xxxxx.json /path/to/Untango/
   ```

2. **Rename** it to `service-account-key.json`:
   ```bash
   cd /path/to/Untango
   mv your-project-xxxxx.json service-account-key.json
   ```

3. **Verify** it's in the correct location:
   ```bash
   ls -la service-account-key.json
   # Should show the file in your project root
   ```

4. **Update** `docker-compose.yaml` with your actual project ID:
   ```yaml
   environment:
     - GOOGLE_CLOUD_PROJECT=your-actual-project-id
     - GOOGLE_CLOUD_LOCATION=global  # or your preferred region
   ```

#### 5. Enable Vertex AI API

1. Go to [Vertex AI in Cloud Console](https://console.cloud.google.com/vertex-ai)
2. Click **ENABLE** if the API is not already enabled
3. Wait for activation (usually 1-2 minutes)

### How It Works

The Docker configuration automatically sets up authentication using:

1. **Volume Mount** (read-only for security):
   ```yaml
   volumes:
     - ./service-account-key.json:/app/gcp-key.json:ro
   ```

2. **Environment Variables** (for Application Default Credentials):
   ```yaml
   environment:
     - GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-key.json
     - GOOGLE_CLOUD_PROJECT=your-project-id
   ```

The Google Cloud client libraries automatically detect and use these credentials via the `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

### Security Best Practices

✅ **DO:**
- Keep `service-account-key.json` in project root (already in `.gitignore`)
- Use read-only mount (`:ro`) in Docker
- Rotate keys periodically in GCP Console
- Limit service account permissions to only what's needed

❌ **DON'T:**
- Commit the key file to version control
- Share the key file publicly
- Hardcode credentials in code
- Use overly permissive roles

### Alternative: Workload Identity (For Cloud Deployments)

If deploying to **Google Kubernetes Engine (GKE)** or **Cloud Run**, consider using **Workload Identity** instead. This approach:
- Eliminates key file management
- Uses short-lived credentials
- More secure for production cloud deployments
- Links Kubernetes/Cloud Run service accounts to GCP service accounts

**Note:** The current setup with key files is ideal for local development and external deployments.

### Configuration

The Docker setup automatically configures credentials via environment variables in `docker-compose.yaml`:

```yaml
environment:
  - GOOGLE_APPLICATION_CREDENTIALS=/app/gcp-key.json
  - GOOGLE_CLOUD_PROJECT=your-project-id
  - GOOGLE_CLOUD_LOCATION=global
```

### Local Development

Set environment variables before running locally:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="./service-account-key.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_LOCATION="global"

source .venv/bin/activate
python -m app.main
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `CHROMA_HOST` | ChromaDB hostname | `localhost` |
| `CHROMA_PORT` | ChromaDB port | `8000` |
| `CHROMA_COLLECTION_NAME` | Collection name | `python_code_chunks` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP key file | - |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | - |
| `GOOGLE_CLOUD_LOCATION` | GCP region | `global` |

### Docker Compose

The `docker-compose.yaml` runs two services:
- **chromadb**: Vector database (port 8000)
- **rag-backend**: FastAPI application (port 8001)

## Development

### Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Tests

```bash
# Check health
curl http://localhost:8001/health

# Test inference (requires GCP setup)
curl -X POST "http://localhost:8001/inference" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Test"}'
```

## Security

- ✅ `service-account-key.json` is in `.gitignore`
- ✅ Credentials mounted read-only in Docker (`:ro`)
- ✅ No hardcoded secrets
- ✅ Type-safe request validation via Pydantic

## Troubleshooting

### ChromaDB Connection Failed
```bash
# Check if ChromaDB is running
docker ps | grep chromadb
docker-compose logs chromadb
```

### GCP Credentials Error
```bash
# Verify credentials file exists
ls -la service-account-key.json

# Check environment in container
docker exec rag-backend env | grep GOOGLE
```

### Port Already in Use
```bash
# Stop existing containers
docker-compose down

# Check for processes using ports
lsof -i :8000
lsof -i :8001
```

## License

MIT

## Support

- API Documentation: http://localhost:8001/docs
- Health Check: http://localhost:8001/health
- ReDoc: http://localhost:8001/redoc
