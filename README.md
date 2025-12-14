# RAG Backend with ChromaDB & Vertex AI

A production-ready FastAPI backend for Retrieval-Augmented Generation (RAG) with intelligent Python code chunking, hybrid search, and Vertex AI integration.

## Features

- **Intelligent Code Chunking**: AST-based parsing for Python code (functions, classes, methods)
- **Hybrid Search**: Combines vector similarity with BM25 keyword matching
- **Code-Aware Tokenization**: Handles camelCase, snake_case, and special characters
- **Vertex AI Integration**: Generate AI responses using Google's Gemini models
- **Production Ready**: Type-safe models, proper error handling, comprehensive API docs

## Quick Start

Get the full stack up and running in minutes. You will need **Docker**, **Node.js** (v18+), and a **Google Cloud Project**.

### 1. Setup Google Cloud & Vertex AI

To use the AI features, you need a Google Cloud Service Account.

#### A. Create a Service Account
1.  Go to [Google Cloud Console > IAM & Admin > Service Accounts](https://console.cloud.google.com/iam-admin/serviceaccounts).
2.  Click **+ CREATE SERVICE ACCOUNT**.
3.  Name it (e.g., `vertex-ai-sa`) and click **CREATE AND CONTINUE**.
4.  Grant the **Vertex AI User** role (`roles/aiplatform.user`).
5.  Click **DONE**.

#### B. Download Key & Configure Project
1.  Click on your new service account -> **KEYS** tab -> **ADD KEY** -> **Create new key** (JSON).
2.  Save the file as `service-account-key.json` in the root of this project (it is git-ignored).
3.  **Crucial Step:** Open `service-account-key.json` and copy the value of `"project_id"`.
4.  Open `docker-compose.yaml` and update the `GOOGLE_CLOUD_PROJECT` environment variable with this ID:
    ```yaml
    environment:
      - GOOGLE_CLOUD_PROJECT=your-copied-project-id  # <--- PASTE HERE
    ```

#### C. Enable Vertex AI API
*   Visit [Vertex AI in Cloud Console](https://console.cloud.google.com/vertex-ai) and click **ENABLE** if not already active.

### 2. Start the Backend
Spin up ChromaDB and the FastAPI backend:
```bash
docker-compose up --build -d
```
> **Note:** The backend runs on `http://localhost:8001` and ChromaDB on `http://localhost:8000`.

### 3. Start the Frontend
In a new terminal, launch the React UI:
```bash
cd frontend
pnpm install && pnpm run dev
```
The UI will open at `http://localhost:5173`.

### 4. Verify System Status
Visit `http://localhost:8001/health` to confirm the backend is connected to the vector database.

---

## Ingesting Repositories

Untango supports both remote GitHub repositories and local project directories.

### Option A: Remote GitHub Repository
1. Open the web UI (`http://localhost:5173`).
2. Click **"Ingest Repository"** in the sidebar.
3. Select the **GitHub** tab.
4. Paste the repository URL (e.g., `https://github.com/fastapi/fastapi`) and optionally specify a branch.
5. Click **Ingest**. The system will clone, parse, and chunk the codebase in the background.

### Option B: Local Repository (CLI)
To ingest a project from your local machine (e.g., for privacy or development speed), use the bundled CLI tool.

**One-time Upload:**
```bash
# Run from the project root
python scripts/untango_local.py /path/to/my-project --server http://localhost:8001
```

**Continuous Sync (Watch Mode):**
```bash
python scripts/untango_local.py /path/to/my-project --server http://localhost:8001 --watch
```
*The CLI will upload changes automatically whenever you save a file.*


## Apple Silicon Support (Development)

To leverage **MPS (Metal Performance Shaders)** for hardware-accelerated embeddings on macOS, you must run the backend **natively** (outside Docker) while keeping the database containerized.

### 1. Start Dependencies (ChromaDB)
```bash
docker-compose -f docker-compose.mps.yaml up -d
```

### 2. Run Backend Natively
```bash
# Ensure your virtual environment is active
source .venv/bin/activate

# Run the backend
./run_backend_mps.sh
```

## Running with Remote ChromaDB (GCP)

To run the backend connected to a remote ChromaDB instance on Google Cloud Platform:

### 1. Configuration
Set the connection details for your remote ChromaDB instance:
```bash
export CHROMA_HOST="your-chroma-service-url.a.run.app"
export CHROMA_PORT="443"
```

### 2. Run the Container
```bash
docker-compose -f docker-compose.gcp.yaml up --build
```
This starts the `rag-backend` service with Google IAM authentication enabled.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ingest` | POST | Ingest and chunk Python code |
| `/query` | POST | Vector similarity search |
| `/query-hybrid` | POST | Hybrid search (vector + BM25) |
| `/query-db` | POST | **Complete RAG pipeline**: Retrieve + Generate |
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

### RAG Query (Retrieve + Generate)
The `/query-db` endpoint is the **complete RAG pipeline**:
```bash
curl -X POST "http://localhost:8001/query-db" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does authentication work in this codebase?",
    "n_results": 10,
    "confidence_threshold": 0.2,
    "model": "gemini-3-pro-preview"
  }'
```

### Review Response
Response includes:
- **answer**: AI-generated response
- **retrieved_chunks**: All chunks used for context
- **chunks_used**: Number of chunks above confidence threshold
- **usage**: Token usage and cost information

## Configuration

### Environment Variables
| Variable | Description |
|----------|-------------|
| `CHROMA_HOST` | ChromaDB hostname |
| `CHROMA_PORT` | ChromaDB port |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP key file |
| `GOOGLE_CLOUD_PROJECT` | GCP project ID |
| `GOOGLE_CLOUD_LOCATION` | GCP region (default: global) |

### Docker Compose
The `docker-compose.yaml` runs:
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
```

## Security
- **Credentials**: `service-account-key.json` is git-ignored and mounted read-only in Docker.
- **Secrets**: No hardcoded secrets in the codebase.


## License

MIT

## Support

- API Documentation: http://localhost:8001/docs
- Health Check: http://localhost:8001/health
- ReDoc: http://localhost:8001/redoc
