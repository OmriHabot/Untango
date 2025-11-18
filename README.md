# Untango

## RAG Backend with ChromaDB

A FastAPI-based Retrieval-Augmented Generation (RAG) backend that intelligently chunks Python code using AST parsing and provides hybrid search capabilities combining vector similarity and BM25 keyword search.

### Project Structure

```
app/
├── __init__.py      # Package initialization
├── main.py          # FastAPI routes and endpoints
├── models.py        # Pydantic request/response models
├── database.py      # ChromaDB client and collection setup
├── chunker.py       # AST-based code chunking logic
└── search.py        # Search utilities (vector + BM25 hybrid)
```

### Features

- **Intelligent Code Chunking**: Uses Python AST parsing to chunk code by functions, classes, and methods
- **Hybrid Search**: Combines vector similarity search with BM25 keyword matching for better retrieval
- **Code-Aware Tokenization**: Custom tokenization for code that handles camelCase, snake_case, and special characters
- **ChromaDB Integration**: Persistent vector storage with automatic embedding generation

### API Endpoints

- `POST /ingest` - Ingest Python code files and chunk them intelligently
- `POST /query` - Vector similarity search
- `POST /query-hybrid` - Hybrid search (vector + BM25)
- `GET /health` - Health check for ChromaDB connection
- `DELETE /collection` - Reset the collection (use with caution)

### Running the Application

```bash
# Start ChromaDB and the application using Docker Compose
docker-compose up

# Or run locally (requires ChromaDB running separately)
python -m app.main
```

The API will be available at `http://localhost:8001`

---

## GCP Sample Inference

This guide demonstrates how to run inference using Google Cloud Platform's Vertex AI with the Gemini API. The example includes token usage tracking and cost estimation.

### Prerequisites

1. **Install Google Cloud SDK (gcloud CLI)**
   - Download and install from [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
   - Verify installation: `gcloud --version`

2. **Authenticate with Application Default Credentials**
   ```zsh
   gcloud auth application-default login
   ```
   This command opens a browser window for authentication and sets up credentials for local development.

3. **Install Required Python Package**
   ```bash
   pip install google-genai
   ```

### Configuration

Before running inference, configure your GCP project settings:

- **Project ID**: Use a GCP project that has Vertex AI API enabled and contains credits/quota
- **Location**: Use `'global'` for the global endpoint (recommended for better availability), or specify a region like `'us-central1'` or `'us-east1'`

### Basic Usage

```python
YOUR_PROJECT_ID = 'PROJECTID-CONTAINING-CREDITS'
YOUR_LOCATION = 'global'

from google import genai

# Initialize the Vertex AI client
client = genai.Client(
    vertexai=True, 
    project=YOUR_PROJECT_ID, 
    location=YOUR_LOCATION,
)

# Specify the model (Gemini 2.5 Flash Lite Preview)
model = "gemini-2.5-flash-lite-preview-09-2025"

# Generate content
response = client.models.generate_content(
    model=model,
    contents=[
        "Tell me a joke about alligators"
    ],
)

print(response.text)
```

### Token Usage and Cost Tracking

The response includes usage metadata that can be used to track token consumption and estimate costs:

```python
# Extract token usage from response metadata
usage = response.usage_metadata
input_tokens = usage.prompt_token_count
output_tokens = usage.candidates_token_count
total_tokens = usage.total_token_count

# Cost calculation (per million tokens)
# Note: These rates are for gemini-2.5-flash-lite-preview-09-2025
# Adjust rates based on your specific model and current pricing
input_cost_per_million = 0.1
output_cost_per_million = 0.4

input_cost = (input_tokens * input_cost_per_million) / 1_000_000
output_cost = (output_tokens * output_cost_per_million) / 1_000_000
total_cost = input_cost + output_cost

print(f"Input tokens: {input_tokens}")
print(f"Output tokens: {output_tokens}")
print(f"Total tokens: {total_tokens}")
print(f"Estimated cost: ${total_cost:.6f}")
```

### Notes

- **Model Availability**: Model names and availability may vary by region. Check the [Vertex AI Model Garden](https://console.cloud.google.com/vertex-ai/model-garden) for available models.
- **Pricing**: Token costs vary by model. Refer to [Google Cloud Pricing](https://cloud.google.com/vertex-ai/pricing) for current rates.
- **Global Endpoint**: Using `'global'` as the location provides better availability but may have different quotas than regional endpoints.