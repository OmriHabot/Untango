"""
FastAPI application for RAG backend with ChromaDB and Vertex AI.
"""
import os
from fastapi import FastAPI, HTTPException

from .models import (
    CodeIngestRequest,
    QueryRequest,
    InferenceRequest,
    InferenceResponse,
    HealthResponse,
    TokenUsage
)
from .database import (
    get_collection,
    get_client,
    get_collection_name,
    reset_collection
)
from .chunker import chunk_python_code
from .search import perform_vector_search, perform_hybrid_search


app = FastAPI(
    title="RAG Backend",
    description="Intelligent code chunking and retrieval with hybrid search",
    version="1.0.0"
)


@app.post("/ingest")
async def ingest_code(request: CodeIngestRequest):
    """ingest python code, chunk it via ast, and store in chromadb"""
    try:
        chunks = chunk_python_code(request.code, request.filepath, request.repo_name)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="no valid code chunks found")
        
        # prepare batch insertion
        ids = [chunk["id"] for chunk in chunks]
        documents = [chunk["content"] for chunk in chunks]
        
        # filter out None values from metadata - chromadb cannot handle None in metadata fields
        metadatas = []
        for chunk in chunks:
            filtered_metadata = {}
            for key, value in chunk["metadata"].items():
                if value is not None:  # only include non-None values
                    filtered_metadata[key] = value
            metadatas.append(filtered_metadata)
        
        # chromadb will automatically generate embeddings using the integrated function
        collection = get_collection()
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        
        return {
            "status": "success",
            "chunks_ingested": len(chunks),
            "collection_name": get_collection_name()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ingestion failed: {str(e)}")


@app.post("/query")
async def query_code(request: QueryRequest):
    """
    query the vector db using vector similarity search.
    for hybrid search (vector + bm25), use the /query-hybrid endpoint.
    """
    try:
        # perform vector similarity search
        vector_results = perform_vector_search(request.query, request.n_results)
        
        # format results
        formatted_results = []
        if vector_results.get("ids") and vector_results["ids"] and vector_results["ids"][0]:
            for i, doc_id in enumerate(vector_results["ids"][0]):
                formatted_results.append({
                    "id": doc_id,
                    "content": vector_results["documents"][0][i],
                    "metadata": vector_results["metadatas"][0][i],
                    "distance": vector_results["distances"][0][i]
                })
        
        return {
            "status": "success",
            "query": request.query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"query failed: {str(e)}")


@app.post("/query-hybrid")
async def query_hybrid(request: QueryRequest):
    """
    hybrid search combining vector similarity and bm25 keyword search.
    this provides better results by leveraging both semantic and lexical matching.
    """
    try:
        sorted_results = perform_hybrid_search(request.query, request.n_results)
        
        return {
            "status": "success",
            "query": request.query,
            "results": sorted_results,
            "count": len(sorted_results)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"hybrid search failed: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health including ChromaDB and GCP configuration"""
    try:
        client = get_client()
        heartbeat = client.heartbeat()
        
        # Check if GCP is configured
        gcp_configured = bool(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and 
            os.getenv("GOOGLE_CLOUD_PROJECT")
        )
        
        return HealthResponse(
            status="healthy",
            chroma_heartbeat=heartbeat,
            collection_name=get_collection_name(),
            gcp_configured=gcp_configured
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"service unavailable: {str(e)}")


@app.delete("/collection")
async def delete_collection():
    """delete the entire collection (use with caution)"""
    try:
        reset_collection()
        return {
            "status": "success",
            "message": f"collection '{get_collection_name()}' reset"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"collection reset failed: {str(e)}")


@app.post("/inference", response_model=InferenceResponse)
async def generate_inference(request: InferenceRequest) -> InferenceResponse:
    """
    Generate AI inference using Vertex AI.
    Requires GCP credentials to be configured.
    """
    try:
        # Check if GCP is configured
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        if not credentials_path or not project_id:
            raise HTTPException(
                status_code=503,
                detail="GCP not configured. Set GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT"
            )
        
        # Import Vertex AI client
        try:
            from google import genai
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="google-genai not installed"
            )
        
        # Initialize and call Vertex AI
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        
        response = client.models.generate_content(
            model=request.model,
            contents=request.prompt
        )
        
        # Get response text
        response_text = getattr(response, 'text', '') or ''
        
        # Extract token usage
        usage = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            metadata = response.usage_metadata
            usage = TokenUsage(
                input_tokens=getattr(metadata, 'prompt_token_count', 0),
                output_tokens=getattr(metadata, 'candidates_token_count', 0),
                total_tokens=getattr(metadata, 'total_token_count', 0)
            )
        
        return InferenceResponse(
            status="success",
            model=request.model,
            response=response_text,
            usage=usage
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"inference failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        timeout_keep_alive=75,
        timeout_graceful_shutdown=10
    )
