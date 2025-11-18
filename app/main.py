"""
FastAPI application for RAG backend with ChromaDB and Vertex AI.
"""
import logging
import os
from fastapi import FastAPI, HTTPException

from .models import (
    CodeIngestRequest,
    QueryRequest,
    InferenceRequest,
    InferenceResponse,
    HealthResponse,
    TokenUsage,
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievedChunk
)
from .database import (
    get_collection,
    get_client,
    get_collection_name,
    reset_collection
)
from .chunker import chunk_python_code
from .search import perform_vector_search, perform_hybrid_search
from .logger import setup_logging, get_logger


LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO")
JSON_LOGS = os.getenv("APP_LOG_JSON", "false").lower() in {"1", "true", "yes"}

setup_logging(log_level=LOG_LEVEL, json_logs=JSON_LOGS)
logger = get_logger(__name__)


app = FastAPI(
    title="RAG Backend",
    description="Intelligent code chunking and retrieval with hybrid search",
    version="1.0.0"
)


@app.post("/ingest")
async def ingest_code(request: CodeIngestRequest):
    """ingest python code, chunk it via ast, and store in chromadb"""
    try:
        logger.info(
            "Ingest request received for repo '%s' file '%s'",
            request.repo_name,
            request.filepath
        )
        chunks = chunk_python_code(request.code, request.filepath, request.repo_name)
        
        if not chunks:
            logger.warning(
                "No valid code chunks found for repo '%s' file '%s'",
                request.repo_name,
                request.filepath
            )
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
        logger.info(
            "Ingested %d chunk(s) into collection '%s'",
            len(chunks),
            get_collection_name()
        )
        
        return {
            "status": "success",
            "chunks_ingested": len(chunks),
            "collection_name": get_collection_name()
        }
    except HTTPException as http_exc:
        logger.warning("Ingest request failed: %s", http_exc.detail)
        raise
    except Exception as e:
        logger.exception(
            "Unexpected error during ingestion for repo '%s' file '%s'",
            request.repo_name,
            request.filepath
        )
        raise HTTPException(status_code=500, detail=f"ingestion failed: {str(e)}")


@app.post("/query")
async def query_code(request: QueryRequest):
    """
    query the vector db using vector similarity search.
    for hybrid search (vector + bm25), use the /query-hybrid endpoint.
    """
    try:
        logger.info(
            "Vector query received: '%s' (top %d, similarity_threshold=%s)",
            request.query,
            request.n_results,
            request.vector_similarity_threshold
        )
        # perform vector similarity search with optional threshold
        vector_results = perform_vector_search(
            request.query,
            request.n_results,
            similarity_threshold=request.vector_similarity_threshold
        )
        
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
        
        response = {
            "status": "success",
            "query": request.query,
            "results": formatted_results,
            "count": len(formatted_results)
        }
        logger.info(
            "Vector query complete: %d result(s) returned",
            len(formatted_results)
        )
        return response
    except Exception as e:
        logger.exception("Vector query failed for query '%s'", request.query)
        raise HTTPException(status_code=500, detail=f"query failed: {str(e)}")


@app.post("/query-hybrid")
async def query_hybrid(request: QueryRequest):
    """
    hybrid search combining vector similarity and bm25 keyword search.
    this provides better results by leveraging both semantic and lexical matching.
    """
    try:
        logger.info(
            "Hybrid query received: '%s' (top %d, vector_threshold=%s, bm25_threshold=%s)",
            request.query,
            request.n_results,
            request.vector_similarity_threshold,
            request.bm25_score_threshold
        )
        sorted_results = perform_hybrid_search(
            request.query,
            request.n_results,
            vector_similarity_threshold=request.vector_similarity_threshold,
            bm25_score_threshold=request.bm25_score_threshold
        )
        
        response = {
            "status": "success",
            "query": request.query,
            "results": sorted_results,
            "count": len(sorted_results)
        }
        logger.info(
            "Hybrid query complete: %d result(s) returned",
            len(sorted_results)
        )
        return response
    except Exception as e:
        logger.exception("Hybrid query failed for query '%s'", request.query)
        raise HTTPException(status_code=500, detail=f"hybrid search failed: {str(e)}")


@app.post("/query-db", response_model=RAGQueryResponse)
async def query_db(request: RAGQueryRequest) -> RAGQueryResponse:
    """
    Complete RAG pipeline: retrieve relevant chunks using hybrid search,
    filter by confidence threshold, and generate answer using Vertex AI.
    
    This endpoint combines:
    1. Hybrid search (vector + BM25) for retrieval
    2. Confidence filtering (only chunks with combined_score > threshold)
    3. LLM inference with retrieved context
    """
    try:
        logger.info(
            "RAG query received: '%s' (top %d, threshold %.2f, vector_threshold=%s, bm25_threshold=%s, model %s)",
            request.query,
            request.n_results,
            request.confidence_threshold,
            request.vector_similarity_threshold,
            request.bm25_score_threshold,
            request.model
        )
        # Check if GCP is configured
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        if not credentials_path or not project_id:
            logger.warning(
                "RAG query blocked: missing GCP configuration (credentials=%s, project=%s)",
                bool(credentials_path),
                bool(project_id)
            )
            raise HTTPException(
                status_code=503,
                detail="GCP not configured. Set GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT"
            )
        
        # Step 1: Perform hybrid search to retrieve relevant chunks with optional thresholds
        hybrid_results = perform_hybrid_search(
            request.query,
            request.n_results,
            vector_similarity_threshold=request.vector_similarity_threshold,
            bm25_score_threshold=request.bm25_score_threshold
        )
        logger.info(
            "Hybrid retrieval returned %d chunk(s) for RAG query",
            len(hybrid_results)
        )
        
        # Step 2: Filter chunks by confidence threshold
        filtered_chunks = [
            chunk for chunk in hybrid_results 
            if chunk["combined_score"] > request.confidence_threshold
        ]
        
        if not filtered_chunks:
            logger.warning(
                "No chunks passed confidence threshold %.2f for query '%s'",
                request.confidence_threshold,
                request.query
            )
            raise HTTPException(
                status_code=404,
                detail=f"No chunks found with confidence > {request.confidence_threshold}. Try lowering the threshold."
            )
        
        # Step 3: Build context from retrieved chunks
        context_parts = []
        for i, chunk in enumerate(filtered_chunks, 1):
            metadata = chunk["metadata"]
            context_parts.append(
                f"--- Chunk {i} (confidence: {chunk['combined_score']:.3f}) ---\n"
                f"File: {metadata.get('filepath', 'unknown')}\n"
                f"Type: {metadata.get('chunk_type', 'unknown')}\n"
                f"Lines: {metadata.get('start_line', '?')}-{metadata.get('end_line', '?')}\n\n"
                f"{chunk['content']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Step 4: Build the prompt
        prompt = f"""You are a helpful AI assistant. Use the following retrieved code chunks to answer the user's question.
If the answer cannot be found in the provided chunks, say so.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{request.query}

ANSWER:"""
        
        # Step 5: Call Vertex AI for inference
        try:
            from google import genai
        except ImportError:
            logger.exception("google-genai package not installed for RAG query")
            raise HTTPException(
                status_code=500,
                detail="google-genai not installed"
            )
        
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        
        response = client.models.generate_content(
            model=request.model,
            contents=prompt
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
        
        # Step 6: Format retrieved chunks for response
        retrieved_chunks = [
            RetrievedChunk(
                id=chunk.get("id", "unknown"),
                content=chunk["content"],
                metadata=chunk["metadata"],
                vector_score=chunk.get("rrf_dense", 0.0),
                bm25_score=chunk.get("rrf_bm25", 0.0),
                combined_score=chunk["combined_score"]
            )
            for chunk in filtered_chunks
        ]
        
        logger.info(
            "RAG answer generated with %d chunk(s) using model %s",
            len(filtered_chunks),
            request.model
        )
        
        return RAGQueryResponse(
            status="success",
            query=request.query,
            retrieved_chunks=retrieved_chunks,
            chunks_used=len(filtered_chunks),
            answer=response_text,
            model=request.model,
            usage=usage
        )
        
    except HTTPException as http_exc:
        logger.warning("RAG query failed: %s", http_exc.detail)
        raise
    except Exception as e:
        logger.exception("Unexpected error during RAG query for '%s'", request.query)
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check service health including ChromaDB and GCP configuration"""
    try:
        logger.debug("Health check requested")
        client = get_client()
        heartbeat = client.heartbeat()
        
        # Check if GCP is configured
        gcp_configured = bool(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS") and 
            os.getenv("GOOGLE_CLOUD_PROJECT")
        )
        
        response = HealthResponse(
            status="healthy",
            chroma_heartbeat=heartbeat,
            collection_name=get_collection_name(),
            gcp_configured=gcp_configured
        )
        logger.info(
            "Health check success (chroma heartbeat=%s, gcp_configured=%s)",
            heartbeat,
            gcp_configured
        )
        return response
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail=f"service unavailable: {str(e)}")


@app.delete("/collection")
async def delete_collection():
    """delete the entire collection (use with caution)"""
    try:
        logger.warning("Collection reset requested")
        reset_collection()
        logger.info("Collection '%s' reset successfully", get_collection_name())
        response = {
            "status": "success",
            "message": f"collection '{get_collection_name()}' reset"
        }
        return response
    except Exception as e:
        logger.exception("Collection reset failed")
        raise HTTPException(status_code=500, detail=f"collection reset failed: {str(e)}")


@app.post("/inference", response_model=InferenceResponse)
async def generate_inference(request: InferenceRequest) -> InferenceResponse:
    """
    Generate AI inference using Vertex AI.
    Requires GCP credentials to be configured.
    """
    try:
        logger.info(
            "Inference request received for model %s",
            request.model
        )
        # Check if GCP is configured
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        if not credentials_path or not project_id:
            logger.warning(
                "Inference blocked: missing GCP configuration (credentials=%s, project=%s)",
                bool(credentials_path),
                bool(project_id)
            )
            raise HTTPException(
                status_code=503,
                detail="GCP not configured. Set GOOGLE_APPLICATION_CREDENTIALS and GOOGLE_CLOUD_PROJECT"
            )
        
        # Import Vertex AI client
        try:
            from google import genai
        except ImportError:
            logger.exception("google-genai package not installed for inference")
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
        
        logger.info("Inference completed for model %s", request.model)
        return InferenceResponse(
            status="success",
            model=request.model,
            response=response_text,
            usage=usage
        )
        
    except HTTPException as http_exc:
        logger.warning("Inference failed: %s", http_exc.detail)
        raise
    except Exception as e:
        logger.exception("Unexpected error during inference for model %s", request.model)
        raise HTTPException(
            status_code=500,
            detail=f"inference failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting RAG backend server on 0.0.0.0:8001")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        timeout_keep_alive=75,
        timeout_graceful_shutdown=10
    )
