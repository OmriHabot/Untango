"""
FastAPI application for RAG backend with ChromaDB and Vertex AI.
"""
import logging
import os
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    CodeIngestRequest,
    QueryRequest,
    InferenceRequest,
    InferenceResponse,
    HealthResponse,
    TokenUsage,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievedChunk,
    RunbookRequest,
    RunbookResponse,
    ChatRequest,
    ChatResponse,
    RepositorySource,
    IngestRepositoryRequest,
    RepositoryInfo,
    SetActiveRepositoryRequest,
    SetActiveRepositoryRequest,
    ListRepositoriesResponse,
    ToolCall,
    MessagePart,
    Message
)
from .database import (
    get_collection,
    get_client,
    get_collection_name,
    reset_collection
)
from .chunker import chunk_python_code
from .ingest_manager import ingest_manager, IngestManager
from .repo_manager import repo_manager, RepositoryContext
from .active_repo_state import active_repo_state
from .search import perform_hybrid_search
from .logger import setup_logging, get_logger
from .agents.chat_agent import chat_with_agent, chat_with_agent_stream
from .chat_history import chat_history_manager


LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO")
JSON_LOGS = os.getenv("APP_LOG_JSON", "false").lower() in {"1", "true", "yes"}

setup_logging(log_level=LOG_LEVEL, json_logs=JSON_LOGS)
logger = get_logger(__name__)


app = FastAPI(
    title="RAG Backend",
    description="Intelligent code chunking and retrieval with hybrid search",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Added for the new endpoint
async def run_background_ingest(repo_path: str, repo_id: str, repo_name: str):
    """Helper for background ingestion."""
    try:
        logger.info(f"Starting background ingestion for {repo_name}")
        manager = IngestManager(repo_path=repo_path, repo_id=repo_id, repo_name=repo_name)
        await manager.sync_repo()
        logger.info(f"Background ingestion finished for {repo_name}")
    except Exception as e:
        logger.error(f"Background ingestion failed for {repo_name}: {e}")

@app.post("/api/select-repo")
async def select_repo(request: SetActiveRepositoryRequest, background_tasks: BackgroundTasks):
    """Select a repository to be active."""
    try:
        # Verify repo exists
        repos = repo_manager.list_repositories()
        repo = next((r for r in repos if r['repo_id'] == request.repo_id), None)
        
        if not repo:
            raise HTTPException(status_code=404, detail="Repository not found")
            
        # Set active repo
        active_repo_state.set_active_repo(request.repo_id)
        
        # Initialize Context Manager for this repo
        context_manager.initialize_context(repo['path'], repo['name'])
        
        # Trigger RAG Ingestion in Background
        background_tasks.add_task(run_background_ingest, repo['path'], request.repo_id, repo['name'])
        
        return {"status": "success", "message": f"Selected repo: {repo['name']}"}
    except Exception as e:
        logger.error(f"Error selecting repo: {e}")
        raise HTTPException(status_code=500, detail=str(e))

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
            logger.info(
                "RAG query token usage: input=%d, output=%d, total=%d",
                usage.input_tokens, usage.output_tokens, usage.total_tokens
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
            logger.info(
                "Inference token usage: input=%d, output=%d, total=%d",
                usage.input_tokens, usage.output_tokens, usage.total_tokens
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


from .agents.runbook_generator import generate_runbook_content

@app.post("/generate-runbook", response_model=RunbookResponse)
async def generate_runbook(request: RunbookRequest) -> RunbookResponse:
    """
    Generate a comprehensive runbook for a repository.
    Orchestrates environment scanning, repo mapping, and LLM generation.
    """
    logger.info("Runbook generation requested for repo '%s'", request.repo_name)
    
    try:
        # 1. Get Context (Env + Repo)
        # We use initialize_context to ensure it's fresh or loaded
        report = context_manager.initialize_context(request.repo_path, request.repo_name)
        
        # 2. Generate Runbook
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        if not project_id:
             raise HTTPException(status_code=503, detail="GCP Project ID not configured")

        runbook_content = await generate_runbook_content(
            repo_map=report.repo_map,
            env_info=report.env_info,
            dependency_analysis=report.dependency_analysis,
            project_id=project_id,
            location=location
        )
        
        return RunbookResponse(
            status="success",
            runbook=runbook_content,
            env_info=report.env_info,
            repo_map=report.repo_map
        )
    except Exception as e:
        logger.exception("Runbook generation failed")
        raise HTTPException(status_code=500, detail=f"Runbook generation failed: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """
    Chat with the agent about the codebase.
    Supports RAG, file exploration, and dependency inspection.
    """
    logger.info("Chat request received")
    
    # Trigger smart ingestion (background sync)
    # We await it to ensure consistency for the current query
    try:
        active_repo_id = active_repo_state.get_active_repo_id()
        if active_repo_id and active_repo_id != "default":
            # Get repo context to find path
            repo_path = os.path.join(repo_manager.repos_base_path, active_repo_id)
            # Create specific ingest manager
            # Note: In a real app we might want to cache these managers
            current_ingest_manager = IngestManager(
                repo_path=repo_path,
                repo_id=active_repo_id,
                repo_name=active_repo_id  # We could lookup name but ID is sufficient for logging
            )
            await current_ingest_manager.sync_repo()
        else:
            # Fallback to default
            await ingest_manager.sync_repo()
    except Exception as e:
        logger.error(f"Smart ingestion failed: {e}")
        
    # Save user message
    if request.messages:
        active_repo_id = active_repo_state.get_active_repo_id()
        if active_repo_id and active_repo_id != "default":
            chat_history_manager.add_message(active_repo_id, request.messages[-1])

    response = await chat_with_agent(request)
    
    # Save assistant response
    if active_repo_id and active_repo_id != "default":
        chat_history_manager.add_message(active_repo_id, Message(role="assistant", content=response.response))
        
    return response


@app.post("/chat-stream")
async def chat_stream_endpoint(request: ChatRequest):
    """
    Chat with the agent with streaming response.
    Returns a stream of JSON events.
    """
    logger.info("Chat stream request received")
    
    # Trigger smart ingestion (background sync)
    try:
        active_repo_id = active_repo_state.get_active_repo_id()
        if active_repo_id and active_repo_id != "default":
            repo_path = os.path.join(repo_manager.repos_base_path, active_repo_id)
            current_ingest_manager = IngestManager(
                repo_path=repo_path,
                repo_id=active_repo_id,
                repo_name=active_repo_id
            )
            await current_ingest_manager.sync_repo()
        else:
            await ingest_manager.sync_repo()
    except Exception as e:
        logger.error(f"Smart ingestion failed: {e}")
        
    # Save user message
    if request.messages:
        active_repo_id = active_repo_state.get_active_repo_id()
        if active_repo_id and active_repo_id != "default":
            chat_history_manager.add_message(active_repo_id, request.messages[-1])

    async def stream_wrapper():
        full_response = ""
        parts = []
        tool_calls = []
        current_tool_call = None
        
        async for chunk in chat_with_agent_stream(request):
            # Capture content for history
            try:
                # Chunk might be multiple lines or raw NDJSON
                lines = chunk.strip().split('\n')
                for line in lines:
                    if not line: continue
                    # Handle both SSE (data: ...) and raw NDJSON
                    if line.startswith("data: "):
                        line = line[6:]
                    
                    try:
                        data = json.loads(line)
                        event_type = data.get("type")
                        
                        if event_type == "token":
                            content = data.get("content", "")
                            full_response += content
                            
                            # Update or create text part
                            if parts and parts[-1].type == "text":
                                parts[-1].content = (parts[-1].content or "") + content
                            else:
                                parts.append(MessagePart(type="text", content=content))
                                
                        elif event_type == "tool_start":
                            tool_call = ToolCall(
                                tool=data.get("tool"),
                                args=data.get("args"),
                                status="running"
                            )
                            tool_calls.append(tool_call)
                            parts.append(MessagePart(type="tool", tool_call=tool_call))
                            current_tool_call = tool_call
                            
                        elif event_type == "tool_end":
                            if current_tool_call and current_tool_call.tool == data.get("tool"):
                                current_tool_call.result = data.get("result")
                                current_tool_call.status = "completed"
                                current_tool_call = None
                                
                        elif event_type == "usage":
                            # Usage stats are usually the last event
                            pass
                            
                    except json.JSONDecodeError:
                        pass
            except Exception:
                pass
            yield chunk
            
        # Save full assistant response with parts and tool calls
        if active_repo_id and active_repo_id != "default":
             msg = Message(
                 role="assistant", 
                 content=full_response,
                 parts=parts,
                 tool_calls=tool_calls
             )
             chat_history_manager.add_message(active_repo_id, msg)

    return StreamingResponse(
            stream_wrapper(),
            media_type="application/x-ndjson"
        )


async def ingest_repo_background(repo_context: RepositoryContext, repo_ingest_manager: IngestManager):
    """Background task for repository ingestion."""
    try:
        active_repo_state.set_ingestion_status(repo_context.repo_id, "ingesting")
        logger.info(f"Starting background ingestion for {repo_context.repo_id}")
        
        await repo_ingest_manager.sync_repo()
        
        active_repo_state.set_ingestion_status(repo_context.repo_id, "completed")
        logger.info(f"Background ingestion completed for {repo_context.repo_id}")
        
        # Set as active repository automatically upon completion
        active_repo_state.set_active_repo_id(repo_context.repo_id)
        
    except Exception as e:
        logger.exception(f"Background ingestion failed for {repo_context.repo_id}")
        active_repo_state.set_ingestion_status(repo_context.repo_id, "failed")


@app.post("/ingest-repository", response_model=RepositoryInfo)
async def ingest_repository_endpoint(
    request: IngestRepositoryRequest, 
    background_tasks: BackgroundTasks
) -> RepositoryInfo:
    """
    Ingest a repository from GitHub or local filesystem.
    Starts ingestion in the background and returns 'pending' status.
    """
    logger.info(f"Ingesting repository: {request.source.type} - {request.source.location}")
    
    try:
        # Create repository context
        repo_context = repo_manager.create_repository_context(
            source_type=request.source.type,
            source_location=request.source.location,
            branch=request.source.branch if request.source.type == "github" else None,
            parse_dependencies=request.parse_dependencies
        )
        
        logger.info(f"Repository context created: {repo_context.repo_id} - {repo_context.repo_name}")
        
        # Create IngestManager for this repository
        repo_ingest_manager = IngestManager(
            repo_path=repo_context.repo_path,
            repo_id=repo_context.repo_id,
            repo_name=repo_context.repo_name
        )
        
        # Set initial status
        active_repo_state.set_ingestion_status(repo_context.repo_id, "pending")
        
        # Start background task
        background_tasks.add_task(ingest_repo_background, repo_context, repo_ingest_manager)
        
        # Count files (approximate from cache, likely 0 initially)
        file_count = len(repo_ingest_manager.cache)
        
        return RepositoryInfo(
            repo_id=repo_context.repo_id,
            repo_name=repo_context.repo_name,
            source_type=repo_context.source_type,
            source_location=repo_context.source_location,
            local_path=repo_context.repo_path,
            dependencies=repo_context.dependencies,
            file_count=file_count,
            status="pending"
        )
    except Exception as e:
        logger.exception("Repository ingestion failed")
        raise HTTPException(status_code=500, detail=f"Failed to ingest repository: {str(e)}")


@app.get("/repository/{repo_id}/status")
async def get_repository_status_endpoint(repo_id: str) -> dict:
    """Get the ingestion status of a repository."""
    status = active_repo_state.get_ingestion_status(repo_id)
    return {
        "repo_id": repo_id,
        "status": status
    }


@app.post("/set-active-repository")
async def set_active_repository_endpoint(request: SetActiveRepositoryRequest) -> dict:
    """
    Set the active repository for queries.
    """
    logger.info(f"Setting active repository: {request.repo_id}")
    active_repo_state.set_active_repo_id(request.repo_id)
    
    return {
        "status": "success",
        "active_repo_id": request.repo_id,
        "message": f"Active repository set to {request.repo_id}"
    }


@app.get("/active-repository")
async def get_active_repository_endpoint() -> dict:
    """
    Get the currently active repository.
    """
    repo_id = active_repo_state.get_active_repo_id()
    return {
        "active_repo_id": repo_id
    }


@app.get("/list-repositories", response_model=ListRepositoriesResponse)
async def list_repositories_endpoint() -> ListRepositoriesResponse:
    """
    List all available repositories.
    """
    repos = repo_manager.list_repositories()
    return ListRepositoriesResponse(
        repositories=repos,
        count=len(repos)
    )

@app.get("/chat/history")
async def get_chat_history():
    """Get chat history for the active repository."""
    active_repo_id = active_repo_state.get_active_repo_id()
    if not active_repo_id or active_repo_id == "default":
        return {"history": []}
    
    history = chat_history_manager.get_history(active_repo_id)
    return {"history": history}


@app.delete("/chat/history")
async def clear_chat_history():
    """Clear chat history for the active repository."""
    active_repo_id = active_repo_state.get_active_repo_id()
    if active_repo_id and active_repo_id != "default":
        chat_history_manager.clear_history(active_repo_id)
    return {"status": "success"}


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
