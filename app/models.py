"""
Pydantic models for request/response schemas.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class CodeIngestRequest(BaseModel):
    """Request model for ingesting python code"""
    code: str = Field(..., description="Python code content to ingest")
    filepath: str = Field(..., description="Path of the source file")
    repo_name: str = Field(..., description="Repository name")


class QueryRequest(BaseModel):
    """Request model for querying the vector db"""
    query: str = Field(..., description="Search query text")
    n_results: int = Field(default=5, ge=1, le=50, description="Number of results to return")


class InferenceRequest(BaseModel):
    """Request model for Vertex AI inference"""
    prompt: str = Field(..., description="Prompt for the AI model", min_length=1, max_length=10000)
    model: str = Field(default="gemini-2.0-flash-exp", description="Model to use for inference")


class ChunkMetadata(BaseModel):
    """Metadata for each code chunk"""
    filepath: str
    repo_name: str
    chunk_type: str
    start_line: int
    end_line: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    imports: Optional[str] = None


class SearchResult(BaseModel):
    """Single search result"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float


class TokenUsage(BaseModel):
    """Token usage information"""
    input_tokens: int
    output_tokens: int
    total_tokens: int


class InferenceResponse(BaseModel):
    """Response model for inference"""
    status: str
    model: str
    response: str
    usage: Optional[TokenUsage] = None


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    chroma_heartbeat: int
    collection_name: str
    gcp_configured: bool

