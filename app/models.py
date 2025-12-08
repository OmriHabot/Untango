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
    n_results: int = Field(default=5, ge=1, description="Number of results to return")
    vector_similarity_threshold: Optional[float] = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Minimum vector similarity score (0-1). Results below this are filtered out."
    )
    bm25_score_threshold: Optional[float] = Field(
        default=1.0,
        ge=0.0,
        description="Minimum BM25 score. Results below this are filtered out. (Hybrid search only)"
    )


class InferenceRequest(BaseModel):
    """Request model for Vertex AI inference"""
    prompt: str = Field(..., description="Prompt for the AI model", min_length=1, max_length=10000)
    model: str = Field(default="gemini-2.5-flash", description="Model to use for inference")


class ChunkMetadata(BaseModel):
    """Metadata for each code chunk"""
    filepath: str
    repo_id: Optional[str] = None  # Repository ID for multi-repo support
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


class RAGQueryRequest(BaseModel):
    """Request model for RAG query with retrieval + inference"""
    query: str = Field(..., description="User question to answer using RAG", min_length=1)
    n_results: int = Field(default=10, ge=1, description="Number of chunks to retrieve for ranking")
    confidence_threshold: float = Field(default=0.010, ge=0.0, le=1.0, description="Minimum combined confidence score")
    vector_similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum vector similarity score (0-1) before RRF fusion"
    )
    bm25_score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Minimum BM25 score before RRF fusion"
    )
    model: str = Field(default="gemini-2.5-flash", description="Model to use for inference")


class RetrievedChunk(BaseModel):
    """A retrieved chunk with its metadata and score"""
    id: str
    content: str
    metadata: Dict[str, Any]
    vector_score: float
    bm25_score: float
    combined_score: float


class RAGQueryResponse(BaseModel):
    """Response model for RAG query with retrieved chunks and generated answer"""
    status: str
    query: str
    retrieved_chunks: List[RetrievedChunk]
    chunks_used: int
    answer: str
    model: str
    usage: Optional[TokenUsage] = None


class EnvInfo(BaseModel):
    """Environment information"""
    os_info: str
    python_version: str
    cuda_available: bool
    gpu_info: str
    installed_packages: List[str]


class RepoMap(BaseModel):
    """Repository structure map"""
    repo_name: str
    root_path: str
    structure: Dict[str, Any]
    entry_points: List[str]
    dependencies: List[str]
    last_updated: Optional[str] = None


class RunbookRequest(BaseModel):
    """Request model for generating a runbook"""
    repo_path: str = Field(..., description="Absolute path to the repository")
    repo_name: str = Field(..., description="Name of the repository")
    repo_id: str = Field(..., description="ID of the repository")


class RunbookResponse(BaseModel):
    """Response model for runbook generation"""
    status: str
    runbook: str
    env_info: EnvInfo
    repo_map: RepoMap


class ToolCall(BaseModel):
    """Tool call model"""
    tool: str
    args: Dict[str, Any]
    result: Optional[str] = None
    status: str = Field(default="completed", description="Status: 'running', 'completed', 'failed'")


class MessagePart(BaseModel):
    """Part of a message (text or tool call)"""
    type: str = Field(..., description="Type: 'text' or 'tool'")
    content: Optional[str] = None
    tool_call: Optional[ToolCall] = None


class Message(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    parts: Optional[List[MessagePart]] = Field(default=None, description="Message parts for rich rendering")
    tool_calls: Optional[List[ToolCall]] = Field(default=None, description="Tool calls made in this message")
    usage: Optional[TokenUsage] = Field(default=None, description="Token usage stats")


class ChatRequest(BaseModel):
    """Request model for chat"""
    messages: List[Message] = Field(..., description="Chat history")
    model: str = Field(default="gemini-2.5-flash", description="Model to use")  # Field(default="gemini-2.5-flash", description="Model to use")


class ChatResponse(BaseModel):
    """Response model for chat"""
    status: str
    response: str
    trace: Optional[List[Dict[str, Any]]] = None
    usage: Optional[TokenUsage] = None


class RepositorySource(BaseModel):
    """Source specification for a repository"""
    type: str = Field(..., description="Source type: 'github' or 'local'")
    location: str = Field(..., description="GitHub URL or local filesystem path")
    branch: Optional[str] = Field(default=None, description="Git branch for GitHub repos")


class IngestRepositoryRequest(BaseModel):
    """Request to ingest a repository from a source"""
    source: RepositorySource
    parse_dependencies: bool = Field(default=True, description="Parse and detect dependencies")


class RepositoryInfo(BaseModel):
    """Information about an ingested repository"""
    repo_id: str
    repo_name: str
    source_type: str
    source_location: str
    local_path: str
    dependencies: List[str]
    file_count: int
    status: str = Field(default="completed", description="Ingestion status: pending, ingesting, completed, failed")


class SetActiveRepositoryRequest(BaseModel):
    """Request to set the active repository"""
    repo_id: str


class RepositorySummary(BaseModel):
    """Summary of a repository"""
    repo_id: str
    name: str
    path: str


class ListRepositoriesResponse(BaseModel):
    """Response listing all ingested repositories"""
    repositories: List[RepositorySummary]
    count: int
