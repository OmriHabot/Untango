"""
Pydantic models for request/response schemas.
"""
from typing import Optional
from pydantic import BaseModel


class CodeIngestRequest(BaseModel):
    """request model for ingesting python code"""
    code: str
    filepath: str
    repo_name: str


class QueryRequest(BaseModel):
    """request model for querying the vector db"""
    query: str
    n_results: int = 5


class ChunkMetadata(BaseModel):
    """metadata for each code chunk"""
    filepath: str
    repo_name: str
    chunk_type: str
    start_line: int
    end_line: int
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    imports: Optional[str] = None

