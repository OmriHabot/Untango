"""
Agent 3/4: Runbook Generator
Synthesizes environment info, repo structure, and RAG insights to generate a Markdown runbook.
"""
import logging
from typing import Optional

from ..models import EnvInfo, RepoMap, RAGQueryRequest
# We will import query_db dynamically or pass it to avoid circular imports if possible, 
# but since query_db is a route handler, we might want to extract the logic to a service function.
# For now, we will assume we can call the search logic directly or via internal API.
from ..search import perform_hybrid_search
from google import genai
import os

logger = logging.getLogger(__name__)

async def generate_runbook_content(
    repo_map: RepoMap, 
    env_info: EnvInfo, 
    project_id: str, 
    location: str,
    model: str = "gemini-2.0-flash-exp"
) -> str:
    """
    Generate the runbook content using LLM.
    """
    logger.info("Generating runbook content...")
    
    # 1. Construct a context-rich prompt
    prompt = f"""
You are an expert DevOps engineer and Technical Writer. Your task is to generate a "Quick Start / Runbook" for a developer who wants to run this repository.

CONTEXT:

1. **Repository Structure**:
{repo_map.structure}

2. **Detected Entry Points**:
{repo_map.entry_points}

3. **Dependencies**:
{repo_map.dependencies}

4. **Target Environment (User's Machine)**:
- OS: {env_info.os_info}
- Python: {env_info.python_version}
- GPU/CUDA: {env_info.gpu_info} (Available: {env_info.cuda_available})

TASK:
Create a Markdown runbook (`RUNBOOK.md`) that explains:
1. **Prerequisites**: What needs to be installed (based on dependencies and env).
2. **Setup**: How to install dependencies (pip, conda, etc.).
3. **Execution**: How to run the main entry points.
4. **Troubleshooting**: Any potential issues based on the environment (e.g., if CUDA is missing but required).

Keep it concise, actionable, and specific to the provided file structure.
"""

    # 2. Call Vertex AI
    try:
        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )
        
        response = client.models.generate_content(
            model=model,
            contents=prompt
        )
        
        # Log usage
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            meta = response.usage_metadata
            logger.info(
                "Runbook generation token usage: input=%d, output=%d, total=%d",
                getattr(meta, 'prompt_token_count', 0),
                getattr(meta, 'candidates_token_count', 0),
                getattr(meta, 'total_token_count', 0)
            )
        
        return getattr(response, 'text', '') or "Error: No response generated."
        
    except Exception as e:
        logger.exception("Failed to generate runbook with LLM")
        return f"Error generating runbook: {str(e)}"
