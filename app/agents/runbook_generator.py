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

from typing import List
from ..context_manager import DependencyStatus

async def generate_runbook_content(
    repo_map: RepoMap, 
    env_info: EnvInfo,
    dependency_analysis: List[DependencyStatus],
    project_id: str, 
    location: str,
    model: str = "gemini-2.0-flash-exp"
) -> str:
    """
    Generate the runbook content using LLM with RAG and Dependency Analysis.
    """
    logger.info("Generating runbook content...")
    
    # 1. Perform RAG Search for Context
    # We search for setup instructions to give the LLM ground truth
    search_queries = [
        "how to install dependencies",
        "how to run the application",
        "setup instructions",
        "deployment guide"
    ]
    
    rag_context = []
    for query in search_queries:
        results = perform_hybrid_search(query, n_results=2, repo_id=None) # repo_id should be handled by context if possible, or passed
        for r in results:
            rag_context.append(f"Source: {r['metadata'].get('filepath')}\nContent:\n{r['content']}")
            
    rag_context_str = "\n---\n".join(rag_context[:5]) # Limit context size

    # 2. Format Dependency Issues
    dep_issues = [d for d in dependency_analysis if d.status != "OK"]
    dep_issues_str = "None detected."
    if dep_issues:
        dep_issues_str = "\n".join([
            f"- {d.package}: Status={d.status} (Required: {d.required_version}, Installed: {d.installed_version})"
            for d in dep_issues
        ])

    # 3. Construct a context-rich prompt
    prompt = f"""
You are an expert DevOps engineer and Technical Writer. Your task is to generate a "Quick Start / Runbook" for a developer who wants to run this repository.

CONTEXT:

1. **Repository Structure**:
{repo_map.structure}
Last Updated: {repo_map.last_updated}

2. **Detected Entry Points**:
{repo_map.entry_points}

3. **Dependency Analysis (CRITICAL)**:
The following issues were detected in the user's environment:
{dep_issues_str}

4. **Retrieved Setup Instructions (RAG)**:
Use these snippets as ground truth for commands if relevant:
{rag_context_str}

5. **Target Environment (User's Machine)**:
- OS: {env_info.os_info}
- Python: {env_info.python_version}
- GPU/CUDA: {env_info.gpu_info} (Available: {env_info.cuda_available})

TASK:
Create a Markdown runbook (`RUNBOOK.md`) that explains:
1. What is the purpose of this repository?
2. **Prerequisites**: 
   - List required tools.
   - If the dependency version is different from the required version, warn the user that dependencies might be outdated or incompatible with modern Python versions.
   - **Time Rot Warning**: If the "Last Updated" date is old (e.g. > 1 year), warn the user that dependencies might be outdated or incompatible with modern Python versions.
3. **Setup**: 
   - Provide the suggested commands to install dependencies.
   - Use the "Retrieved Setup Instructions" to find the correct commands (e.g. `pip install -r requirements.txt` vs `poetry install`).
   - Provide the suggested commands to update the dependencies that may be outdated, and label it as optional.
   - Address environment limitations (e.g. if CUDA is missing).
4. **Execution**: 
   - Provide the suggested commands to run the repository after updating the dependencies.

Keep it concise, actionable, and specific to the provided file structure.
"""

    # 4. Call Vertex AI
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
