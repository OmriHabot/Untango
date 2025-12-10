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
from datetime import datetime

logger = logging.getLogger(__name__)

from typing import List
from ..context_manager import DependencyStatus

async def generate_runbook_content(
    repo_map: RepoMap, 
    env_info: EnvInfo,
    dependency_analysis: List[DependencyStatus],
    project_id: str, 
    location: str,
    model: str = "gemini-3.0-flash"
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
    repo_name_display = repo_map.detected_name or repo_map.repo_name
    readme_status = "Found" if repo_map.readme_exists else "Not Found"
    current_date = datetime.now().strftime("%A, %B %d, %Y")

    prompt = f"""
You are an expert DevOps engineer and Technical Writer. Your task is to generate a "Quick Start / Runbook" for a developer who wants to run this repository: '{repo_name_display}'.

Current Date: {current_date}

CONTEXT:

1. **Repository Structure**:
Name: {repo_name_display} (Folder: {repo_map.repo_name})
Type: {repo_map.repo_type}
Last Updated: {repo_map.last_updated} (Approximate)
Structure:
{repo_map.structure}
Last Updated: {repo_map.last_updated} (Approximate - inferred from file system or git logs)

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
   - **If Type=library**: Suggest `pip install {repo_name_display}` (public) or `pip install .` (local). Avoid `git clone` unless dev.
   - **If Type=application**: Suggest `git clone` + install deps (e.g., `pip install -r requirements.txt`).
   - Use "Ref Docs" for specific commands.
   - Address env issues (e.g., missing CUDA).
4. **Execution**: Commands to run/test.

Style: Concise, actionable, Markdown.
"""

    # 4. Call Vertex AI
    
    # Fallback models in order
    FALLBACK_MODELS = ["gemini-3.0-flash-lite", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
    models_to_try = [model] + FALLBACK_MODELS
    # Remove duplicates
    unique_models = []
    seen = set()
    for m in models_to_try:
        if m not in seen:
            unique_models.append(m)
            seen.add(m)

    last_error = None

    for current_model in unique_models:
        try:
            logger.info(f"Attempting runbook generation with model: {current_model}")
            client = genai.Client(
                vertexai=True,
                project=project_id,
                location=location
            )
            
            response = client.models.generate_content(
                model=current_model,
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
            logger.warning(f"Model {current_model} failed: {e}")
            last_error = e
            continue

    logger.error(f"All models failed. Last error: {last_error}")
    return f"Error generating runbook: {str(last_error)}"
