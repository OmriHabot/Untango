"""
Orchestrator
Coordinates the execution of agents to generate a runbook.
"""
import logging
import os
from fastapi import HTTPException

from .models import RunbookRequest, RunbookResponse
from .agents.env_scanner import scan_environment
from .agents.repo_mapper import map_repo
from .agents.runbook_generator import generate_runbook_content

logger = logging.getLogger(__name__)

async def generate_runbook_orchestrator(request: RunbookRequest) -> RunbookResponse:
    """
    Orchestrate the runbook generation process.
    1. Scan Environment (Agent 0)
    2. Map Repository (Agent 1)
    3. Generate Runbook (Agent 3/4)
    """
    try:
        # 1. Scan Environment
        logger.info("Step 1: Scanning environment...")
        env_info = scan_environment()
        
        # 2. Map Repository
        logger.info(f"Step 2: Mapping repository at {request.repo_path}...")
        if not os.path.exists(request.repo_path):
             raise HTTPException(status_code=404, detail=f"Repository path not found: {request.repo_path}")
             
        repo_map = map_repo(request.repo_path, request.repo_name)
        
        # 3. Generate Runbook
        logger.info("Step 3: Generating runbook...")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        
        if not project_id:
             raise HTTPException(status_code=503, detail="GCP Project ID not configured")

        runbook_content = await generate_runbook_content(
            repo_map=repo_map,
            env_info=env_info,
            project_id=project_id,
            location=location
        )
        
        return RunbookResponse(
            status="success",
            runbook=runbook_content,
            env_info=env_info,
            repo_map=repo_map
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Orchestration failed")
        raise HTTPException(status_code=500, detail=f"Runbook generation failed: {str(e)}")
