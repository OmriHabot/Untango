"""
Script to ingest the local repository into the running RAG backend.
Walks the directory and sends each Python file to the /ingest endpoint.
"""
import os
import requests
import logging

# Configuration
API_URL = "http://localhost:8001/ingest"
REPO_PATH = "."
REPO_NAME = "Untango"
IGNORE_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'node_modules', '.idea', '.vscode', 'data'}

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def ingest_file(filepath: str):
    """Read file and send to API."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
            
        payload = {
            "code": code,
            "filepath": filepath,
            "repo_name": REPO_NAME
        }
        
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            logger.info(f"Successfully ingested: {filepath}")
        else:
            logger.error(f"Failed to ingest {filepath}: {response.text}")
            
    except Exception as e:
        logger.error(f"Error processing {filepath}: {e}")

def main():
    logger.info(f"Starting ingestion of {REPO_NAME} from {REPO_PATH}...")
    
    count = 0
    for root, dirs, files in os.walk(REPO_PATH):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        for file in files:
            if file.endswith('.py') or file.endswith('.md'):
                filepath = os.path.join(root, file)
                # Skip the script itself if it's in the path
                if "ingest_repo.py" in filepath:
                    continue
                    
                ingest_file(filepath)
                count += 1
                
    logger.info(f"Ingestion complete. Processed {count} files.")

if __name__ == "__main__":
    main()
