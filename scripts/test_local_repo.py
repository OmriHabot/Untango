"""
Simple test for local repository ingestion.
"""
import requests
import json

API_URL = "http://localhost:8001"

def test_local_ingestion():
    """Test ingesting the current local directory."""
    print("Testing local directory ingestion...")
    
    payload = {
        "source": {
            "type": "local",
            "location": "."
        },
        "parse_dependencies": True
    }
    
    response = requests.post(f"{API_URL}/ingest-repository", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Successfully ingested repository:")
        print(f"  - Repo ID: {data['repo_id']}")
        print(f"  - Name: {data['repo_name']}")
        print(f"  - Files: {data['file_count']}")
        print(f"  - Dependencies: {len(data['dependencies'])}")
        print(f"  - Sample dependencies: {data['dependencies'][:5]}")
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return False

if __name__ == "__main__":
    if test_local_ingestion():
        print("\n✓ Test passed!")
    else:
        print("\n✗ Test failed!")
