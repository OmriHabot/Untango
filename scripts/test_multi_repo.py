"""
Test script for multi-repository ingestion.
Tests both GitHub and local directory ingestion.
"""
import requests
import json
import sys

API_URL = "http://localhost:8001"


def test_github_ingestion():
    """Test ingesting a GitHub repository."""
    print("\n=== Testing GitHub Repository Ingestion ===")
    
    payload = {
        "source": {
            "type": "github",
            "location": "https://github.com/tiangolo/fastapi.git",
            "branch": "master"
        },
        "parse_dependencies": True
    }
    
    print(f"Ingesting: {payload['source']['location']}")
    response = requests.post(f"{API_URL}/ingest-repository", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Successfully ingested repository:")
        print(f"  - Repo ID: {data['repo_id']}")
        print(f"  - Name: {data['repo_name']}")
        print(f"  - Files: {data['file_count']}")
        print(f"  - Dependencies: {len(data['dependencies'])}")
        print(f"  - Local path: {data['local_path']}")
        return data['repo_id']
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return None


def test_local_ingestion():
    """Test ingesting a local directory."""
    print("\n=== Testing Local Directory Ingestion ===")
    
    payload = {
        "source": {
            "type": "local",
            "location": "/app"  # Inside Docker container
        },
        "parse_dependencies": True
    }
    
    print(f"Ingesting: {payload['source']['location']}")
    response = requests.post(f"{API_URL}/ingest-repository", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Successfully ingested local directory:")
        print(f"  - Repo ID: {data['repo_id']}")
        print(f"  - Name: {data['repo_name']}")
        print(f"  - Files: {data['file_count']}")
        print(f"  - Dependencies: {len(data['dependencies'])}")
        return data['repo_id']
    else:
        print(f"✗ Failed: {response.status_code}")
        print(response.text)
        return None


def test_set_active_repo(repo_id: str):
    """Test setting the active repository."""
    print(f"\n=== Setting Active Repository to {repo_id} ===")
    
    payload = {"repo_id": repo_id}
    response = requests.post(f"{API_URL}/set-active-repository", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ {data['message']}")
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        return False


def test_get_active_repo():
    """Test getting the active repository."""
    print("\n=== Getting Active Repository ===")
    
    response = requests.get(f"{API_URL}/active-repository")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Active repository: {data['active_repo_id']}")
        return data['active_repo_id']
    else:
        print(f"✗ Failed: {response.status_code}")
        return None


def test_query_active_repo():
    """Test querying the active repository."""
    print("\n=== Testing Query with Active Repository ===")
    
    payload = {
        "messages": [
            {"role": "user", "content": "What is this codebase about?"}
        ]
    }
    
    response = requests.post(f"{API_URL}/chat", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Query successful")
        print(f"Response: {data['response'][:200]}...")
        return True
    else:
        print(f"✗ Failed: {response.status_code}")
        return False


def main():
    """Run all tests."""
    print("Multi-Repository Ingestion Test Script")
    print("=" * 50)
    
    # Test GitHub ingestion
    github_repo_id = test_github_ingestion()
    if not github_repo_id:
        print("\n✗ GitHub ingestion test failed")
        return
    
    # Test local ingestion
    local_repo_id = test_local_ingestion()
    if not local_repo_id:
        print("\n✗ Local ingestion test failed")
        return
    
    # Test setting active repo
    if not test_set_active_repo(github_repo_id):
        print("\n✗ Set active repo test failed")
        return
    
    # Test getting active repo
    active_id = test_get_active_repo()
    if active_id != github_repo_id:
        print(f"\n✗ Active repo mismatch: expected {github_repo_id}, got {active_id}")
        return
    
    # Test query with active repo
    if not test_query_active_repo():
        print("\n✗ Query test failed")
        return
    
    # Switch to local repo
    if not test_set_active_repo(local_repo_id):
        print("\n✗ Set active repo (local) test failed")
        return
    
    # Query local repo
    if not test_query_active_repo():
        print("\n✗ Query test (local) failed")
        return
    
    print("\n" + "=" * 50)
    print("✓ All tests passed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        sys.exit(1)
