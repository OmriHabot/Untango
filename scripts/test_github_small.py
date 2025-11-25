"""
Test script for ingesting requests from GitHub.
This uses a smaller repository (psf/requests) for faster testing.
"""
import requests
import json
import time
import sys

API_URL = "http://localhost:8001"


def test_requests_github_ingestion():
    """Test ingesting requests from GitHub."""
    print("Testing GitHub repository ingestion (psf/requests)...")
    
    payload = {
        "source": {
            "type": "github",
            "location": "https://github.com/psf/requests.git",
            "branch": "main"
        },
        "parse_dependencies": True
    }
    
    print(f"\nIngesting: {payload['source']['location']}")
    
    try:
        response = requests.post(f"{API_URL}/ingest-repository", json=payload, timeout=30)
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Could not connect to {API_URL}. Is the server running?")
        return None
    
    if response.status_code == 200:
        data = response.json()
        repo_id = data['repo_id']
        print(f"\n✓ Ingestion started for requests:")
        print(f"  - Repo ID: {repo_id}")
        print(f"  - Status: {data.get('status', 'unknown')}")
        
        # Poll for completion
        print("\nWaiting for ingestion to complete...", end="", flush=True)
        start_time = time.time()
        while True:
            try:
                status_res = requests.get(f"{API_URL}/repository/{repo_id}/status")
                if status_res.status_code == 200:
                    status = status_res.json()['status']
                    if status == "completed":
                        print("\n✓ Ingestion completed!")
                        break
                    elif status == "failed":
                        print("\n✗ Ingestion failed!")
                        return None
                
                if time.time() - start_time > 300:  # 5 minute timeout
                    print("\n✗ Timeout waiting for ingestion")
                    return None
                    
                print(".", end="", flush=True)
                time.sleep(2)
            except Exception as e:
                print(f"\nError polling status: {e}")
                return None
                
        return repo_id
    else:
        print(f"\n✗ Failed: {response.status_code}")
        print(response.text)
        return None


def test_query_requests():
    """Test querying the requests repository."""
    print("\n" + "="*60)
    print("Testing query on requests repository...")
    
    payload = {
        "messages": [
            {"role": "user", "content": "How do I make a simple GET request using this library? Show me a code example based on the docs."}
        ]
    }
    
    try:
        response = requests.post(f"{API_URL}/chat", json=payload)
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Could not connect to {API_URL}")
        return False
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Query successful")
        print(f"\nResponse:\n{data['response']}")
        
        if data.get('usage'):
            usage = data['usage']
            print(f"\nToken usage:")
            print(f"  - Input: {usage['input_tokens']}")
            print(f"  - Output: {usage['output_tokens']}")
            print(f"  - Total: {usage['total_tokens']}")
        return True
    else:
        print(f"\n✗ Query failed: {response.status_code}")
        print(response.text)
        return False


def test_get_active_repo():
    """Check which repository is currently active."""
    print("\n" + "="*60)
    print("Checking active repository...")
    
    try:
        response = requests.get(f"{API_URL}/active-repository")
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Could not connect to {API_URL}")
        return None
    
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Active repository: {data['active_repo_id']}")
        return data['active_repo_id']
    else:
        print(f"✗ Failed: {response.status_code}")
        return None


def main():
    """Run all tests."""
    print("=" * 60)
    print("Requests GitHub Repository Ingestion Test")
    print("=" * 60)
    
    # Get current active repo
    initial_repo = test_get_active_repo()
    if initial_repo is None and False: # Skip check if server down, but we handled connection error above
       pass

    # Ingest requests
    requests_repo_id = test_requests_github_ingestion()
    if not requests_repo_id:
        print("\n✗ Requests ingestion failed")
        return
    
    # Verify it's now active
    active_repo = test_get_active_repo()
    if active_repo != requests_repo_id:
        print(f"\n✗ Active repo mismatch: expected {requests_repo_id}, got {active_repo}")
        return
    
    # Query requests
    if not test_query_requests():
        print("\n✗ Query test failed")
        return
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print(f"\nRequests repository successfully ingested and queryable!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
