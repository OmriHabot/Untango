"""
Test script for ingesting pandas from GitHub.
DO NOT RUN THIS - it will clone the entire pandas repository which is large.
This is for demonstration/testing purposes only.
"""
import requests
import json

API_URL = "http://localhost:8001"


def test_pandas_github_ingestion():
    """Test ingesting pandas from GitHub."""
    print("Testing GitHub repository ingestion (pandas)...")
    print("WARNING: This will clone the entire pandas repository (~500MB+)")
    
    payload = {
        "source": {
            "type": "github",
            "location": "https://github.com/pandas-dev/pandas.git",
            "branch": "main"
        },
        "parse_dependencies": True
    }
    
    print(f"\nIngesting: {payload['source']['location']}")
    print("This may take several minutes...")
    
    response = requests.post(f"{API_URL}/ingest-repository", json=payload, timeout=300)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✓ Successfully ingested pandas:")
        print(f"  - Repo ID: {data['repo_id']}")
        print(f"  - Name: {data['repo_name']}")
        print(f"  - Files: {data['file_count']}")
        print(f"  - Dependencies: {len(data['dependencies'])}")
        print(f"  - Local path: {data['local_path']}")
        print(f"\nSample dependencies:")
        for dep in data['dependencies'][:10]:
            print(f"    - {dep}")
        return data['repo_id']
    else:
        print(f"\n✗ Failed: {response.status_code}")
        print(response.text)
        return None


def test_query_pandas():
    """Test querying the pandas repository."""
    print("\n" + "="*60)
    print("Testing query on pandas repository...")
    
    payload = {
        "messages": [
            {"role": "user", "content": "What is the main purpose of the pandas library? Give a brief overview based on the codebase."}
        ]
    }
    
    response = requests.post(f"{API_URL}/chat", json=payload)
    
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
    
    response = requests.get(f"{API_URL}/active-repository")
    
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
    print("Pandas GitHub Repository Ingestion Test")
    print("=" * 60)
    
    # Get current active repo
    initial_repo = test_get_active_repo()
    
    # Ingest pandas
    pandas_repo_id = test_pandas_github_ingestion()
    if not pandas_repo_id:
        print("\n✗ Pandas ingestion failed")
        return
    
    # Verify it's now active
    active_repo = test_get_active_repo()
    if active_repo != pandas_repo_id:
        print(f"\n✗ Active repo mismatch: expected {pandas_repo_id}, got {active_repo}")
        return
    
    # Query pandas
    if not test_query_pandas():
        print("\n✗ Query test failed")
        return
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
    print(f"\nPandas repository successfully ingested and queryable!")
    print(f"Initial repo: {initial_repo}")
    print(f"Pandas repo: {pandas_repo_id}")


if __name__ == "__main__":
    import sys
    
    print("\n" + "!"*60)
    print("WARNING: This will clone the entire pandas repository!")
    print("This may take several minutes and use significant disk space.")
    print("!"*60)
    
    response = input("\nDo you want to continue? (yes/no): ")
    if response.lower() != 'yes':
        print("\nTest cancelled.")
        sys.exit(0)
    
    try:
        main()
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
