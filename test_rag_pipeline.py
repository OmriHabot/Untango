#!/usr/bin/env python3
"""
Test script for the complete RAG pipeline.
Demonstrates: ingest -> query-db -> RAG response
"""
import requests
import json

BASE_URL = "http://localhost:8001"

# Sample Python code to ingest
SAMPLE_CODE = """
import hashlib
import secrets

def hash_password(password: str, salt: str = None) -> tuple:
    \"\"\"
    Hash a password using SHA-256 with a salt.
    Returns a tuple of (hashed_password, salt).
    \"\"\"
    if salt is None:
        salt = secrets.token_hex(16)
    
    password_salt = password + salt
    hashed = hashlib.sha256(password_salt.encode()).hexdigest()
    return hashed, salt

def verify_password(password: str, hashed: str, salt: str) -> bool:
    \"\"\"
    Verify a password against a hash and salt.
    Returns True if the password matches.
    \"\"\"
    computed_hash, _ = hash_password(password, salt)
    return computed_hash == hashed

class UserAuth:
    \"\"\"Handles user authentication operations.\"\"\"
    
    def __init__(self):
        self.users = {}
    
    def register_user(self, username: str, password: str):
        \"\"\"Register a new user with hashed password.\"\"\"
        if username in self.users:
            raise ValueError(f"User {username} already exists")
        
        hashed, salt = hash_password(password)
        self.users[username] = {"hash": hashed, "salt": salt}
        return True
    
    def authenticate_user(self, username: str, password: str) -> bool:
        \"\"\"Authenticate a user by verifying their password.\"\"\"
        if username not in self.users:
            return False
        
        user_data = self.users[username]
        return verify_password(password, user_data["hash"], user_data["salt"])
"""

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Status: {data['status']}")
        print(f"   ChromaDB: {data['chroma_heartbeat']}ms")
        print(f"   GCP configured: {data['gcp_configured']}")
        return True
    else:
        print(f"❌ Health check failed: {response.text}")
        return False


def test_ingest():
    """Test code ingestion"""
    print("\n📥 Testing code ingestion...")
    
    payload = {
        "code": SAMPLE_CODE,
        "filepath": "auth/user_auth.py",
        "repo_name": "secure-app"
    }
    
    response = requests.post(f"{BASE_URL}/ingest", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Ingested {data['chunks_ingested']} chunks")
        print(f"   Collection: {data['collection_name']}")
        return True
    else:
        print(f"❌ Ingestion failed: {response.text}")
        return False


def test_hybrid_search():
    """Test hybrid search"""
    print("\n🔍 Testing hybrid search...")
    
    payload = {
        "query": "how to hash passwords securely",
        "n_results": 5
    }
    
    response = requests.post(f"{BASE_URL}/query-hybrid", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {data['count']} results")
        
        for i, result in enumerate(data['results'][:3], 1):
            print(f"\n   Result {i}:")
            print(f"   - Combined score: {result['combined_score']:.3f}")
            print(f"   - Vector score: {result['vector_score']:.3f}")
            print(f"   - BM25 score: {result['bm25_score']:.3f}")
            print(f"   - Type: {result['metadata'].get('chunk_type')}")
            print(f"   - Name: {result['metadata'].get('function_name', 'N/A')}")
        return True
    else:
        print(f"❌ Hybrid search failed: {response.text}")
        return False


def test_rag_query():
    """Test complete RAG pipeline"""
    print("\n🤖 Testing complete RAG pipeline (query-db)...")
    
    payload = {
        "query": "How does this codebase implement password hashing? Explain the security approach.",
        "n_results": 10,
        "confidence_threshold": 0.5,
        "model": "gemini-2.0-flash-exp"
    }
    
    response = requests.post(f"{BASE_URL}/query-db", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ RAG Query successful!")
        print(f"\n📊 Retrieved Chunks:")
        print(f"   - Total chunks: {data['chunks_used']}")
        
        print(f"\n📚 Top chunks used:")
        for i, chunk in enumerate(data['retrieved_chunks'][:3], 1):
            print(f"   {i}. {chunk['metadata'].get('chunk_type')} - {chunk['metadata'].get('function_name', 'N/A')}")
            print(f"      Score: {chunk['combined_score']:.3f}")
        
        print(f"\n🎯 AI-Generated Answer:")
        print(f"   Model: {data['model']}")
        
        # Format the answer nicely
        answer = data['answer']
        lines = answer.split('\n')
        for line in lines[:15]:  # Show first 15 lines
            print(f"   {line}")
        
        if len(lines) > 15:
            print(f"   ... ({len(lines) - 15} more lines)")
        
        # Show token usage
        if data.get('usage'):
            usage = data['usage']
            print(f"\n💰 Token Usage:")
            print(f"   - Input tokens: {usage['input_tokens']}")
            print(f"   - Output tokens: {usage['output_tokens']}")
            print(f"   - Total tokens: {usage['total_tokens']}")
            
            # Cost calculation (adjust based on model pricing)
            input_cost = 0.1 / 1_000_000  # example: $0.10 per 1M input tokens
            output_cost = 0.4 / 1_000_000  # example: $0.40 per 1M output tokens
            cost = (usage['input_tokens'] * input_cost) + (usage['output_tokens'] * output_cost)
            print(f"   - Estimated cost: ${cost:.6f}")
        
        return True
    else:
        print(f"❌ RAG query failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return False


def main():
    """Run all tests"""
    print("=" * 70)
    print("🚀 RAG Pipeline Test Suite")
    print("=" * 70)
    
    # Test health
    if not test_health():
        print("\n❌ Health check failed. Make sure services are running:")
        print("   docker-compose up -d")
        return
    
    # Test ingestion
    if not test_ingest():
        print("\n❌ Ingestion failed. Cannot continue tests.")
        return
    
    # Test hybrid search
    test_hybrid_search()
    
    # Test RAG pipeline (the main feature)
    test_rag_query()
    
    print("\n" + "=" * 70)
    print("✨ Test suite completed!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Connection Error!")
        print("   Make sure the services are running:")
        print("   docker-compose up -d")
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

