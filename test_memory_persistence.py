#!/usr/bin/env python3
"""
Memory Garden API Test Script
Tests the memory persistence functionality
"""

import requests
import json
import time

# API Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def test_health():
    """Test API health endpoint"""
    print("🔍 Testing API Health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print(f"✅ API Health: {response.json()}")
            return True
        else:
            print(f"❌ API Health failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Health error: {e}")
        return False

def test_store_memory():
    """Test storing a memory"""
    print("\n💾 Testing Memory Storage...")
    
    test_memory = {
        "content": "User is building a Memory Garden system for AI persistence. The system includes ChromaDB for vector storage, FastAPI backend, and OpenWebUI frontend with ChatGPT integration.",
        "user_id": "test_user_001",
        "metadata": {
            "source": "chatgpt",
            "importance": "high",
            "topic": "ai_memory_system",
            "timestamp": time.time()
        }
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/memory/store",
            headers=HEADERS,
            json=test_memory
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Memory stored successfully: {result}")
            return result.get("memory_id")
        else:
            print(f"❌ Memory storage failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Memory storage error: {e}")
        return None

def test_query_memory(memory_id):
    """Test querying memories"""
    print("\n🔍 Testing Memory Query...")
    
    query = {
        "query": "Memory Garden system AI persistence",
        "user_id": "test_user_001",
        "max_results": 5
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/ask",
            headers=HEADERS,
            json=query
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Memory query successful:")
            print(f"   Context found: {len(result.get('context', []))} memories")
            for i, memory in enumerate(result.get('context', [])):
                print(f"   Memory {i+1}: {memory.get('content', '')[:100]}...")
            return result
        else:
            print(f"❌ Memory query failed: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ Memory query error: {e}")
        return None

def test_middleware_health():
    """Test middleware health"""
    print("\n🔍 Testing Middleware Health...")
    try:
        response = requests.get("http://localhost:3001/health")
        if response.status_code == 200:
            print(f"✅ Middleware Health: {response.json()}")
            return True
        else:
            print(f"❌ Middleware Health failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Middleware Health error: {e}")
        return False

def test_openwebui_integration():
    """Test OpenWebUI integration endpoint"""
    print("\n🔍 Testing OpenWebUI Integration...")
    
    test_conversation = {
        "user_id": "test_user_001",
        "conversation": [
            {
                "role": "user",
                "content": "Tell me about the Memory Garden system we're building"
            },
            {
                "role": "assistant", 
                "content": "The Memory Garden system is an AI persistence platform that allows ChatGPT to maintain context across sessions using ChromaDB for vector storage."
            }
        ]
    }
    
    try:
        response = requests.post(
            "http://localhost:3001/store_conversation",
            headers=HEADERS,
            json=test_conversation
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ OpenWebUI integration successful: {result}")
            return True
        else:
            print(f"❌ OpenWebUI integration failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ OpenWebUI integration error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 Memory Garden Persistence Test Suite")
    print("=" * 50)
    
    # Test 1: API Health
    if not test_health():
        print("❌ API is not healthy. Exiting.")
        return
    
    # Test 2: Middleware Health
    if not test_middleware_health():
        print("⚠️  Middleware may not be running")
    
    # Test 3: Store Memory
    memory_id = test_store_memory()
    if not memory_id:
        print("❌ Memory storage failed. Exiting.")
        return
    
    # Test 4: Query Memory
    query_result = test_query_memory(memory_id)
    if not query_result:
        print("❌ Memory query failed.")
        return
    
    # Test 5: OpenWebUI Integration
    test_openwebui_integration()
    
    print("\n" + "=" * 50)
    print("🎉 Memory Garden Persistence Test Complete!")
    print("\nNext Steps:")
    print("1. Open OpenWebUI in your browser")
    print("2. Start a conversation with ChatGPT")
    print("3. Ask about the Memory Garden system")
    print("4. Verify ChatGPT remembers previous context")

if __name__ == "__main__":
    main() 