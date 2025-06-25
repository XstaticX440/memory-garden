#!/usr/bin/env python3
import requests
import json

# Test Memory Garden API
print("Testing Memory Garden API...")

# Test 1: Health check
try:
    response = requests.get("http://localhost:8000/health")
    print(f"API Health: {response.json()}")
except Exception as e:
    print(f"API Health Error: {e}")

# Test 2: Store memory
try:
    memory_data = {
        "content": "Testing Memory Garden persistence system",
        "user_id": "test_user",
        "metadata": {"test": True}
    }
    response = requests.post("http://localhost:8000/memory/store", json=memory_data)
    print(f"Store Memory: {response.json()}")
except Exception as e:
    print(f"Store Memory Error: {e}")

# Test 3: Query memory
try:
    query_data = {
        "query": "Memory Garden",
        "user_id": "test_user",
        "max_results": 5
    }
    response = requests.post("http://localhost:8000/ask", json=query_data)
    result = response.json()
    print(f"Query Memory: Found {len(result.get('context', []))} memories")
except Exception as e:
    print(f"Query Memory Error: {e}")

print("Test complete!") 