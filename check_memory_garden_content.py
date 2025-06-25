#!/usr/bin/env python3
"""
Memory Garden Content Checker
Shows what's stored in the Memory Garden and how to access it
"""

import requests
import json
import os

# API Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def check_memory_garden_content():
    """Check what's stored in the Memory Garden"""
    print("🧠 Memory Garden Content Checker")
    print("=" * 50)
    
    # Test 1: Health Check
    print("🔍 Checking API Health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print(f"✅ API Health: {response.json()}")
        else:
            print(f"❌ API Health failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ API Health error: {e}")
        return
    
    # Test 2: Query for build specification content
    print("\n🔍 Searching for Build Specification content...")
    
    queries = [
        "Memory Garden build specification",
        "build sequence implementation plan",
        "ChromaDB setup FastAPI backend",
        "OpenWebUI integration",
        "system architecture documentation"
    ]
    
    for query in queries:
        print(f"\n📝 Querying: '{query}'")
        try:
            response = requests.post(
                f"{BASE_URL}/ask",
                headers=HEADERS,
                json={
                    "query": query,
                    "user_id": "test_user_001",
                    "max_results": 3
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                context = result.get('context', [])
                print(f"   Found {len(context)} relevant memories:")
                
                for i, memory in enumerate(context):
                    content = memory.get('content', '')
                    metadata = memory.get('metadata', {})
                    print(f"   Memory {i+1}: {content[:150]}...")
                    print(f"      Tags: {metadata.get('tags', [])}")
                    print(f"      Source: {metadata.get('source', 'unknown')}")
            else:
                print(f"   ❌ Query failed: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Query error: {e}")
    
    # Test 3: Show how to access specific content
    print("\n" + "=" * 50)
    print("🎯 How to Access Build Specification in ChatGPT:")
    print("\n1. Open OpenWebUI in your browser")
    print("2. Start a conversation with ChatGPT")
    print("3. Ask these questions to access the build spec:")
    print("\n   Example queries:")
    print("   - 'Show me the Memory Garden build sequence'")
    print("   - 'What's the system architecture for the Memory Garden?'")
    print("   - 'How do I set up ChromaDB for the Memory Garden?'")
    print("   - 'What are the API endpoints for the Memory Garden?'")
    print("   - 'Show me the installation guide for the Memory Garden'")
    
    print("\n4. ChatGPT should now have access to all the stored documentation!")
    
    # Test 4: Check if we can store the build spec now
    print("\n" + "=" * 50)
    print("💾 Storing Build Specification Content...")
    
    # Read the build specification files
    build_folder = "Memory Garden Build"
    if os.path.exists(build_folder):
        print(f"📁 Found build folder: {build_folder}")
        
        for filename in os.listdir(build_folder):
            if filename.endswith('.md'):
                filepath = os.path.join(build_folder, filename)
                print(f"\n📄 Storing: {filename}")
                
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Store the content in Memory Garden
                    memory_data = {
                        "content": f"Build Specification - {filename}: {content}",
                        "user_id": "test_user_001",
                        "metadata": {
                            "source": "build_specification",
                            "filename": filename,
                            "importance": "high",
                            "topic": "memory_garden_build",
                            "tags": ["build_spec", "documentation", "memory_garden"]
                        }
                    }
                    
                    response = requests.post(
                        f"{BASE_URL}/memory/store",
                        headers=HEADERS,
                        json=memory_data
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        print(f"   ✅ Stored: {result.get('memory_id')}")
                    else:
                        print(f"   ❌ Failed to store: {response.status_code}")
                        
                except Exception as e:
                    print(f"   ❌ Error reading file: {e}")
    else:
        print(f"❌ Build folder not found: {build_folder}")
    
    print("\n" + "=" * 50)
    print("🎉 Memory Garden Content Check Complete!")
    print("\nNext Steps:")
    print("1. The build specification is now stored in Memory Garden")
    print("2. ChatGPT can access it through OpenWebUI")
    print("3. Ask ChatGPT about any aspect of the Memory Garden build")
    print("4. Test memory persistence across sessions")

if __name__ == "__main__":
    check_memory_garden_content() 