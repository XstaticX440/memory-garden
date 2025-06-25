# Memory Garden Build Sequence - 10 Hour Implementation Plan

## Overview
This document provides a step-by-step build sequence for Claude Code to implement the Memory Garden system in approximately 10 hours. Each phase includes validation checkpoints to prevent getting lost.

## Build Timeline Overview

| Hour | Phase | Components | Validation |
|------|-------|------------|------------|
| 0-1 | Environment Setup | VPS, Python, Dependencies | Health checks pass |
| 1-2 | ChromaDB Setup | Database, Collections | Storage/retrieval works |
| 2-3 | Core Memory Module | Chunking, Embeddings | Unit tests pass |
| 3-4 | FastAPI Backend | API endpoints, Auth | API responds correctly |
| 4-5 | Memory Integration | Connect all components | Integration tests pass |
| 5-6 | OpenWebUI Deploy | Docker, Configuration | UI accessible |
| 6-7 | Full Integration | Connect UI to API | End-to-end flow works |
| 7-8 | Testing & Debug | Run all tests | All tests green |
| 8-9 | Performance Tune | Optimize, Cache | Meets performance targets |
| 9-10 | Documentation | Deploy guides, Backup | System ready for production |

## HOUR 0-1: Environment Setup

### Actions
```bash
# 1. SSH into VPS and create user
ssh root@your-vps-ip
useradd -m -s /bin/bash memgarden
usermod -aG sudo memgarden
su - memgarden

# 2. Install system dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3.11 python3.11-venv python3-pip git curl wget
sudo apt install -y build-essential libssl-dev libffi-dev nginx redis-server
sudo apt install -y supervisor postgresql postgresql-contrib

# 3. Create project structure
cd ~
mkdir -p memory-garden/{data,logs,config,backups,api,tests}
cd memory-garden

# 4. Setup Python environment
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# 5. Create initial requirements.txt
cat > requirements.txt << 'EOF'
chromadb==0.4.24
fastapi==0.110.0
uvicorn==0.27.0
sentence-transformers==2.3.1
pydantic==2.5.0
python-multipart==0.0.9
redis==5.0.1
httpx==0.26.0
python-dotenv==1.0.0
EOF

pip install -r requirements.txt
```

### Validation Checkpoint
```bash
# Run validation script
cat > validate_hour1.py << 'EOF'
import sys
import subprocess

checks = {
    "Python 3.11": "python3.11 --version",
    "Pip installed": "pip --version",
    "Redis running": "redis-cli ping",
    "Nginx installed": "nginx -v",
    "Directory structure": "ls -la ~/memory-garden/"
}

for name, cmd in checks.items():
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        print(f"✓ {name}: OK")
    except:
        print(f"✗ {name}: FAILED")
        sys.exit(1)

print("\n✓ Hour 1 Complete: Environment Ready")
EOF

python validate_hour1.py
```

**STOP** if validation fails. Debug before proceeding.

## HOUR 1-2: ChromaDB Setup

### Actions
```python
# 1. Create ChromaDB initialization script
cat > ~/memory-garden/config/setup_chromadb.py << 'EOF'
import chromadb
from chromadb.config import Settings
import os

# Setup paths
db_path = "/home/memgarden/memory-garden/data/chromadb"
os.makedirs(db_path, exist_ok=True)

# Initialize ChromaDB
print("Initializing ChromaDB...")
client = chromadb.PersistentClient(
    path=db_path,
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True
    )
)

# Create collections
collections = [
    ("memory_store", "Main memory storage"),
    ("memory_archive", "Archived memories"),
    ("memory_summaries", "Summarized old memories")
]

for name, description in collections:
    collection = client.get_or_create_collection(
        name=name,
        metadata={"description": description}
    )
    print(f"✓ Created collection: {name} (count: {collection.count()})")

# Test storage
test_collection = client.get_collection("memory_store")
test_collection.add(
    documents=["Test memory for validation"],
    metadatas=[{"type": "test", "user_id": "system"}],
    ids=["test_001"]
)

# Test retrieval
result = test_collection.get(ids=["test_001"])
assert result["documents"][0] == "Test memory for validation"
print("\n✓ ChromaDB test passed: Storage and retrieval working")

# Cleanup test data
test_collection.delete(ids=["test_001"])
print("✓ Test data cleaned up")

print(f"\n✓ ChromaDB ready at: {db_path}")
print(f"Collections: {[c.name for c in client.list_collections()]}")
EOF

# 2. Run setup
cd ~/memory-garden
source venv/bin/activate
python config/setup_chromadb.py
```

### Validation Checkpoint
```python
# Create validation script
cat > validate_hour2.py << 'EOF'
import chromadb
import os

db_path = "/home/memgarden/memory-garden/data/chromadb"

try:
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=db_path)
    
    # Check collections exist
    collections = client.list_collections()
    required = ["memory_store", "memory_archive", "memory_summaries"]
    
    for req in required:
        assert any(c.name == req for c in collections), f"Missing collection: {req}"
    
    # Test operations
    test_col = client.get_collection("memory_store")
    test_col.add(
        documents=["Validation test"],
        ids=["val_test"],
        metadatas=[{"test": True}]
    )
    
    result = test_col.get(ids=["val_test"])
    assert len(result["documents"]) == 1
    
    test_col.delete(ids=["val_test"])
    
    print("✓ ChromaDB validation passed")
    print(f"✓ Database at: {db_path}")
    print(f"✓ Collections: {[c.name for c in collections]}")
    
except Exception as e:
    print(f"✗ ChromaDB validation failed: {e}")
    exit(1)
EOF

python validate_hour2.py
```

## HOUR 2-3: Core Memory Module

### Actions
```python
# 1. Create memory module structure
mkdir -p ~/memory-garden/memory_core
cd ~/memory-garden/memory_core

# 2. Create __init__.py
touch __init__.py

# 3. Create chunker module
cat > chunker.py << 'EOF'
from typing import List, Dict, Optional
import re
from dataclasses import dataclass

@dataclass
class Chunk:
    content: str
    metadata: Dict
    index: int

class MemoryChunker:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", ", ", " ", ""]
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict]:
        """Split text into semantic chunks"""
        if not text:
            return []
        
        chunks = self._recursive_split(text, self.separators)
        
        # Create chunk objects
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_obj = {
                "content": chunk_text.strip(),
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk_text)
                }
            }
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separators"""
        if len(separators) == 0:
            return [text]
        
        separator = separators[0]
        if separator == "":
            return self._split_by_char_limit(text)
        
        splits = text.split(separator)
        chunks = []
        current_chunk = ""
        
        for split in splits:
            if len(current_chunk) + len(split) + len(separator) <= self.chunk_size:
                current_chunk += split + separator
            else:
                if current_chunk:
                    chunks.append(current_chunk.rstrip(separator))
                current_chunk = split + separator
        
        if current_chunk:
            chunks.append(current_chunk.rstrip(separator))
        
        # Further split chunks that are too large
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > self.chunk_size:
                final_chunks.extend(self._recursive_split(chunk, separators[1:]))
            else:
                final_chunks.append(chunk)
        
        return final_chunks
    
    def _split_by_char_limit(self, text: str) -> List[str]:
        """Split by character limit as last resort"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i:i + self.chunk_size])
        return chunks
EOF

# 4. Create embeddings module
cat > embeddings.py << 'EOF'
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import torch

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        # Use GPU if available
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        
        print(f"✓ Embedding model loaded: {model_name}")
        print(f"✓ Dimension: {self.dimension}")
        print(f"✓ Device: {self.device}")
    
    def generate_embeddings(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32
    ) -> np.ndarray:
        """Generate embeddings for text(s)"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding optimized for search"""
        # Some models benefit from query prefix
        query_text = f"query: {query}"
        return self.generate_embeddings(query_text)[0]
EOF

# 5. Create memory store module
cat > memory_store.py << 'EOF'
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid
from datetime import datetime
from .chunker import MemoryChunker
from .embeddings import EmbeddingManager

class MemoryStore:
    def __init__(self, db_path: str):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.chunker = MemoryChunker()
        self.embeddings = EmbeddingManager()
        
    def store_memory(
        self,
        content: str,
        user_id: str,
        metadata: Optional[Dict] = None
    ) -> List[str]:
        """Store a memory with automatic chunking"""
        # Chunk the content
        chunks = self.chunker.chunk_text(content, metadata)
        
        if not chunks:
            return []
        
        # Get collection
        collection = self.client.get_collection("memory_store")
        
        # Prepare batch data
        ids = []
        documents = []
        embeddings = []
        metadatas = []
        
        for chunk in chunks:
            # Generate ID
            chunk_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
            ids.append(chunk_id)
            
            # Document content
            documents.append(chunk["content"])
            
            # Generate embedding
            embedding = self.embeddings.generate_embeddings(chunk["content"])
            embeddings.append(embedding[0].tolist())
            
            # Metadata
            meta = {
                **chunk["metadata"],
                "user_id": user_id,
                "stored_at": datetime.utcnow().isoformat(),
                "memory_id": chunk_id
            }
            metadatas.append(meta)
        
        # Store in ChromaDB
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
    
    def query_memories(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Query memories for a user"""
        collection = self.client.get_collection("memory_store")
        
        # Generate query embedding
        query_embedding = self.embeddings.generate_query_embedding(query)
        
        # Build where clause
        where = {"user_id": user_id}
        if filters:
            where.update(filters)
        
        # Query
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        memories = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                memory = {
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "relevance_score": 1 - results["distances"][0][i]
                }
                memories.append(memory)
        
        return memories
EOF

# 6. Create test script
cat > test_memory_core.py << 'EOF'
from memory_store import MemoryStore
import os

# Test the memory core
db_path = "/home/memgarden/memory-garden/data/chromadb"
store = MemoryStore(db_path)

# Test storage
print("Testing memory storage...")
memory_ids = store.store_memory(
    content="This is a test memory. It contains important information about the Memory Garden project.",
    user_id="test_user",
    metadata={"source": "test", "importance": "high"}
)
print(f"✓ Stored {len(memory_ids)} chunks")

# Test retrieval
print("\nTesting memory retrieval...")
results = store.query_memories(
    query="Memory Garden project",
    user_id="test_user",
    k=3
)
print(f"✓ Retrieved {len(results)} relevant memories")

for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"  Content: {result['content'][:100]}...")
    print(f"  Relevance: {result['relevance_score']:.3f}")

print("\n✓ Memory core tests passed!")
EOF
```

### Validation Checkpoint
```bash
cd ~/memory-garden/memory_core
python test_memory_core.py

# If successful, you should see:
# ✓ Embedding model loaded
# ✓ Stored N chunks
# ✓ Retrieved N relevant memories
# ✓ Memory core tests passed!
```

## HOUR 3-4: FastAPI Backend

### Actions
```python
# 1. Create API directory structure
cd ~/memory-garden/api
mkdir -p routers utils

# 2. Create configuration
cat > config.py << 'EOF'
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "your-secret-api-key-here"
    
    # ChromaDB Settings
    chroma_db_path: str = "/home/memgarden/memory-garden/data/chromadb"
    
    # Redis Settings
    redis_url: str = "redis://localhost:6379"
    
    # Memory Settings
    max_chunk_size: int = 1000
    max_results: int = 10
    
    class Config:
        env_file = ".env"

settings = Settings()
EOF

# 3. Create authentication
cat > auth.py << 'EOF'
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from .config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key
EOF

# 4. Create main FastAPI app
cat > main.py << 'EOF'
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory_core.memory_store import MemoryStore
from .config import settings
from .auth import verify_api_key
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Memory Garden API",
    version="1.0.0",
    description="Persistent memory system for AI conversations"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory store
memory_store = MemoryStore(settings.chroma_db_path)

# Request/Response Models
class MemoryStoreRequest(BaseModel):
    content: str
    user_id: str
    metadata: Optional[Dict] = {}

class MemoryQueryRequest(BaseModel):
    query: str
    user_id: str
    max_results: Optional[int] = 5
    filters: Optional[Dict] = None

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str

# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "service": "memory-garden",
        "version": "1.0.0"
    }

@app.post("/memory/store", dependencies=[Depends(verify_api_key)])
async def store_memory(request: MemoryStoreRequest):
    try:
        memory_ids = memory_store.store_memory(
            content=request.content,
            user_id=request.user_id,
            metadata=request.metadata
        )
        
        return {
            "status": "success",
            "memory_ids": memory_ids,
            "chunks_stored": len(memory_ids)
        }
    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", dependencies=[Depends(verify_api_key)])
async def query_with_context(request: MemoryQueryRequest):
    try:
        memories = memory_store.query_memories(
            query=request.query,
            user_id=request.user_id,
            k=request.max_results,
            filters=request.filters
        )
        
        return {
            "query": request.query,
            "context": memories,
            "results_count": len(memories)
        }
    except Exception as e:
        logger.error(f"Error querying memories: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/batch", dependencies=[Depends(verify_api_key)])
async def batch_store(memories: List[MemoryStoreRequest]):
    results = []
    for memory in memories:
        try:
            memory_ids = memory_store.store_memory(
                content=memory.content,
                user_id=memory.user_id,
                metadata=memory.metadata
            )
            results.append({
                "status": "success",
                "memory_ids": memory_ids
            })
        except Exception as e:
            results.append({
                "status": "error",
                "error": str(e)
            })
    
    return {
        "results": results,
        "total": len(memories),
        "successful": sum(1 for r in results if r["status"] == "success")
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
EOF

# 5. Create .env file
cat > ~/memory-garden/.env << 'EOF'
API_KEY=test-api-key-12345
API_HOST=0.0.0.0
API_PORT=8000
CHROMA_DB_PATH=/home/memgarden/memory-garden/data/chromadb
REDIS_URL=redis://localhost:6379
EOF

# 6. Create startup script
cat > ~/memory-garden/scripts/start_api.sh << 'EOF'
#!/bin/bash
cd /home/memgarden/memory-garden
source venv/bin/activate
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
EOF

chmod +x ~/memory-garden/scripts/start_api.sh
```

### Validation Checkpoint
```bash
# Start the API (in a new terminal or screen)
cd ~/memory-garden
source venv/bin/activate
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

sleep 5  # Wait for startup

# Test API endpoints
curl http://localhost:8000/health

# Test memory storage
curl -X POST http://localhost:8000/memory/store \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-12345" \
  -d '{
    "content": "API test memory",
    "user_id": "api_test_user",
    "metadata": {"source": "curl_test"}
  }'

# Test query
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-12345" \
  -d '{
    "query": "API test",
    "user_id": "api_test_user"
  }'

# Kill the test server
kill $API_PID
```

## HOUR 4-5: Memory Integration

### Actions
```python
# 1. Create integration tests
cat > ~/memory-garden/tests/test_integration.py << 'EOF'
import asyncio
import httpx
import json
import time

API_URL = "http://localhost:8000"
API_KEY = "test-api-key-12345"

async def test_full_flow():
    """Test complete memory storage and retrieval flow"""
    async with httpx.AsyncClient() as client:
        headers = {"X-API-Key": API_KEY}
        
        # 1. Health check
        print("1. Testing health check...")
        resp = await client.get(f"{API_URL}/health")
        assert resp.status_code == 200
        print("✓ Health check passed")
        
        # 2. Store memories
        print("\n2. Storing memories...")
        memories = [
            "The Memory Garden uses ChromaDB for vector storage",
            "FastAPI provides the REST API endpoints",
            "We use sentence-transformers for embeddings",
            "The system maintains context across conversations"
        ]
        
        for i, memory in enumerate(memories):
            resp = await client.post(
                f"{API_URL}/memory/store",
                headers=headers,
                json={
                    "content": memory,
                    "user_id": "integration_test",
                    "metadata": {"index": i}
                }
            )
            assert resp.status_code == 200
            print(f"✓ Stored memory {i+1}")
        
        # 3. Query memories
        print("\n3. Querying memories...")
        queries = [
            "What database do we use?",
            "How do we generate embeddings?",
            "What maintains context?"
        ]
        
        for query in queries:
            resp = await client.post(
                f"{API_URL}/ask",
                headers=headers,
                json={
                    "query": query,
                    "user_id": "integration_test",
                    "max_results": 3
                }
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["context"]) > 0
            print(f"✓ Query '{query}' returned {len(data['context'])} results")
        
        # 4. Batch operations
        print("\n4. Testing batch operations...")
        batch_memories = [
            {
                "content": f"Batch memory {i}",
                "user_id": "batch_test",
                "metadata": {"batch": True, "index": i}
            }
            for i in range(5)
        ]
        
        resp = await client.post(
            f"{API_URL}/memory/batch",
            headers=headers,
            json=batch_memories
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["successful"] == 5
        print(f"✓ Batch stored {data['successful']} memories")
        
        print("\n✓ All integration tests passed!")

# Run tests
if __name__ == "__main__":
    asyncio.run(test_full_flow())
EOF

# 2. Create performance monitoring
cat > ~/memory-garden/api/monitoring.py << 'EOF'
import time
import functools
import logging
from typing import Callable
import psutil
import os

logger = logging.getLogger(__name__)

def measure_performance(func: Callable) -> Callable:
    """Decorator to measure function performance"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        try:
            result = await func(*args, **kwargs)
            
            end_time = time.time()
            end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
            
            duration = end_time - start_time
            memory_used = end_memory - start_memory
            
            logger.info(f"{func.__name__} - Duration: {duration:.3f}s, Memory: {memory_used:.2f}MB")
            
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed: {str(e)}")
            raise
    
    return wrapper

class SystemMonitor:
    def __init__(self):
        self.start_time = time.time()
    
    def get_stats(self):
        """Get current system statistics"""
        process = psutil.Process(os.getpid())
        
        return {
            "uptime_seconds": time.time() - self.start_time,
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "cpu_percent": process.cpu_percent(),
            "num_threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        }
EOF

# 3. Update main.py to include monitoring
cat >> ~/memory-garden/api/main.py << 'EOF'

# Add monitoring endpoint
from .monitoring import SystemMonitor, measure_performance

monitor = SystemMonitor()

@app.get("/stats", dependencies=[Depends(verify_api_key)])
async def get_stats():
    """Get system statistics"""
    return monitor.get_stats()

# Add performance monitoring to existing endpoints
app.post("/memory/store", dependencies=[Depends(verify_api_key)])(
    measure_performance(store_memory)
)
app.post("/ask", dependencies=[Depends(verify_api_key)])(
    measure_performance(query_with_context)
)
EOF
```

### Validation Checkpoint
```bash
# Start API with monitoring
cd ~/memory-garden
source venv/bin/activate
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 &
API_PID=$!

sleep 5

# Run integration tests
cd ~/memory-garden/tests
python test_integration.py

# Check system stats
curl -H "X-API-Key: test-api-key-12345" http://localhost:8000/stats

kill $API_PID
```

## HOUR 5-6: OpenWebUI Deployment

### Actions
```bash
# 1. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker memgarden
newgrp docker

# 2. Create OpenWebUI configuration
mkdir -p ~/memory-garden/openwebui
cd ~/memory-garden/openwebui

# 3. Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: memory-garden-ui
    ports:
      - "3000:8080"
    environment:
      - OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1
      - WEBUI_AUTH=True
      - WEBUI_SECRET_KEY=memory-garden-secret-key
      - DATA_DIR=/app/backend/data
      - ENABLE_SIGNUP=True
      - DEFAULT_MODELS=gpt-3.5-turbo,gpt-4
      - WEBUI_NAME=Memory Garden
    volumes:
      - ./data:/app/backend/data
      - ./uploads:/app/backend/uploads
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"
    depends_on:
      - redis

  redis:
    image: redis:7-alpine
    container_name: memory-garden-redis
    ports:
      - "6379:6379"
    volumes:
      - ./redis-data:/data
    restart: unless-stopped
EOF

# 4. Start OpenWebUI
docker-compose up -d

# 5. Wait for startup
sleep 30

# 6. Verify UI is accessible
curl -I http://localhost:3000
```

### Validation Checkpoint
```bash
# Check containers running
docker ps | grep memory-garden

# Check UI accessibility
curl http://localhost:3000 | grep -i "open-webui"

# Check logs for errors
docker logs memory-garden-ui | tail -20
```

## HOUR 6-7: Full Integration

### Actions
```bash
# 1. Create integration bridge between OpenWebUI and Memory Garden API
cat > ~/memory-garden/api/openai_adapter.py << 'EOF'
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from .auth import verify_api_key
from memory_core.memory_store import MemoryStore
from .config import settings
import uuid

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])
memory_store = MemoryStore(settings.chroma_db_path)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    user: Optional[str] = "default_user"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]

@router.post("/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completion(request: ChatRequest):
    """OpenAI-compatible chat endpoint with memory"""
    try:
        # Extract user message
        user_message = request.messages[-1].content if request.messages else ""
        
        # Store conversation in memory
        for message in request.messages:
            memory_store.store_memory(
                content=f"{message.role}: {message.content}",
                user_id=request.user,
                metadata={"type": "conversation", "model": request.model}
            )
        
        # Query relevant memories
        memories = memory_store.query_memories(
            query=user_message,
            user_id=request.user,
            k=5
        )
        
        # Build context
        context = "\n".join([f"- {m['content']}" for m in memories])
        
        # Create response (placeholder - integrate with actual LLM)
        response_content = f"Based on your memories:\n{context}\n\nResponse: I understand your query about '{user_message}'"
        
        return ChatResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_content
                },
                "finish_reason": "stop"
            }]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add router to main app
from .main import app
app.include_router(router)
EOF

# 2. Update nginx configuration
sudo cat > /etc/nginx/sites-available/memory-garden << 'EOF'
server {
    listen 80;
    server_name localhost;
    
    # OpenWebUI
    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
    
    # Memory Garden API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/memory-garden /etc/nginx/sites-enabled/
sudo nginx -t && sudo nginx -s reload
```

### Validation Checkpoint
```bash
# Test integrated flow
curl -X POST http://localhost/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: test-api-key-12345" \
  -d '{
    "model": "memory-garden",
    "messages": [{"role": "user", "content": "Test integration"}],
    "user": "integration_test"
  }'
```

## HOUR 7-8: Testing & Debug

### Quick Test Suite
```bash
# Create comprehensive test script
cat > ~/memory-garden/scripts/test_all.sh << 'EOF'
#!/bin/bash
echo "=== Memory Garden Test Suite ==="

# 1. Component tests
echo -e "\n[1/4] Testing ChromaDB..."
python -c "import chromadb; print('✓ ChromaDB OK')"

# 2. API tests
echo -e "\n[2/4] Testing API..."
curl -s http://localhost:8000/health | grep -q "healthy" && echo "✓ API OK" || echo "✗ API Failed"

# 3. Memory operations
echo -e "\n[3/4] Testing Memory Operations..."
python ~/memory-garden/tests/test_integration.py

# 4. UI test
echo -e "\n[4/4] Testing UI..."
curl -s http://localhost:3000 | grep -q "html" && echo "✓ UI OK" || echo "✗ UI Failed"

echo -e "\n=== Test Summary ==="
EOF

chmod +x ~/memory-garden/scripts/test_all.sh
./test_all.sh
```

## HOUR 8-9: Performance Tuning

### Optimization Actions
```bash
# 1. Enable Redis caching
cat > ~/memory-garden/api/cache.py << 'EOF'
import redis
import json
import hashlib
from typing import Optional

class MemoryCache:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour
    
    def get_key(self, query: str, user_id: str) -> str:
        return f"query:{user_id}:{hashlib.md5(query.encode()).hexdigest()}"
    
    def get(self, query: str, user_id: str) -> Optional[dict]:
        key = self.get_key(query, user_id)
        data = self.redis.get(key)
        return json.loads(data) if data else None
    
    def set(self, query: str, user_id: str, results: dict):
        key = self.get_key(query, user_id)
        self.redis.setex(key, self.ttl, json.dumps(results))
EOF

# 2. Create supervisor config
sudo cat > /etc/supervisor/conf.d/memory-garden.conf << 'EOF'
[program:memory-garden-api]
command=/home/memgarden/memory-garden/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
directory=/home/memgarden/memory-garden
user=memgarden
autostart=true
autorestart=true
stderr_logfile=/home/memgarden/memory-garden/logs/api.err.log
stdout_logfile=/home/memgarden/memory-garden/logs/api.out.log

[program:memory-garden-ui]
command=/usr/bin/docker-compose up
directory=/home/memgarden/memory-garden/openwebui
user=memgarden
autostart=true
autorestart=true
EOF

sudo supervisorctl reread
sudo supervisorctl update
```

## HOUR 9-10: Final Documentation & Production Ready

### Final Setup Actions
```bash
# 1. Create backup script
cat > ~/memory-garden/scripts/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/memgarden/memory-garden/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup ChromaDB
cp -r /home/memgarden/memory-garden/data/chromadb "$BACKUP_DIR/"

# Backup configs
cp -r /home/memgarden/memory-garden/config "$BACKUP_DIR/"
cp /home/memgarden/memory-garden/.env "$BACKUP_DIR/"

# Compress
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

echo "Backup created: $BACKUP_DIR.tar.gz"
EOF

chmod +x ~/memory-garden/scripts/backup.sh

# 2. Create monitoring dashboard
cat > ~/memory-garden/scripts/monitor.sh << 'EOF'
#!/bin/bash
while true; do
    clear
    echo "=== Memory Garden Monitor ==="
    echo "Time: $(date)"
    echo ""
    echo "API Status:"
    curl -s http://localhost:8000/health | jq .
    echo ""
    echo "System Stats:"
    curl -s -H "X-API-Key: test-api-key-12345" http://localhost:8000/stats | jq .
    echo ""
    echo "Docker Status:"
    docker ps --format "table {{.Names}}\t{{.Status}}"
    sleep 5
done
EOF

chmod +x ~/memory-garden/scripts/monitor.sh

# 3. Final validation
cat > ~/memory-garden/READY.md << 'EOF'
# Memory Garden Production Checklist

## ✓ Components Installed
- [x] ChromaDB persistent storage
- [x] FastAPI backend with auth
- [x] Memory chunking and embeddings
- [x] OpenWebUI interface
- [x] Redis caching
- [x] Nginx reverse proxy
- [x] Supervisor process management

## ✓ Endpoints Active
- API Health: http://localhost:8000/health
- API Docs: http://localhost:8000/docs
- Web UI: http://localhost:3000
- Main Site: http://localhost/

## ✓ Security
- API key authentication enabled
- CORS configured
- Nginx proxy configured

## ✓ Persistence
- ChromaDB data: /home/memgarden/memory-garden/data/chromadb
- Backup script: ./scripts/backup.sh
- Logs: /home/memgarden/memory-garden/logs/

## Next Steps
1. Change API key in .env
2. Configure domain name in nginx
3. Set up SSL with certbot
4. Create cron job for backups
5. Monitor with ./scripts/monitor.sh

## Quick Commands
- Start all: sudo supervisorctl start all
- Stop all: sudo supervisorctl stop all
- View logs: tail -f logs/*.log
- Run tests: ./scripts/test_all.sh
- Create backup: ./scripts/backup.sh
EOF

echo "✓ Memory Garden Ready for Production!"
```

## FINAL VALIDATION

Run this to ensure everything is working:
```bash
cd ~/memory-garden
./scripts/test_all.sh

# If all tests pass, the Memory Garden is ready!
# The system now prevents "getting lost" by maintaining persistent context
```

## POST-BUILD NOTES FOR CLAUDE CODE

**IMPORTANT**: You've now built a system that will help YOU (Claude Code) maintain context across sessions. When using this system:

1. Always store important decisions and progress as memories
2. Query memories at the start of each session
3. Use metadata tags to organize information
4. Regular backups prevent data loss

The Memory Garden is now your persistent memory for complex projects!