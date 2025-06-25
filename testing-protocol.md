# Testing Protocol for Memory Garden

## Overview
This document provides comprehensive testing procedures to ensure the Memory Garden system functions correctly at component and integration levels.

## Testing Environment Setup

### Test Dependencies
```bash
# Add to requirements-test.txt
pytest==8.0.0
pytest-asyncio==0.23.0
pytest-cov==4.1.0
httpx==0.26.0
faker==22.0.0
pytest-benchmark==4.0.0
pytest-timeout==2.2.0
locust==2.20.0
```

### Test Directory Structure
```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── unit/                    # Unit tests
│   ├── test_chunker.py
│   ├── test_embeddings.py
│   ├── test_memory_store.py
│   └── test_api_endpoints.py
├── integration/             # Integration tests
│   ├── test_full_flow.py
│   ├── test_persistence.py
│   └── test_retrieval.py
├── performance/             # Performance tests
│   ├── test_load.py
│   └── test_benchmarks.py
└── fixtures/                # Test data
    ├── sample_conversations.json
    └── test_documents.txt
```

## Component Testing

### 1. ChromaDB Connection Tests
```python
# tests/unit/test_memory_store.py
import pytest
import chromadb
from chromadb.config import Settings
import tempfile
import shutil

class TestChromaDBConnection:
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_chromadb_initialization(self, temp_db_path):
        """Test ChromaDB can be initialized"""
        client = chromadb.PersistentClient(
            path=temp_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        assert client is not None
        collections = client.list_collections()
        assert len(collections) == 0
    
    def test_collection_creation(self, temp_db_path):
        """Test collection can be created"""
        client = chromadb.PersistentClient(path=temp_db_path)
        
        collection = client.create_collection(
            name="test_collection",
            metadata={"description": "Test collection"}
        )
        
        assert collection.name == "test_collection"
        assert collection.count() == 0
    
    def test_persistence(self, temp_db_path):
        """Test data persists across client instances"""
        # First client - add data
        client1 = chromadb.PersistentClient(path=temp_db_path)
        collection1 = client1.create_collection("test")
        collection1.add(
            documents=["Test document"],
            ids=["test1"]
        )
        
        # Second client - verify data exists
        client2 = chromadb.PersistentClient(path=temp_db_path)
        collection2 = client2.get_collection("test")
        
        assert collection2.count() == 1
        result = collection2.get(ids=["test1"])
        assert result["documents"][0] == "Test document"
```

### 2. Memory Chunking Tests
```python
# tests/unit/test_chunker.py
import pytest
from memory_garden.chunker import MemoryChunker

class TestMemoryChunker:
    @pytest.fixture
    def chunker(self):
        return MemoryChunker(chunk_size=100, chunk_overlap=20)
    
    def test_basic_chunking(self, chunker):
        """Test basic text chunking"""
        text = "This is a test. " * 20  # ~320 characters
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(len(c["content"]) <= 120 for c in chunks)  # Allow for overlap
    
    def test_code_block_preservation(self, chunker):
        """Test code blocks are preserved"""
        text = """
        Here is some text.
        
        ```python
        def hello():
            return "World"
        ```
        
        More text here.
        """
        chunks = chunker.chunk_text(text)
        
        # Find chunk with code
        code_chunks = [c for c in chunks if c["metadata"]["has_code"]]
        assert len(code_chunks) > 0
        assert "```" in code_chunks[0]["content"]
    
    def test_metadata_preservation(self, chunker):
        """Test metadata is preserved across chunks"""
        text = "Test text. " * 50
        metadata = {"source": "test", "user_id": "123"}
        
        chunks = chunker.chunk_text(text, metadata)
        
        assert all(c["metadata"]["source"] == "test" for c in chunks)
        assert all(c["metadata"]["user_id"] == "123" for c in chunks)
        assert all("chunk_index" in c["metadata"] for c in chunks)
    
    @pytest.mark.parametrize("text_length,expected_chunks", [
        (50, 1),    # Short text
        (500, 5),   # Medium text
        (2000, 20), # Long text
    ])
    def test_various_text_lengths(self, chunker, text_length, expected_chunks):
        """Test chunking with various text lengths"""
        text = "Word " * (text_length // 5)
        chunks = chunker.chunk_text(text)
        
        # Allow some variance in chunk count
        assert abs(len(chunks) - expected_chunks) <= 2
```

### 3. Embedding Generation Tests
```python
# tests/unit/test_embeddings.py
import pytest
import numpy as np
from memory_garden.embeddings import EmbeddingManager

class TestEmbeddings:
    @pytest.fixture
    def embedding_manager(self):
        return EmbeddingManager(model_name="all-MiniLM-L6-v2")
    
    def test_single_embedding(self, embedding_manager):
        """Test single text embedding generation"""
        text = "This is a test sentence."
        embedding = embedding_manager.generate_embeddings(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 384)  # MiniLM-L6 dimension
        assert np.allclose(np.linalg.norm(embedding), 1.0)  # Normalized
    
    def test_batch_embeddings(self, embedding_manager):
        """Test batch embedding generation"""
        texts = ["First text", "Second text", "Third text"]
        embeddings = embedding_manager.generate_embeddings(texts)
        
        assert embeddings.shape == (3, 384)
        assert all(np.allclose(np.linalg.norm(e), 1.0) for e in embeddings)
    
    def test_empty_text(self, embedding_manager):
        """Test handling of empty text"""
        embedding = embedding_manager.generate_embeddings("")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (1, 384)
    
    def test_query_embedding(self, embedding_manager):
        """Test query-specific embedding"""
        query = "What is the weather?"
        embedding = embedding_manager.generate_query_embedding(query)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (384,)  # Single embedding, not batch
    
    @pytest.mark.benchmark
    def test_embedding_performance(self, embedding_manager, benchmark):
        """Benchmark embedding generation"""
        text = "This is a sample text for performance testing."
        
        result = benchmark(embedding_manager.generate_embeddings, text)
        assert result is not None
```

### 4. API Endpoint Tests
```python
# tests/unit/test_api_endpoints.py
import pytest
from fastapi.testclient import TestClient
from memory_garden.api.main import app
import json

class TestAPIEndpoints:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        return {"X-API-Key": "test-api-key"}
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
    
    def test_store_memory(self, client, auth_headers):
        """Test memory storage endpoint"""
        payload = {
            "content": "Test memory content",
            "user_id": "test_user",
            "metadata": {
                "tags": ["test"],
                "importance": "medium"
            }
        }
        
        response = client.post(
            "/memory/store",
            json=payload,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "memory_id" in data
    
    def test_query_with_context(self, client, auth_headers):
        """Test query endpoint with context retrieval"""
        # First store a memory
        store_payload = {
            "content": "The project uses PostgreSQL for data storage",
            "user_id": "test_user",
            "metadata": {"tags": ["database"]}
        }
        client.post("/memory/store", json=store_payload, headers=auth_headers)
        
        # Then query for it
        query_payload = {
            "query": "What database are we using?",
            "user_id": "test_user",
            "include_context": True
        }
        
        response = client.post(
            "/ask",
            json=query_payload,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "context" in data
        assert len(data["context"]) > 0
    
    def test_invalid_api_key(self, client):
        """Test unauthorized access"""
        response = client.post(
            "/memory/store",
            json={"content": "test"},
            headers={"X-API-Key": "invalid-key"}
        )
        
        assert response.status_code == 401
    
    def test_batch_operations(self, client, auth_headers):
        """Test batch memory operations"""
        payload = {
            "operations": [
                {
                    "action": "store",
                    "data": {
                        "content": f"Memory {i}",
                        "user_id": "test_user",
                        "metadata": {"index": i}
                    }
                }
                for i in range(5)
            ]
        }
        
        response = client.post(
            "/memory/batch",
            json=payload,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["successful"] == 5
        assert data["failed"] == 0
```

## Integration Testing

### 1. Full Flow Test
```python
# tests/integration/test_full_flow.py
import pytest
import asyncio
from memory_garden import MemoryGarden

class TestFullFlow:
    @pytest.fixture
    async def memory_garden(self):
        """Initialize Memory Garden system"""
        mg = MemoryGarden(test_mode=True)
        await mg.initialize()
        yield mg
        await mg.cleanup()
    
    @pytest.mark.asyncio
    async def test_complete_conversation_flow(self, memory_garden):
        """Test complete conversation storage and retrieval flow"""
        user_id = "test_user_123"
        
        # Simulate a conversation
        conversation_parts = [
            "I'm working on a new authentication system",
            "We decided to use JWT tokens with refresh rotation",
            "The refresh tokens will expire after 7 days",
            "We'll store the tokens in Redis for fast validation"
        ]
        
        # Store each part
        memory_ids = []
        for part in conversation_parts:
            memory_id = await memory_garden.store_memory(
                content=part,
                user_id=user_id,
                metadata={"topic": "authentication"}
            )
            memory_ids.append(memory_id)
        
        # Query about the conversation
        result = await memory_garden.query_with_context(
            query="What authentication approach are we using?",
            user_id=user_id
        )
        
        assert result is not None
        assert len(result.context) > 0
        assert any("JWT" in ctx["content"] for ctx in result.context)
        assert any("refresh" in ctx["content"] for ctx in result.context)
    
    @pytest.mark.asyncio
    async def test_multi_user_isolation(self, memory_garden):
        """Test that users' memories are isolated"""
        user1 = "user_1"
        user2 = "user_2"
        
        # Store memories for user 1
        await memory_garden.store_memory(
            content="User 1 secret information",
            user_id=user1,
            metadata={"private": True}
        )
        
        # Store memories for user 2
        await memory_garden.store_memory(
            content="User 2 public information",
            user_id=user2,
            metadata={"private": False}
        )
        
        # Query as user 1
        result1 = await memory_garden.query_with_context(
            query="secret information",
            user_id=user1
        )
        
        # Query as user 2
        result2 = await memory_garden.query_with_context(
            query="secret information",
            user_id=user2
        )
        
        # Verify isolation
        assert len(result1.context) > 0
        assert len(result2.context) == 0
        assert "User 1 secret" in result1.context[0]["content"]
```

### 2. Persistence Test
```python
# tests/integration/test_persistence.py
import pytest
import os
import shutil
from memory_garden import MemoryGarden

class TestPersistence:
    @pytest.mark.asyncio
    async def test_restart_persistence(self, tmp_path):
        """Test data persists across system restarts"""
        db_path = tmp_path / "test_db"
        user_id = "persist_test_user"
        
        # First session - store data
        mg1 = MemoryGarden(db_path=str(db_path))
        await mg1.initialize()
        
        memory_id = await mg1.store_memory(
            content="Important data that must persist",
            user_id=user_id,
            metadata={"critical": True}
        )
        
        await mg1.shutdown()
        
        # Second session - verify data exists
        mg2 = MemoryGarden(db_path=str(db_path))
        await mg2.initialize()
        
        result = await mg2.query_with_context(
            query="Important data",
            user_id=user_id
        )
        
        assert len(result.context) > 0
        assert "Important data that must persist" in result.context[0]["content"]
        
        await mg2.shutdown()
    
    @pytest.mark.asyncio
    async def test_backup_restore(self, memory_garden, tmp_path):
        """Test backup and restore functionality"""
        backup_path = tmp_path / "backup.tar.gz"
        user_id = "backup_test_user"
        
        # Store some data
        await memory_garden.store_memory(
            content="Data to be backed up",
            user_id=user_id
        )
        
        # Create backup
        await memory_garden.create_backup(backup_path)
        assert backup_path.exists()
        
        # Clear data
        await memory_garden.clear_all_data()
        
        # Verify data is gone
        result = await memory_garden.query_with_context(
            query="backed up",
            user_id=user_id
        )
        assert len(result.context) == 0
        
        # Restore from backup
        await memory_garden.restore_from_backup(backup_path)
        
        # Verify data is restored
        result = await memory_garden.query_with_context(
            query="backed up",
            user_id=user_id
        )
        assert len(result.context) > 0
```

### 3. Performance Tests
```python
# tests/performance/test_benchmarks.py
import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from memory_garden import MemoryGarden

class TestPerformance:
    @pytest.fixture
    def performance_garden(self):
        """Memory Garden instance for performance testing"""
        mg = MemoryGarden(test_mode=True, cache_enabled=True)
        yield mg
    
    @pytest.mark.benchmark
    def test_storage_performance(self, performance_garden, benchmark):
        """Benchmark memory storage performance"""
        async def store_memory():
            await performance_garden.store_memory(
                content="Performance test content " * 50,
                user_id="perf_user",
                metadata={"test": "performance"}
            )
        
        result = benchmark(asyncio.run, store_memory())
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, performance_garden):
        """Test system under concurrent load"""
        user_id = "concurrent_user"
        num_operations = 100
        
        # Store memories concurrently
        async def store_single(index):
            return await performance_garden.store_memory(
                content=f"Concurrent memory {index}",
                user_id=user_id,
                metadata={"index": index}
            )
        
        start_time = time.time()
        
        # Execute concurrent stores
        tasks = [store_single(i) for i in range(num_operations)]
        results = await asyncio.gather(*tasks)
        
        store_time = time.time() - start_time
        
        assert len(results) == num_operations
        assert store_time < 10  # Should complete within 10 seconds
        
        # Test concurrent queries
        async def query_single():
            return await performance_garden.query_with_context(
                query="Concurrent memory",
                user_id=user_id,
                max_results=5
            )
        
        start_time = time.time()
        
        query_tasks = [query_single() for _ in range(50)]
        query_results = await asyncio.gather(*query_tasks)
        
        query_time = time.time() - start_time
        
        assert all(len(r.context) > 0 for r in query_results)
        assert query_time < 5  # Should complete within 5 seconds
    
    @pytest.mark.asyncio
    async def test_large_document_handling(self, performance_garden):
        """Test handling of large documents"""
        user_id = "large_doc_user"
        
        # Create a large document (1MB)
        large_content = "Large document content. " * 50000
        
        start_time = time.time()
        
        memory_id = await performance_garden.store_memory(
            content=large_content,
            user_id=user_id,
            metadata={"type": "large_document"}
        )
        
        store_time = time.time() - start_time
        
        assert memory_id is not None
        assert store_time < 5  # Should handle 1MB in under 5 seconds
        
        # Query the large document
        result = await performance_garden.query_with_context(
            query="Large document",
            user_id=user_id
        )
        
        assert len(result.context) > 0
```

## Load Testing

### Locust Load Test Configuration
```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between
import json

class MemoryGardenUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup before testing"""
        self.headers = {
            "X-API-Key": "load-test-key",
            "Content-Type": "application/json"
        }
        self.user_id = f"load_test_user_{self.environment.runner.user_count}"
    
    @task(3)
    def store_memory(self):
        """Store a memory"""
        payload = {
            "content": f"Load test memory content at {time.time()}",
            "user_id": self.user_id,
            "metadata": {
                "source": "load_test",
                "timestamp": time.time()
            }
        }
        
        self.client.post(
            "/memory/store",
            json=payload,
            headers=self.headers
        )
    
    @task(7)
    def query_memories(self):
        """Query for memories"""
        payload = {
            "query": "Load test memory",
            "user_id": self.user_id,
            "max_results": 5
        }
        
        self.client.post(
            "/ask",
            json=payload,
            headers=self.headers
        )
    
    @task(1)
    def health_check(self):
        """Check system health"""
        self.client.get("/health")
```

### Running Load Tests
```bash
# Run load test with 100 users
locust -f tests/performance/locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10

# Headless mode for CI/CD
locust -f tests/performance/locustfile.py --host=http://localhost:8000 --users=100 --spawn-rate=10 --headless --run-time=5m
```

## Test Data Generation

### Fixture Generator
```python
# tests/fixtures/generate_test_data.py
from faker import Faker
import json
import random

def generate_test_conversations(num_conversations=100):
    """Generate realistic test conversations"""
    fake = Faker()
    topics = [
        "authentication", "database", "frontend", "deployment",
        "testing", "security", "performance", "architecture"
    ]
    
    conversations = []
    
    for i in range(num_conversations):
        conversation = {
            "id": f"conv_{i}",
            "user_id": f"user_{random.randint(1, 10)}",
            "messages": []
        }
        
        num_messages = random.randint(5, 20)
        topic = random.choice(topics)
        
        for j in range(num_messages):
            message = {
                "content": fake.paragraph(nb_sentences=random.randint(2, 5)),
                "timestamp": fake.date_time_this_year().isoformat(),
                "metadata": {
                    "topic": topic,
                    "importance": random.choice(["low", "medium", "high"]),
                    "tags": random.sample(topics, k=random.randint(1, 3))
                }
            }
            conversation["messages"].append(message)
        
        conversations.append(conversation)
    
    return conversations

# Generate and save test data
if __name__ == "__main__":
    conversations = generate_test_conversations()
    with open("tests/fixtures/sample_conversations.json", "w") as f:
        json.dump(conversations, f, indent=2)
```

## Continuous Integration

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Memory Garden Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 6379:6379
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run unit tests
      run: |
        pytest tests/unit -v --cov=memory_garden --cov-report=xml
    
    - name: Run integration tests
      run: |
        pytest tests/integration -v
    
    - name: Run performance benchmarks
      run: |
        pytest tests/performance -v --benchmark-only
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## Testing Checklist

### Before Each Release
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Performance benchmarks meet targets
- [ ] Load tests show no degradation
- [ ] Memory leak tests pass
- [ ] Security tests pass
- [ ] Documentation is updated

### Performance Targets
| Metric | Target | Critical |
|--------|--------|----------|
| Memory storage | < 500ms | < 1s |
| Query response | < 200ms | < 500ms |
| Embedding generation | < 100ms | < 200ms |
| Concurrent users | 100 | 50 |
| Memory usage | < 2GB | < 4GB |

---
**CHECKPOINT**: Testing suite ready when:
1. All component tests pass independently
2. Integration tests verify full flow
3. Performance meets targets under load
4. CI/CD pipeline runs automatically
5. Test coverage exceeds 80%