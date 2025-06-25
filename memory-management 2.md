    return {"removed": len(ids_to_remove)}

## Performance Optimization

### Batch Processing
```python
from concurrent.futures import ThreadPoolExecutor
import asyncio

class BatchProcessor:
    def __init__(self, memory_store: ChromaMemoryStore):
        self.store = memory_store
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def batch_store_conversations(
        self,
        conversations: List[Dict],
        user_id: str,
        batch_size: int = 100
    ):
        """
        Store multiple conversations efficiently
        """
        all_chunks = []
        
        # Process conversations in parallel
        chunker = MemoryChunker()
        
        loop = asyncio.get_event_loop()
        chunking_tasks = []
        
        for conv in conversations:
            task = loop.run_in_executor(
                self.executor,
                chunker.chunk_text,
                conv["content"],
                conv.get("metadata", {})
            )
            chunking_tasks.append(task)
        
        # Wait for all chunking to complete
        chunked_results = await asyncio.gather(*chunking_tasks)
        
        # Flatten results
        for chunks in chunked_results:
            all_chunks.extend(chunks)
        
        # Store in batches
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            self.store.store_memories(
                "memory_store",
                batch,
                user_id
            )
        
        return {"total_chunks": len(all_chunks)}
```

### Caching Strategy
```python
from functools import lru_cache
import redis
import pickle

class MemoryCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_client = redis.from_url(redis_url)
        self.ttl = 3600  # 1 hour cache
    
    def cache_key(self, query: str, user_id: str) -> str:
        """Generate cache key"""
        return f"memory:{user_id}:{hashlib.md5(query.encode()).hexdigest()}"
    
    def get_cached_results(self, query: str, user_id: str) -> Optional[List[Dict]]:
        """Get cached query results"""
        key = self.cache_key(query, user_id)
        cached = self.redis_client.get(key)
        
        if cached:
            return pickle.loads(cached)
        return None
    
    def cache_results(self, query: str, user_id: str, results: List[Dict]):
        """Cache query results"""
        key = self.cache_key(query, user_id)
        self.redis_client.setex(
            key,
            self.ttl,
            pickle.dumps(results)
        )
    
    @lru_cache(maxsize=1000)
    def get_embedding_cached(self, text: str) -> np.ndarray:
        """LRU cache for embeddings"""
        return self.embedding_manager.generate_embeddings(text)
```

## Monitoring and Analytics

### Memory Usage Analytics
```python
class MemoryAnalytics:
    def __init__(self, memory_store: ChromaMemoryStore):
        self.store = memory_store
    
    def get_usage_stats(self, user_id: str) -> Dict:
        """Get detailed usage statistics"""
        collection = self.store.client.get_collection("memory_store")
        
        # Get all user memories
        user_memories = collection.get(
            where={"user_id": user_id},
            include=["metadatas", "documents"]
        )
        
        if not user_memories["ids"]:
            return {"total_memories": 0}
        
        # Calculate statistics
        stats = {
            "total_memories": len(user_memories["ids"]),
            "total_characters": sum(len(doc) for doc in user_memories["documents"]),
            "average_length": np.mean([len(doc) for doc in user_memories["documents"]]),
            "tags": self._analyze_tags(user_memories["metadatas"]),
            "temporal_distribution": self._analyze_temporal(user_memories["metadatas"]),
            "importance_distribution": self._analyze_importance(user_memories["metadatas"])
        }
        
        return stats
    
    def _analyze_tags(self, metadatas: List[Dict]) -> Dict:
        """Analyze tag distribution"""
        tag_counts = {}
        
        for metadata in metadatas:
            tags = metadata.get("tags", [])
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        return dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20])
    
    def _analyze_temporal(self, metadatas: List[Dict]) -> Dict:
        """Analyze temporal distribution"""
        temporal = {
            "by_hour": {},
            "by_day": {},
            "by_month": {}
        }
        
        for metadata in metadatas:
            stored_at = metadata.get("stored_at", "")
            if stored_at:
                dt = datetime.fromisoformat(stored_at)
                
                hour = dt.hour
                day = dt.strftime("%A")
                month = dt.strftime("%Y-%m")
                
                temporal["by_hour"][hour] = temporal["by_hour"].get(hour, 0) + 1
                temporal["by_day"][day] = temporal["by_day"].get(day, 0) + 1
                temporal["by_month"][month] = temporal["by_month"].get(month, 0) + 1
        
        return temporal
```

## Best Practices Summary

### Do's and Don'ts

| DO | DON'T |
|----|-------|
| ✅ Chunk text semantically | ❌ Use fixed-size chunks blindly |
| ✅ Include metadata for filtering | ❌ Store everything in one collection |
| ✅ Normalize embeddings | ❌ Ignore embedding model dimensions |
| ✅ Batch operations when possible | ❌ Process one document at a time |
| ✅ Implement caching for queries | ❌ Query without user_id filter |
| ✅ Archive old memories | ❌ Let collection grow unbounded |
| ✅ Monitor query performance | ❌ Ignore slow queries |
| ✅ Deduplicate similar content | ❌ Store exact duplicates |

### Optimization Checklist

1. **Embedding Model Selection**
   - Use smaller models (384 dims) for speed
   - Use larger models (768 dims) for accuracy
   - Test retrieval quality with your data

2. **Chunk Size Tuning**
   - Start with 1000 tokens
   - Adjust based on content type
   - Measure retrieval accuracy

3. **Query Performance**
   - Cache frequent queries
   - Pre-compute common embeddings
   - Use metadata filters before vector search

4. **Storage Efficiency**
   - Compress old embeddings
   - Archive inactive memories
   - Regular deduplication runs

5. **Monitoring Metrics**
   - Query response time < 200ms
   - Embedding generation < 100ms
   - Memory storage < 500ms
   - Cache hit rate > 60%

## Common Issues and Solutions

### Issue: Slow Query Performance
```python
# Solution: Implement query optimization
def optimize_query_performance(collection_name: str):
    # 1. Add metadata indexes
    collection = store.client.get_collection(collection_name)
    
    # 2. Implement query caching
    cache = MemoryCache()
    
    # 3. Use metadata pre-filtering
    def optimized_query(query: str, user_id: str, filters: Dict = None):
        # Check cache first
        cached = cache.get_cached_results(query, user_id)
        if cached:
            return cached
        
        # Pre-filter by metadata
        where_clause = {"user_id": user_id}
        if filters:
            where_clause.update(filters)
        
        # Then do vector search
        results = collection.query(
            query_texts=[query],
            n_results=5,
            where=where_clause
        )
        
        # Cache results
        cache.cache_results(query, user_id, results)
        
        return results
```

### Issue: Memory Bloat
```python
# Solution: Implement progressive summarization
def progressive_summarization(user_id: str, days: int = 30):
    """
    Summarize old memories to reduce storage
    """
    cutoff = datetime.utcnow() - timedelta(days=days)
    
    # Get old memories
    old_memories = collection.get(
        where={
            "user_id": user_id,
            "stored_at": {"$lt": cutoff.isoformat()}
        }
    )
    
    # Group by topic/tags
    grouped = group_by_similarity(old_memories)
    
    # Summarize each group
    summaries = []
    for group in grouped:
        summary = summarize_group(group)
        summaries.append({
            "content": summary,
            "metadata": {
                "type": "summary",
                "original_count": len(group),
                "date_range": get_date_range(group)
            }
        })
    
    # Store summaries and archive originals
    store_summaries(summaries)
    archive_originals(group)
```

### Issue: Poor Retrieval Quality
```python
# Solution: Implement hybrid search
def hybrid_search(
    query: str,
    user_id: str,
    alpha: float = 0.7  # Weight for semantic search
) -> List[Dict]:
    """
    Combine semantic and keyword search
    """
    # Semantic search
    semantic_results = semantic_search(query, user_id)
    
    # Keyword search
    keyword_results = keyword_search(query, user_id)
    
    # Combine results
    combined = {}
    
    # Add semantic results
    for i, result in enumerate(semantic_results):
        doc_id = result["id"]
        combined[doc_id] = {
            "content": result["content"],
            "metadata": result["metadata"],
            "semantic_score": 1 - (i / len(semantic_results)),
            "keyword_score": 0
        }
    
    # Add keyword scores
    for i, result in enumerate(keyword_results):
        doc_id = result["id"]
        if doc_id in combined:
            combined[doc_id]["keyword_score"] = 1 - (i / len(keyword_results))
        else:
            combined[doc_id] = {
                "content": result["content"],
                "metadata": result["metadata"],
                "semantic_score": 0,
                "keyword_score": 1 - (i / len(keyword_results))
            }
    
    # Calculate final scores
    for doc_id, doc in combined.items():
        doc["final_score"] = (
            alpha * doc["semantic_score"] + 
            (1 - alpha) * doc["keyword_score"]
        )
    
    # Sort by final score
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x["final_score"],
        reverse=True
    )
    
    return sorted_results[:5]
```

---
**CHECKPOINT**: Memory management system ready when:
1. Chunking preserves semantic meaning
2. Embeddings are efficiently generated and cached
3. Queries return relevant results < 200ms
4. Storage grows sustainably with archival
5. Monitoring shows healthy metrics# Memory Management Guide

## Overview
This guide details how memO and ChromaDB work together to store, index, and retrieve context efficiently in the Memory Garden system.

## Memory Storage Architecture

### Storage Layers
```
┌─────────────────────┐
│   User Input        │
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Text Chunking     │ ← Smart text segmentation
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   Embedding Gen     │ ← Vector representation
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│   ChromaDB Store    │ ← Persistent storage
└─────────────────────┘
```

## Text Chunking Strategy

### Chunking Implementation
```python
from typing import List, Dict
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MemoryChunker:
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: List[str] = None
    ):
        self.separators = separators or [
            "\n\n",    # Paragraphs
            "\n",      # Lines
            ". ",      # Sentences
            ", ",      # Clauses
            " ",       # Words
            ""         # Characters
        ]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=self.separators,
            length_function=len
        )
    
    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into semantic chunks with metadata preservation
        """
        # Pre-process text to identify special sections
        code_blocks = self._extract_code_blocks(text)
        
        # Split the main text
        chunks = self.splitter.split_text(text)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            chunk_obj = {
                "content": chunk,
                "metadata": {
                    **(metadata or {}),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "has_code": any(code in chunk for code in code_blocks),
                    "chunk_type": self._identify_chunk_type(chunk)
                }
            }
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks to handle them specially"""
        code_pattern = r'```[\s\S]*?```'
        return re.findall(code_pattern, text)
    
    def _identify_chunk_type(self, chunk: str) -> str:
        """Identify the type of content in the chunk"""
        if "```" in chunk:
            return "code"
        elif chunk.strip().startswith(("#", "##", "###")):
            return "heading"
        elif len(chunk.split()) < 50:
            return "short"
        else:
            return "paragraph"
```

### Optimal Chunk Sizes

| Content Type | Recommended Size | Overlap | Rationale |
|--------------|------------------|---------|-----------|
| **Conversations** | 500-1000 tokens | 100-200 | Maintains context flow |
| **Technical Docs** | 1000-1500 tokens | 200-300 | Preserves code blocks |
| **General Text** | 800-1200 tokens | 150-250 | Balance of context |
| **Code Snippets** | Full functions | 0 | Keep code intact |

## Embedding Generation

### Embedding Models
```python
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        Models to consider:
        - all-MiniLM-L6-v2: Fast, good quality (384 dims)
        - all-mpnet-base-v2: Higher quality (768 dims)
        - all-MiniLM-L12-v2: Balanced (384 dims)
        """
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def generate_embeddings(
        self, 
        texts: Union[str, List[str]], 
        batch_size: int = 32
    ) -> np.ndarray:
        """Generate embeddings for text(s)"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings in batches for efficiency
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,
            normalize_embeddings=True  # L2 normalization
        )
        
        return embeddings
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query"""
        # Prepend "query: " for better retrieval (some models benefit)
        query_text = f"query: {query}"
        return self.generate_embeddings(query_text)[0]
```

## ChromaDB Indexing

### Collection Management
```python
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid

class ChromaMemoryStore:
    def __init__(self, persist_path: str):
        self.client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        self.embedding_manager = EmbeddingManager()
    
    def create_collection(
        self, 
        name: str, 
        metadata: Dict = None
    ) -> chromadb.Collection:
        """Create or get a collection"""
        return self.client.get_or_create_collection(
            name=name,
            metadata=metadata or {},
            embedding_function=None  # We'll provide embeddings
        )
    
    def store_memories(
        self,
        collection_name: str,
        chunks: List[Dict],
        user_id: str
    ):
        """Store memory chunks in ChromaDB"""
        collection = self.create_collection(collection_name)
        
        # Prepare data for batch insert
        documents = []
        embeddings = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            # Generate unique ID
            chunk_id = f"{user_id}_{uuid.uuid4().hex[:8]}"
            
            # Prepare document
            documents.append(chunk["content"])
            
            # Generate embedding
            embedding = self.embedding_manager.generate_embeddings(
                chunk["content"]
            )
            embeddings.append(embedding.tolist())
            
            # Prepare metadata
            metadata = {
                **chunk.get("metadata", {}),
                "user_id": user_id,
                "stored_at": datetime.utcnow().isoformat()
            }
            metadatas.append(metadata)
            ids.append(chunk_id)
        
        # Batch insert
        collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids
```

### Indexing Strategies

#### 1. HNSW Index Configuration
```python
# ChromaDB uses HNSW (Hierarchical Navigable Small World) by default
# Optimal settings for different use cases:

# For small collections (<10k documents)
index_settings = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 100,
    "hnsw:search_ef": 50,
    "hnsw:M": 16
}

# For large collections (>100k documents)
index_settings = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 200,
    "hnsw:search_ef": 100,
    "hnsw:M": 32
}
```

#### 2. Metadata Indexing
```python
def optimize_metadata_for_filtering(metadata: Dict) -> Dict:
    """Optimize metadata for efficient filtering"""
    optimized = {}
    
    # Convert dates to timestamps for range queries
    for key, value in metadata.items():
        if isinstance(value, datetime):
            optimized[f"{key}_timestamp"] = value.timestamp()
            optimized[f"{key}_date"] = value.date().isoformat()
        elif isinstance(value, list):
            # Lists are good for "in" queries
            optimized[key] = value
        elif isinstance(value, str) and len(value) > 100:
            # Long strings: store hash for exact match
            optimized[f"{key}_hash"] = hashlib.md5(value.encode()).hexdigest()
            optimized[key] = value[:100]  # Truncate for display
        else:
            optimized[key] = value
    
    return optimized
```

## Query Optimization

### Retrieval Strategies
```python
class MemoryRetriever:
    def __init__(self, memory_store: ChromaMemoryStore):
        self.store = memory_store
        self.embedding_manager = EmbeddingManager()
    
    def retrieve(
        self,
        query: str,
        collection_name: str,
        user_id: str,
        k: int = 5,
        filters: Dict = None,
        rerank: bool = True
    ) -> List[Dict]:
        """
        Retrieve relevant memories with optional reranking
        """
        collection = self.store.client.get_collection(collection_name)
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_query_embedding(query)
        
        # Build filter
        where_clause = {"user_id": user_id}
        if filters:
            where_clause.update(filters)
        
        # Initial retrieval (get more than k for reranking)
        n_retrieve = k * 3 if rerank else k
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_retrieve,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["documents"][0]:
            return []
        
        # Format results
        memories = []
        for i, doc in enumerate(results["documents"][0]):
            memory = {
                "content": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "relevance_score": 1 - results["distances"][0][i]
            }
            memories.append(memory)
        
        # Rerank if requested
        if rerank:
            memories = self._rerank_results(query, memories, k)
        
        return memories[:k]
    
    def _rerank_results(
        self, 
        query: str, 
        memories: List[Dict], 
        k: int
    ) -> List[Dict]:
        """
        Rerank results using cross-encoder or other methods
        """
        # Simple reranking based on multiple factors
        for memory in memories:
            # Factor 1: Semantic similarity (already have)
            semantic_score = memory["relevance_score"]
            
            # Factor 2: Recency bias
            stored_at = datetime.fromisoformat(
                memory["metadata"].get("stored_at", "2000-01-01")
            )
            recency_days = (datetime.utcnow() - stored_at).days
            recency_score = 1 / (1 + recency_days / 30)  # Decay over 30 days
            
            # Factor 3: Importance
            importance_map = {"low": 0.5, "medium": 1.0, "high": 1.5, "critical": 2.0}
            importance = memory["metadata"].get("importance", "medium")
            importance_score = importance_map.get(importance, 1.0)
            
            # Combined score
            memory["final_score"] = (
                semantic_score * 0.7 +
                recency_score * 0.2 +
                importance_score * 0.1
            )
        
        # Sort by final score
        memories.sort(key=lambda x: x["final_score"], reverse=True)
        
        return memories
```

### Query Expansion
```python
def expand_query(query: str) -> List[str]:
    """
    Expand query to improve retrieval
    """
    expanded_queries = [query]
    
    # Add question variations
    if not query.endswith("?"):
        expanded_queries.append(f"{query}?")
    
    # Add context prefix
    expanded_queries.append(f"Information about {query}")
    expanded_queries.append(f"Details regarding {query}")
    
    return expanded_queries

def multi_query_retrieve(
    queries: List[str],
    collection: chromadb.Collection,
    k: int = 5
) -> List[Dict]:
    """
    Retrieve using multiple query variations
    """
    all_results = []
    seen_ids = set()
    
    for query in queries:
        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        
        for i, id in enumerate(results["ids"][0]):
            if id not in seen_ids:
                seen_ids.add(id)
                all_results.append({
                    "id": id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                })
    
    # Sort by distance and return top k
    all_results.sort(key=lambda x: x["distance"])
    return all_results[:k]
```

## Memory Cleanup and Archiving

### Cleanup Strategy
```python
import asyncio
from datetime import datetime, timedelta

class MemoryMaintenance:
    def __init__(self, memory_store: ChromaMemoryStore):
        self.store = memory_store
    
    async def cleanup_old_memories(
        self,
        collection_name: str,
        days_to_keep: int = 90,
        archive: bool = True
    ):
        """
        Clean up memories older than specified days
        """
        collection = self.store.client.get_collection(collection_name)
        cutoff_date = datetime.utcnow() - timedelta(days=days_to_keep)
        
        # Get all memories
        all_memories = collection.get(
            include=["metadatas", "documents", "embeddings"]
        )
        
        to_delete = []
        to_archive = []
        
        for i, metadata in enumerate(all_memories["metadatas"]):
            stored_at = datetime.fromisoformat(
                metadata.get("stored_at", "2000-01-01")
            )
            
            if stored_at < cutoff_date:
                to_delete.append(all_memories["ids"][i])
                
                if archive:
                    to_archive.append({
                        "id": all_memories["ids"][i],
                        "document": all_memories["documents"][i],
                        "metadata": metadata,
                        "embedding": all_memories["embeddings"][i]
                    })
        
        # Archive if requested
        if archive and to_archive:
            await self._archive_memories(to_archive)
        
        # Delete old memories
        if to_delete:
            collection.delete(ids=to_delete)
            
        return {
            "deleted": len(to_delete),
            "archived": len(to_archive) if archive else 0
        }
    
    async def _archive_memories(self, memories: List[Dict]):
        """Archive memories to cold storage"""
        archive_path = "/home/memgarden/memory-garden/data/archive"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        import json
        import gzip
        
        archive_file = f"{archive_path}/memories_{timestamp}.json.gz"
        
        with gzip.open(archive_file, 'wt', encoding='utf-8') as f:
            json.dump(memories, f, indent=2)
    
    def compact_collection(self, collection_name: str):
        """
        Compact collection for better performance
        """
        collection = self.store.client.get_collection(collection_name)
        
        # ChromaDB handles compaction internally
        # Force persistence to optimize storage
        self.store.client.persist()
        
        # Get collection stats
        count = collection.count()
        
        return {
            "collection": collection_name,
            "document_count": count,
            "status": "compacted"
        }
```

### Deduplication
```python
def deduplicate_memories(
    collection: chromadb.Collection,
    similarity_threshold: float = 0.95
):
    """
    Remove duplicate or near-duplicate memories
    """
    all_docs = collection.get(
        include=["documents", "embeddings", "metadatas"]
    )
    
    if not all_docs["ids"]:
        return {"removed": 0}
    
    embeddings = np.array(all_docs["embeddings"])
    
    # Compute similarity matrix
    similarities = np.dot(embeddings, embeddings.T)
    
    to_remove = set()
    
    for i in range(len(embeddings)):
        if i in to_remove:
            continue
            
        for j in range(i + 1, len(embeddings)):
            if j in to_remove:
                continue
                
            if similarities[i][j] > similarity_threshold:
                # Keep the more recent one
                date_i = all_docs["metadatas"][i].get("stored_at", "")
                date_j = all_docs["metadatas"][j].get("stored_at", "")
                
                if date_i < date_j:
                    to_remove.add(i)
                else:
                    to_remove.add(j)
    
    # Remove duplicates
    ids_to_remove = [all_docs["ids"][i] for i in to_remove]
    if ids_to_remove:
        collection.delete(ids=ids_to_remove)
    
    return {"removed": len(ids_to_remove)}