# Memory Management Guide

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
        collection.delete(ids