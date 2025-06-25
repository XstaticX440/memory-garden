# Memory Garden API Specification

## Base Configuration

### API Base URL
```
Production: https://your-domain.com/api
Development: http://localhost:8000
```

### Authentication
All endpoints require API key authentication:
```
Headers:
  X-API-Key: your-api-key-here
```

### Content Types
```
Request: application/json
Response: application/json
```

## Core Endpoints

### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "memory-garden",
  "version": "1.0.0",
  "timestamp": "2024-01-20T10:30:00Z",
  "components": {
    "chromadb": "connected",
    "memory_service": "active",
    "storage": {
      "used_gb": 2.5,
      "total_gb": 50
    }
  }
}
```

### 2. Query with Memory Context
```http
POST /ask
```

**Request Body:**
```json
{
  "query": "What did we discuss about the authentication system?",
  "user_id": "user123",
  "include_context": true,
  "max_results": 5,
  "filters": {
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-01-20"
    },
    "tags": ["authentication", "security"]
  }
}
```

**Response:**
```json
{
  "query": "What did we discuss about the authentication system?",
  "context": [
    {
      "content": "We decided to use JWT tokens with refresh token rotation...",
      "metadata": {
        "timestamp": "2024-01-15T14:30:00Z",
        "source": "conversation",
        "tags": ["authentication", "jwt"],
        "relevance_score": 0.92
      },
      "distance": 0.08
    }
  ],
  "response": "Based on our previous discussions, you decided to implement JWT authentication with the following features...",
  "tokens_used": 450,
  "processing_time_ms": 234
}
```

### 3. Store Memory
```http
POST /memory/store
```

**Request Body:**
```json
{
  "content": "Decided to use PostgreSQL for user data and ChromaDB for vector storage",
  "user_id": "user123",
  "metadata": {
    "source": "conversation",
    "session_id": "sess_abc123",
    "tags": ["architecture", "database"],
    "importance": "high",
    "project": "memory-garden"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Memory stored successfully",
  "memory_id": "mem_xyz789",
  "timestamp": "2024-01-20T10:30:00Z"
}
```

### 4. Retrieve Memories
```http
POST /memory/retrieve
```

**Request Body:**
```json
{
  "user_id": "user123",
  "filters": {
    "tags": ["architecture"],
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-01-20"
    },
    "importance": ["high", "critical"]
  },
  "sort_by": "relevance",
  "limit": 20,
  "offset": 0
}
```

**Response:**
```json
{
  "memories": [
    {
      "id": "mem_xyz789",
      "content": "Decided to use PostgreSQL for user data...",
      "metadata": {
        "timestamp": "2024-01-20T10:30:00Z",
        "tags": ["architecture", "database"],
        "importance": "high"
      }
    }
  ],
  "total_count": 45,
  "page": 1,
  "has_more": true
}
```

### 5. Update Memory
```http
PUT /memory/{memory_id}
```

**Request Body:**
```json
{
  "metadata": {
    "tags": ["architecture", "database", "postgresql"],
    "importance": "critical",
    "reviewed": true
  }
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Memory updated successfully",
  "memory_id": "mem_xyz789"
}
```

### 6. Delete Memory
```http
DELETE /memory/{memory_id}
```

**Response:**
```json
{
  "status": "success",
  "message": "Memory deleted successfully"
}
```

### 7. Batch Operations
```http
POST /memory/batch
```

**Request Body:**
```json
{
  "operations": [
    {
      "action": "store",
      "data": {
        "content": "First memory content",
        "user_id": "user123",
        "metadata": {"tags": ["batch1"]}
      }
    },
    {
      "action": "update",
      "memory_id": "mem_abc456",
      "data": {
        "metadata": {"reviewed": true}
      }
    }
  ]
}
```

**Response:**
```json
{
  "status": "success",
  "results": [
    {
      "operation_index": 0,
      "status": "success",
      "memory_id": "mem_new123"
    },
    {
      "operation_index": 1,
      "status": "success",
      "memory_id": "mem_abc456"
    }
  ],
  "total_operations": 2,
  "successful": 2,
  "failed": 0
}
```

### 8. Search Memories
```http
POST /memory/search
```

**Request Body:**
```json
{
  "query": "database architecture decisions",
  "user_id": "user123",
  "search_type": "semantic",
  "limit": 10,
  "threshold": 0.7
}
```

**Response:**
```json
{
  "results": [
    {
      "content": "Decided to use PostgreSQL for user data...",
      "score": 0.92,
      "metadata": {
        "timestamp": "2024-01-20T10:30:00Z",
        "tags": ["architecture", "database"]
      }
    }
  ],
  "total_results": 3,
  "search_time_ms": 45
}
```

### 9. Export Memories
```http
POST /memory/export
```

**Request Body:**
```json
{
  "user_id": "user123",
  "format": "json",
  "filters": {
    "date_range": {
      "start": "2024-01-01",
      "end": "2024-01-20"
    }
  },
  "include_embeddings": false
}
```

**Response:**
```json
{
  "export_id": "exp_123456",
  "status": "processing",
  "format": "json",
  "estimated_size_mb": 25,
  "download_url": "/api/memory/export/exp_123456/download"
}
```

### 10. Analytics
```http
GET /memory/analytics/{user_id}
```

**Response:**
```json
{
  "user_id": "user123",
  "statistics": {
    "total_memories": 1523,
    "total_size_mb": 45.2,
    "tags_distribution": {
      "architecture": 234,
      "authentication": 156,
      "database": 189
    },
    "memories_by_month": {
      "2024-01": 523,
      "2023-12": 412
    },
    "average_memory_length": 256,
    "most_active_days": ["Monday", "Wednesday"]
  }
}
```

## WebSocket Endpoints

### Real-time Memory Stream
```javascript
// WebSocket connection for real-time updates
const ws = new WebSocket('ws://localhost:8000/ws/memory-stream');

ws.onopen = () => {
  ws.send(JSON.stringify({
    action: 'subscribe',
    user_id: 'user123',
    api_key: 'your-api-key'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Handle real-time memory updates
};
```

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "MEMORY_NOT_FOUND",
    "message": "The requested memory could not be found",
    "details": {
      "memory_id": "mem_invalid123",
      "user_id": "user123"
    },
    "timestamp": "2024-01-20T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

### Common Error Codes
| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_API_KEY` | 401 | Invalid or missing API key |
| `MEMORY_NOT_FOUND` | 404 | Requested memory does not exist |
| `USER_NOT_FOUND` | 404 | User ID not found |
| `INVALID_REQUEST` | 400 | Malformed request body |
| `QUOTA_EXCEEDED` | 429 | Rate limit or storage quota exceeded |
| `INTERNAL_ERROR` | 500 | Server-side error |

## Rate Limiting

### Default Limits
- **Standard tier**: 100 requests/minute
- **Premium tier**: 1000 requests/minute
- **Batch operations**: 10 requests/minute

### Rate Limit Headers
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705751400
```

## Authentication Implementation

### API Key Middleware
```python
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
import hashlib

api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    # Hash the provided key for comparison
    hashed_key = hashlib.sha256(api_key.encode()).hexdigest()
    
    # Check against stored hashed keys
    valid_keys = load_valid_api_keys()  # Load from database/config
    
    if hashed_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    return api_key

# Use in endpoints
@app.post("/memory/store")
async def store_memory(
    request: MemoryRequest,
    api_key: str = Depends(verify_api_key)
):
    # Implementation
    pass
```

## Request/Response Examples

### Full cURL Examples

**Store Memory:**
```bash
curl -X POST http://localhost:8000/memory/store \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "content": "Important decision about API design",
    "user_id": "user123",
    "metadata": {
      "tags": ["api", "design"],
      "importance": "high"
    }
  }'
```

**Query with Context:**
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "query": "What API design decisions did we make?",
    "user_id": "user123",
    "include_context": true,
    "max_results": 5
  }'
```

## SDK Usage Examples

### Python SDK
```python
from memory_garden import MemoryGardenClient

client = MemoryGardenClient(
    api_key="your-api-key",
    base_url="http://localhost:8000"
)

# Store memory
memory = client.store_memory(
    content="Important project decision",
    user_id="user123",
    tags=["project", "decision"]
)

# Query with context
response = client.ask(
    query="What project decisions have we made?",
    user_id="user123"
)
```

### JavaScript SDK
```javascript
import { MemoryGarden } from 'memory-garden-sdk';

const client = new MemoryGarden({
  apiKey: 'your-api-key',
  baseUrl: 'http://localhost:8000'
});

// Store memory
const memory = await client.storeMemory({
  content: 'Important project decision',
  userId: 'user123',
  metadata: {
    tags: ['project', 'decision']
  }
});

// Query with context
const response = await client.ask({
  query: 'What project decisions have we made?',
  userId: 'user123'
});
```

---
**CHECKPOINT**: Test API endpoints in order:
1. Health check (no auth required)
2. Store a test memory
3. Retrieve the stored memory
4. Query with the stored context
5. Verify error handling with invalid requests