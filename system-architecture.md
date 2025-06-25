# Memory Garden System Architecture

## Overview
The Memory Garden is a memory-persistence system designed to maintain context across AI development sessions, preventing the "getting lost" problem during complex builds.

## Core Architecture Diagram

```
┌─────────────────┐
│   OpenWebUI     │ ← User Interface Layer
│  (Chat + Files) │
└────────┬────────┘
         │ HTTP/WebSocket
┌────────▼────────┐
│    FastAPI      │ ← Orchestration Layer
│  (ask_gateway)  │
└───┬────────┬────┘
    │        │
┌───▼──┐  ┌──▼────────┐
│ memO │  │ LLM Model │ ← Processing Layer
└───┬──┘  │ (Local/   │
    │     │  Remote)  │
┌───▼──────┐└──────────┘
│ ChromaDB │ ← Storage Layer
└──────────┘
```

## Component Relationships

### 1. OpenWebUI (Frontend)
- **Purpose**: Primary user interface for chat and file uploads
- **Technology**: Node.js/React-based web application
- **Key Features**:
  - WebSocket support for real-time chat
  - File upload and management
  - Session persistence
  - Multi-model support

### 2. FastAPI (Orchestration)
- **Purpose**: Central routing and orchestration
- **Technology**: Python 3.11+ with FastAPI framework
- **Key Endpoints**:
  - `/ask` - Main query endpoint
  - `/memory/store` - Store new memories
  - `/memory/retrieve` - Retrieve relevant context
  - `/health` - System health checks

### 3. memO (Memory Management)
- **Purpose**: Persistent memory storage and retrieval
- **Technology**: Python-based memory abstraction layer
- **Key Features**:
  - Semantic chunking of conversations
  - Metadata tagging for context
  - Time-based and relevance-based retrieval

### 4. ChromaDB (Vector Storage)
- **Purpose**: Efficient similarity search for memories
- **Technology**: Embedded vector database
- **Key Features**:
  - HNSW indexing for fast retrieval
  - Persistent storage to disk
  - Collection-based organization

## Data Flow Patterns

### Query Flow
1. User submits query via OpenWebUI
2. OpenWebUI sends request to FastAPI `/ask` endpoint
3. FastAPI queries memO for relevant context
4. memO retrieves vectors from ChromaDB
5. FastAPI combines context with query
6. LLM processes enriched query
7. Response flows back through FastAPI to OpenWebUI

### Memory Storage Flow
1. Conversation chunks identified by FastAPI
2. Chunks sent to memO for processing
3. memO generates embeddings
4. Embeddings stored in ChromaDB with metadata
5. Confirmation sent back to user

## Integration Points

### OpenWebUI ↔ FastAPI
- **Protocol**: HTTP REST + WebSocket
- **Authentication**: API key-based
- **Port**: 8000 (FastAPI), 3000 (OpenWebUI)

### FastAPI ↔ memO
- **Protocol**: Direct Python function calls
- **Threading**: Async/await pattern
- **Error Handling**: Try-except with fallback

### memO ↔ ChromaDB
- **Protocol**: ChromaDB Python client
- **Connection**: Local persistent client
- **Collections**: Separate for different memory types

## Security Considerations

### API Security
- All endpoints require authentication
- Rate limiting on public endpoints
- Input validation and sanitization

### Data Security
- ChromaDB data encrypted at rest
- Memory access controlled by user ID
- No cross-user memory leakage

### Network Security
- HTTPS for all external communications
- Firewall rules limiting port access
- VPN recommended for production

## Scalability Patterns

### Horizontal Scaling
- FastAPI can run multiple workers
- ChromaDB supports sharding
- Load balancer for multiple instances

### Vertical Scaling
- Increase ChromaDB cache size
- Add more FastAPI workers
- GPU acceleration for embeddings

## Technology Stack Rationale

### Why OpenWebUI?
- Self-hosted alternative to ChatGPT UI
- Built-in file handling
- Extensible via plugins
- Active community support

### Why FastAPI?
- High performance async support
- Automatic API documentation
- Type hints for reliability
- Easy integration with Python ML tools

### Why memO + ChromaDB?
- ChromaDB provides efficient vector search
- memO adds semantic layer on top
- Both are Python-native
- Proven in production environments

## Error Recovery

### Component Failure Scenarios
1. **ChromaDB Down**: FastAPI falls back to direct LLM queries
2. **memO Error**: Log and continue without context
3. **OpenWebUI Crash**: FastAPI remains accessible via API
4. **Network Issues**: Local queue for retry

## Monitoring Points
- FastAPI response times
- ChromaDB query performance
- Memory storage success rate
- Error log aggregation

---
**CHECKPOINT**: Before proceeding to installation, verify you understand:
1. How each component connects to others
2. The data flow for queries and memory storage
3. Security considerations for the system
4. Where errors might occur and how to handle them