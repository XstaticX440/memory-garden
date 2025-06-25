# Memory Garden Installation Guide

## Prerequisites

### VPS Requirements
- **OS**: Ubuntu 22.04 LTS (recommended)
- **RAM**: Minimum 8GB, 16GB recommended
- **Storage**: 50GB SSD minimum
- **CPU**: 4 cores minimum
- **Python**: 3.11 or higher
- **Node.js**: 18.x or higher

### Initial VPS Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3.11 python3.11-venv python3-pip
sudo apt install -y nodejs npm git curl wget
sudo apt install -y build-essential libssl-dev libffi-dev
sudo apt install -y nginx supervisor redis-server

# Create project user
sudo useradd -m -s /bin/bash memgarden
sudo usermod -aG sudo memgarden
```

## Component Installation

### 1. Directory Structure Setup

```bash
# Switch to memgarden user
sudo su - memgarden

# Create base directories
mkdir -p ~/memory-garden/{data,logs,config,backups}
mkdir -p ~/memory-garden/data/{chromadb,uploads,models}
cd ~/memory-garden

# Set permissions
chmod 750 ~/memory-garden/data
```

### 2. Python Environment Setup

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. ChromaDB Installation

```bash
# Install ChromaDB
pip install chromadb==0.4.24

# Create ChromaDB initialization script
cat > ~/memory-garden/config/init_chromadb.py << 'EOF'
import chromadb
from chromadb.config import Settings

# Initialize ChromaDB with persistent storage
client = chromadb.PersistentClient(
    path="/home/memgarden/memory-garden/data/chromadb",
    settings=Settings(
        anonymized_telemetry=False,
        allow_reset=True,
        is_persistent=True
    )
)

# Create default collection
collection = client.get_or_create_collection(
    name="memory_store",
    metadata={"description": "Main memory storage for context"}
)

print("ChromaDB initialized successfully")
print(f"Collection count: {len(client.list_collections())}")
EOF

# Run initialization
python ~/memory-garden/config/init_chromadb.py
```

### 4. memO Installation

```bash
# Clone memO repository (using example repo, adjust as needed)
cd ~/memory-garden
git clone https://github.com/your-repo/memo.git memo

# Install memO dependencies
cd memo
pip install -r requirements.txt

# Create memO configuration
cat > ~/memory-garden/config/memo_config.yaml << 'EOF'
database:
  type: chromadb
  path: /home/memgarden/memory-garden/data/chromadb
  collection: memory_store

embedding:
  model: all-MiniLM-L6-v2
  dimension: 384
  
memory:
  chunk_size: 1000
  chunk_overlap: 200
  max_memories_per_query: 10
  
api:
  host: 127.0.0.1
  port: 8001
EOF
```

### 5. FastAPI Application Setup

```bash
# Create FastAPI application
mkdir -p ~/memory-garden/api
cd ~/memory-garden/api

# Create requirements file
cat > requirements.txt << 'EOF'
fastapi==0.110.0
uvicorn==0.27.0
pydantic==2.5.0
chromadb==0.4.24
sentence-transformers==2.3.1
python-multipart==0.0.9
redis==5.0.1
httpx==0.26.0
EOF

# Install dependencies
pip install -r requirements.txt

# Create main FastAPI application
cat > main.py << 'EOF'
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
import logging
from typing import List, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Memory Garden API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(
    path="/home/memgarden/memory-garden/data/chromadb",
    settings=Settings(anonymized_telemetry=False)
)

# Models
class QueryRequest(BaseModel):
    query: str
    user_id: str
    include_context: bool = True
    max_results: int = 5

class MemoryRequest(BaseModel):
    content: str
    user_id: str
    metadata: Optional[dict] = {}

class QueryResponse(BaseModel):
    query: str
    context: List[dict]
    response: Optional[str] = None

# Endpoints
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "memory-garden"}

@app.post("/ask", response_model=QueryResponse)
async def ask_with_memory(request: QueryRequest):
    try:
        collection = chroma_client.get_collection("memory_store")
        
        # Retrieve relevant memories
        results = collection.query(
            query_texts=[request.query],
            n_results=request.max_results,
            where={"user_id": request.user_id}
        )
        
        context = []
        if results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                context.append({
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })
        
        return QueryResponse(
            query=request.query,
            context=context
        )
        
    except Exception as e:
        logger.error(f"Error in ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/store")
async def store_memory(request: MemoryRequest):
    try:
        collection = chroma_client.get_collection("memory_store")
        
        # Add user_id to metadata
        metadata = request.metadata.copy()
        metadata["user_id"] = request.user_id
        
        # Store memory
        collection.add(
            documents=[request.content],
            metadatas=[metadata],
            ids=[f"{request.user_id}_{len(request.content)}_{hash(request.content)}"]
        )
        
        return {"status": "success", "message": "Memory stored successfully"}
        
    except Exception as e:
        logger.error(f"Error storing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
```

### 6. OpenWebUI Installation

```bash
# Create OpenWebUI directory
cd ~/memory-garden
mkdir openwebui

# Download and run OpenWebUI using Docker (recommended)
# First, install Docker if not present
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker memgarden

# Create docker-compose file
cat > ~/memory-garden/openwebui/docker-compose.yml << 'EOF'
version: '3.8'

services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:main
    container_name: open-webui
    ports:
      - "3000:8080"
    environment:
      - OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1
      - WEBUI_AUTH=True
      - WEBUI_SECRET_KEY=your-secret-key-here
      - DATA_DIR=/app/backend/data
    volumes:
      - ./data:/app/backend/data
      - ./uploads:/app/backend/uploads
    restart: unless-stopped
    extra_hosts:
      - "host.docker.internal:host-gateway"
EOF

# Start OpenWebUI
cd ~/memory-garden/openwebui
docker-compose up -d
```

### 7. Nginx Configuration

```bash
# Create Nginx configuration
sudo cat > /etc/nginx/sites-available/memory-garden << 'EOF'
upstream fastapi {
    server 127.0.0.1:8000;
}

upstream openwebui {
    server 127.0.0.1:3000;
}

server {
    listen 80;
    server_name your-domain.com;
    
    # OpenWebUI frontend
    location / {
        proxy_pass http://openwebui;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    
    # FastAPI backend
    location /api/ {
        proxy_pass http://fastapi/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# Enable site
sudo ln -s /etc/nginx/sites-available/memory-garden /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 8. Supervisor Configuration

```bash
# Create supervisor configuration for FastAPI
sudo cat > /etc/supervisor/conf.d/memory-garden.conf << 'EOF'
[program:memory-garden-api]
command=/home/memgarden/memory-garden/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
directory=/home/memgarden/memory-garden/api
user=memgarden
autostart=true
autorestart=true
redirect_stderr=true
stdout_logfile=/home/memgarden/memory-garden/logs/api.log
environment=PATH="/home/memgarden/memory-garden/venv/bin",PYTHONPATH="/home/memgarden/memory-garden"
EOF

# Start services
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start memory-garden-api
```

## Environment Variables

Create `.env` file:

```bash
cat > ~/memory-garden/.env << 'EOF'
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your-secret-key-here

# ChromaDB Configuration
CHROMA_DB_PATH=/home/memgarden/memory-garden/data/chromadb
CHROMA_COLLECTION=memory_store

# OpenWebUI Configuration
WEBUI_PORT=3000
WEBUI_AUTH_ENABLED=true

# Memory Configuration
MAX_MEMORY_SIZE=1000
MEMORY_CHUNK_SIZE=500
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Logging
LOG_LEVEL=INFO
LOG_PATH=/home/memgarden/memory-garden/logs
EOF
```

## Validation Steps

```bash
# 1. Check ChromaDB
python -c "import chromadb; print('ChromaDB OK')"

# 2. Check FastAPI
curl http://localhost:8000/health

# 3. Check OpenWebUI
curl http://localhost:3000

# 4. Test memory storage
curl -X POST http://localhost:8000/memory/store \
  -H "Content-Type: application/json" \
  -d '{"content": "Test memory", "user_id": "test-user"}'

# 5. Test memory retrieval
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Test", "user_id": "test-user"}'
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Check if ports 3000, 8000 are available
   ```bash
   sudo lsof -i :3000
   sudo lsof -i :8000
   ```

2. **Permission errors**: Ensure memgarden user owns all files
   ```bash
   sudo chown -R memgarden:memgarden ~/memory-garden
   ```

3. **ChromaDB errors**: Reset database if corrupted
   ```bash
   rm -rf ~/memory-garden/data/chromadb/*
   python ~/memory-garden/config/init_chromadb.py
   ```

4. **Docker issues**: Restart Docker service
   ```bash
   sudo systemctl restart docker
   docker-compose down
   docker-compose up -d
   ```

---
**CHECKPOINT**: After installation, verify:
1. All services are running (check with `sudo supervisorctl status`)
2. Ports are accessible (3000 for UI, 8000 for API)
3. Test memory storage and retrieval work
4. Logs are being written to ~/memory-garden/logs/