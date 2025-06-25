# Memory Garden File Structure

## Complete Directory Layout

```
/home/memgarden/memory-garden/
├── api/                          # FastAPI application
│   ├── main.py                   # Main FastAPI application
│   ├── requirements.txt          # Python dependencies
│   ├── routers/                  # API route modules
│   │   ├── __init__.py
│   │   ├── memory.py            # Memory-specific endpoints
│   │   └── query.py             # Query handling endpoints
│   └── utils/                    # Utility functions
│       ├── __init__.py
│       ├── embeddings.py        # Embedding generation
│       └── validators.py        # Input validation
│
├── config/                       # Configuration files
│   ├── init_chromadb.py         # ChromaDB initialization
│   ├── memo_config.yaml         # memO configuration
│   ├── nginx.conf               # Nginx configuration backup
│   └── supervisor.conf          # Supervisor configuration backup
│
├── data/                        # Persistent data storage
│   ├── chromadb/                # Vector database files
│   │   ├── chroma.sqlite3       # Main database
│   │   └── collections/         # Collection data
│   ├── uploads/                 # User uploaded files
│   │   └── {user_id}/          # Per-user upload directories
│   └── models/                  # Downloaded ML models
│       └── sentence-transformers/
│
├── logs/                        # Application logs
│   ├── api.log                  # FastAPI logs
│   ├── memory.log               # Memory operations logs
│   ├── errors.log               # Error logs
│   └── access.log               # Nginx access logs
│
├── backups/                     # Backup storage
│   ├── daily/                   # Daily backups
│   ├── weekly/                  # Weekly backups
│   └── backup.sh               # Backup script
│
├── memo/                        # memO module
│   ├── __init__.py
│   ├── memory_manager.py        # Core memory logic
│   ├── chunker.py              # Text chunking logic
│   └── requirements.txt
│
├── openwebui/                   # OpenWebUI deployment
│   ├── docker-compose.yml       # Docker configuration
│   ├── data/                    # OpenWebUI data
│   └── uploads/                 # OpenWebUI uploads
│
├── scripts/                     # Utility scripts
│   ├── start.sh                # Start all services
│   ├── stop.sh                 # Stop all services
│   ├── health_check.sh         # System health check
│   └── migrate.sh              # Migration scripts
│
├── tests/                       # Test files
│   ├── test_api.py             # API tests
│   ├── test_memory.py          # Memory tests
│   └── test_integration.py     # Integration tests
│
├── venv/                        # Python virtual environment
├── .env                         # Environment variables
├── .gitignore                   # Git ignore file
└── README.md                    # Project documentation
```

## Key Configuration Files

### 1. Main Environment File (.env)
```bash
# Location: /home/memgarden/memory-garden/.env
# Purpose: Central configuration for all services
# Access: Read by all Python applications
# Backup: Daily with scripts/backup.sh
```

### 2. API Configuration (api/config.py)
```python
# Create this file to centralize API configuration
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    chroma_db_path: str = "/home/memgarden/memory-garden/data/chromadb"
    log_level: str = "INFO"
    max_memory_size: int = 1000
    
    class Config:
        env_file = "/home/memgarden/memory-garden/.env"

settings = Settings()
```

### 3. Logging Configuration
```python
# Location: /home/memgarden/memory-garden/api/logging_config.py
import logging
import logging.handlers
from pathlib import Path

def setup_logging():
    log_dir = Path("/home/memgarden/memory-garden/logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.handlers.RotatingFileHandler(
                log_dir / "api.log",
                maxBytes=10485760,  # 10MB
                backupCount=5
            ),
            logging.StreamHandler()
        ]
    )
    
    # Configure error logger
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "errors.log",
        maxBytes=10485760,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    logging.getLogger().addHandler(error_handler)
```

## Data Storage Patterns

### ChromaDB Storage
```
data/chromadb/
├── chroma.sqlite3              # Main SQLite database
├── collections/
│   └── memory_store/          # Default collection
│       ├── segments/          # Vector segments
│       ├── metadata.json      # Collection metadata
│       └── index/            # HNSW index files
```

### User Upload Organization
```
data/uploads/
├── {user_id}/
│   ├── documents/
│   │   ├── {timestamp}_{filename}
│   │   └── metadata.json
│   ├── images/
│   └── temp/                  # Temporary processing files
```

### Log Rotation Strategy
```
logs/
├── api.log                    # Current log
├── api.log.1                  # Previous rotation
├── api.log.2                  # Older rotation
└── archived/                  # Compressed old logs
    └── api-2024-01.tar.gz
```

## Backup Procedures

### Backup Script (/home/memgarden/memory-garden/scripts/backup.sh)
```bash
#!/bin/bash
# Daily backup script for Memory Garden

BACKUP_DIR="/home/memgarden/memory-garden/backups"
DATE=$(date +%Y%m%d)
BACKUP_PATH="$BACKUP_DIR/daily/backup-$DATE"

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Backup ChromaDB
cp -r /home/memgarden/memory-garden/data/chromadb "$BACKUP_PATH/"

# Backup configurations
cp -r /home/memgarden/memory-garden/config "$BACKUP_PATH/"

# Backup environment file
cp /home/memgarden/memory-garden/.env "$BACKUP_PATH/"

# Compress backup
cd "$BACKUP_DIR/daily"
tar -czf "backup-$DATE.tar.gz" "backup-$DATE"
rm -rf "backup-$DATE"

# Remove backups older than 7 days
find "$BACKUP_DIR/daily" -name "*.tar.gz" -mtime +7 -delete

# Weekly backup on Sundays
if [ $(date +%u) -eq 7 ]; then
    cp "$BACKUP_DIR/daily/backup-$DATE.tar.gz" "$BACKUP_DIR/weekly/"
    # Keep only 4 weekly backups
    ls -t "$BACKUP_DIR/weekly/"*.tar.gz | tail -n +5 | xargs -r rm
fi

echo "Backup completed: $DATE"
```

### Restore Procedure
```bash
#!/bin/bash
# Restore from backup

BACKUP_FILE=$1
RESTORE_DIR="/home/memgarden/memory-garden"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: ./restore.sh <backup-file.tar.gz>"
    exit 1
fi

# Stop services
sudo supervisorctl stop memory-garden-api
docker-compose -f "$RESTORE_DIR/openwebui/docker-compose.yml" down

# Extract backup
tar -xzf "$BACKUP_FILE" -C /tmp/

# Restore ChromaDB
rm -rf "$RESTORE_DIR/data/chromadb"
cp -r /tmp/backup-*/chromadb "$RESTORE_DIR/data/"

# Restore configurations
cp -r /tmp/backup-*/config/* "$RESTORE_DIR/config/"

# Restore environment
cp /tmp/backup-*/.env "$RESTORE_DIR/"

# Clean up
rm -rf /tmp/backup-*

# Restart services
sudo supervisorctl start memory-garden-api
docker-compose -f "$RESTORE_DIR/openwebui/docker-compose.yml" up -d

echo "Restore completed from: $BACKUP_FILE"
```

## Maintenance Procedures

### Daily Maintenance
```bash
# Add to crontab: crontab -e
0 2 * * * /home/memgarden/memory-garden/scripts/backup.sh
0 3 * * * /home/memgarden/memory-garden/scripts/cleanup_logs.sh
```

### Log Cleanup Script
```bash
#!/bin/bash
# cleanup_logs.sh - Archive old logs

LOG_DIR="/home/memgarden/memory-garden/logs"
ARCHIVE_DIR="$LOG_DIR/archived"

mkdir -p "$ARCHIVE_DIR"

# Compress logs older than 30 days
find "$LOG_DIR" -name "*.log.*" -mtime +30 -exec gzip {} \;

# Move compressed logs to archive
find "$LOG_DIR" -name "*.gz" -exec mv {} "$ARCHIVE_DIR/" \;

# Delete archived logs older than 90 days
find "$ARCHIVE_DIR" -name "*.gz" -mtime +90 -delete
```

### ChromaDB Maintenance
```python
# maintenance.py - ChromaDB optimization
import chromadb
from chromadb.config import Settings

def optimize_chromadb():
    client = chromadb.PersistentClient(
        path="/home/memgarden/memory-garden/data/chromadb",
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Compact database
    client.persist()
    
    # Check collection sizes
    for collection in client.list_collections():
        count = collection.count()
        print(f"Collection {collection.name}: {count} items")
        
        # If collection is too large, consider archiving old data
        if count > 100000:
            print(f"WARNING: Collection {collection.name} is large")

if __name__ == "__main__":
    optimize_chromadb()
```

## Security Considerations

### File Permissions
```bash
# Set proper permissions
chmod 700 /home/memgarden/memory-garden/data
chmod 600 /home/memgarden/memory-garden/.env
chmod 755 /home/memgarden/memory-garden/scripts/*.sh
chmod 644 /home/memgarden/memory-garden/logs/*.log
```

### Sensitive File Locations
- `.env` - Contains API keys and secrets
- `data/chromadb/` - Contains user data
- `config/` - Contains service configurations
- `backups/` - Contains full system backups

---
**CHECKPOINT**: Verify file structure by running:
```bash
tree -L 3 /home/memgarden/memory-garden/
ls -la /home/memgarden/memory-garden/.env
ls -la /home/memgarden/memory-garden/data/
```