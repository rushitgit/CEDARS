# CEDARS Data Flow Analysis

## Overview
This document provides an in-depth analysis of the data flow in the CEDARS (Clinical Event Detection and Recording System) application, specifically focusing on Redis usage, CSV file storage, and search result storage.

## Architecture Components

### 1. Redis Usage and Configuration

#### Redis Server Setup
- **Container**: Redis 7.2-alpine running on port 6379
- **Purpose**: Primary use for session management and task queuing
- **Configuration**: Located in `redis/redis.conf` with custom initialization script

#### Redis Configuration in CEDARS
```python
# From cedars/config.py
RQ = {
    "redis_url": f'redis://{config["REDIS_URL"]}:{config["REDIS_PORT"]}/0',
    "task_queue_name": "cedars",
    "ops_queue_name": "ops",
    "job_timeout": 3600,
    "operation_timeout": 7200
}
```

#### Redis Usage Patterns

**Session Management:**
- All environments (Test, Dev, Prod) use Redis for session storage
- Session configuration:
  - Session type: Redis
  - Session prefix: Environment-specific (e.g., "cedars:dev:", "cedars:prod:")
  - Session serialization: JSON format
  - Session lifetime: 60 minutes (configurable)

**Task Queuing (RQ - Redis Queue):**
- Two main queues:
  - `cedars`: General task queue
  - `ops`: Operations queue
- Worker processes:
  - `worker-task`: Handles general tasks
  - `worker-ops`: Handles operations
- Job timeouts: 1 hour for tasks, 2 hours for operations

**Connection Management:**
- Redis connection established via `Redis.from_url()`
- Connection pooling and health checks implemented
- Automatic reconnection and error handling

### 2. CSV File Upload and Storage Flow

#### Upload Process Overview
1. **File Upload Endpoint**: `/upload_data` (POST)
2. **File Validation**: Supports .csv, .xlsx, .json, .parquet, .pickle, .pkl, .xml
3. **Storage Location**: MinIO object storage
4. **Processing**: MongoDB ingestion via `EMR_to_mongodb()` function

#### Detailed Upload Flow

**Step 1: File Upload and MinIO Storage**
```python
# File uploaded to MinIO bucket
filename = f"uploaded_files/{secure_filename(file.filename)}"
minio.put_object(g.bucket_name, filename, file, size, 
                part_size=10*1024*1024, num_parallel_uploads=10)
```
- **Bucket Naming**: `cedars-{project_id}` (dynamic bucket per project)
- **File Path**: Stored under `uploaded_files/` prefix
- **Chunked Upload**: 10MB chunks with parallel uploads

**Step 2: MongoDB Processing**
```python
# EMR_to_mongodb function processes the file
EMR_to_mongodb(filename)
```
- **Chunk Processing**: Files processed in 1000-row chunks
- **Data Preparation**: Each row converted to note format
- **Bulk Operations**: Notes and patients inserted/updated in batches

**Step 3: Database Collections Created**
- **NOTES**: Individual medical notes from CSV rows
- **PATIENTS**: Patient information extracted from notes
- **NOTES_SUMMARY**: Aggregated statistics (count, date ranges)
- **INFO**: Project metadata and configuration

#### MinIO Storage Details
- **Host**: Configurable via MINIO_HOST environment variable
- **Port**: Configurable via MINIO_PORT environment variable
- **Authentication**: Access key and secret key based
- **Security**: HTTP for development, HTTPS for production
- **Bucket Management**: Automatic bucket creation per project

### 3. Search Query and Results Storage

#### Search Query Flow

**Query Input and Validation:**
1. **Endpoint**: `/upload_query` (POST)
2. **Input**: Regex query string with validation
3. **Options**: NLP processing (PINES), negation handling, duplicate filtering
4. **Storage**: MongoDB QUERY collection

**Query Storage Structure:**
```python
# From save_query function
info = {
    "query": query,
    "exclude_negated": exclude_negated,
    "hide_duplicates": hide_duplicates,
    "skip_after_event": skip_after_event,
    "tag_query": tag_query,
    "date_min": date_min,
    "date_max": date_max,
    "current": True  # Only one current query at a time
}
```

#### Search Results Storage

**Primary Storage Location: MongoDB Collections**

1. **NOTES Collection**
   - Raw medical notes with patient IDs
   - Text content and metadata
   - Indexed for efficient searching

2. **ANNOTATIONS Collection**
   - NLP-extracted annotations from notes
   - Includes negation status, sentence numbers
   - Linked to notes via note_id

3. **PATIENTS Collection**
   - Patient demographic information
   - Linked to notes via patient_id
   - Summary statistics

**Search Result Processing:**
- **Query Execution**: MongoDB aggregation pipelines
- **Result Caching**: Notes summary stored in NOTES_SUMMARY collection
- **Real-time Processing**: No persistent result storage - results computed on-demand

#### PINES Integration for NLP Processing

**PINES Server Management:**
- **Dynamic Deployment**: PINES servers spun up per project
- **API Integration**: RESTful API for NLP predictions
- **Worker Scaling**: Configurable via PINES_WORKERS environment variable
- **GPU Support**: Optional GPU acceleration for NLP tasks

**NLP Result Storage:**
- **Annotations**: Stored in ANNOTATIONS collection
- **Processing Status**: Tracked in database
- **Batch Processing**: Large datasets processed in chunks

### 4. Data Flow Summary

#### Upload Data Flow
```
User Upload → MinIO Storage → MongoDB Processing → Collections Created
     ↓              ↓              ↓                    ↓
  CSV File    uploaded_files/   EMR_to_mongodb    NOTES, PATIENTS, 
  Selection   bucket storage    chunk processing   NOTES_SUMMARY
```

#### Search Query Flow
```
User Query → Validation → MongoDB Storage → PINES Processing → Results
     ↓           ↓            ↓              ↓              ↓
  Regex Input  Pattern     QUERY Collection  NLP Analysis   On-demand
  + Options    Check       + Parameters      + Annotations  Computation
```

#### Redis Usage Flow
```
Session Management → Task Queuing → Worker Processing → Result Storage
       ↓                ↓              ↓                ↓
   User Sessions    RQ Queues      Background Jobs   MongoDB/File
   (60min TTL)     (1-2hr TTL)    (Redis + RQ)     Storage
```

### 5. Key Technical Details

#### Performance Optimizations
- **Chunked Processing**: 1000-row chunks for large files
- **Bulk Operations**: MongoDB bulk insert/update operations
- **Parallel Uploads**: MinIO parallel chunk uploads
- **Connection Pooling**: MongoDB and Redis connection management

#### Scalability Features
- **Worker Scaling**: Configurable worker processes
- **Queue Management**: Separate queues for different operation types
- **Resource Limits**: Memory and CPU constraints per container
- **Health Checks**: Automated health monitoring for all services

#### Data Persistence
- **MinIO**: Persistent object storage with configurable redundancy
- **MongoDB**: Document database with automatic indexing
- **Redis**: In-memory storage with optional persistence
- **File System**: Docker volumes for persistent storage

### 6. Environment Configuration

#### Required Environment Variables
```bash
# Database
DB_PORT=27017
DB_HOST_PORT=27018
DB_USER=admin
DB_PWD=password

# MinIO Storage
MINIO_HOST=minio
MINIO_PORT=9000
MINIO_ACCESS_KEY=root
MINIO_SECRET_KEY=rootpassword

# Redis
REDIS_PORT=6379

# Scaling
WORKER_SCALE=2
GPU_SCALE=1
PINES_WORKERS=1
```

#### Service Dependencies
- **Web Application**: Depends on Redis and MongoDB
- **Worker Processes**: Depend on Redis and web application
- **PINES Workers**: Depend on web application and Redis
- **MinIO**: Independent service for file storage

This architecture provides a robust, scalable system for clinical data processing with clear separation of concerns between file storage, data processing, and search functionality.