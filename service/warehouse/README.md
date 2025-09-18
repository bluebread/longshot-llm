# Warehouse Service

Centralized storage service for trajectories and trained models using MongoDB and GridFS.

## Overview

The Warehouse service provides persistent storage and retrieval for:
- **Trajectories**: Boolean formula construction sequences with avgQ metrics
- **Trained Models**: Serialized neural network models from the trainer service
- **Metadata**: Associated information for trajectories and models

The service uses MongoDB for structured data and GridFS for large model files, providing a RESTful API built with FastAPI.

## Features

- **Trajectory Management**: CRUD operations for formula trajectories
- **Model Storage**: Upload, download, and version management for trained models
- **Batch Operations**: Efficient dataset retrieval and bulk operations
- **Filtering & Queries**: Filter by num_vars, width, timestamps, and tags
- **GridFS Integration**: Efficient storage for large model files
- **Data Purging**: Clean-up operations for both trajectories and models
- **Automatic Indexing**: Optimized queries with MongoDB indexes

## Prerequisites

### MongoDB Setup

The service requires MongoDB. Use the provided Docker Compose configuration:

```bash
cd service
docker-compose up -d
```

This starts:
- MongoDB on port 27017
- Mongo Express web UI on port 8081 (optional)

### Connection Configuration

Default MongoDB connection string in `main.py`:
```python
mongo_client = MongoClient("mongodb://haowei:bread861122@mongo-bread:27017")
```

Update this for your environment if needed.

## Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- fastapi>=0.104.1
- uvicorn>=0.24.0
- pydantic>=2.5.0
- pymongo>=4.0.0

Note: The longshot library must also be installed (see library README).

## Running the Service

### Development Mode

```bash
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Production Mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once running, interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Trajectory Management

#### GET /trajectory
Retrieve a single trajectory by ID.

**Parameters:**
- `traj_id` (query, required): Trajectory UUID

**Response:** Trajectory with steps and metadata

#### POST /trajectory
Create a new trajectory.

**Request Body:**
```json
{
  "num_vars": 4,
  "width": 3,
  "steps": [
    [token_type, token_literals, cur_avgQ],
    ...
  ]
}
```

**Response:** Created trajectory ID

#### PUT /trajectory
Update an existing trajectory's steps.

**Request Body:**
```json
{
  "traj_id": "uuid-string",
  "steps": [...]
}
```

#### DELETE /trajectory
Delete a trajectory by ID.

**Parameters:**
- `traj_id` (query, required): Trajectory UUID

#### GET /trajectory/dataset
Retrieve filtered trajectory dataset.

**Parameters:**
- `num_vars` (query, optional): Filter by number of variables
- `width` (query, optional): Filter by width
- `since` (query, optional): Start timestamp (ISO 8601)
- `until` (query, optional): End timestamp (ISO 8601)

**Response:** Array of trajectories matching filters

#### DELETE /trajectory/purge
Purge all trajectories from the database.

**Response:** Deletion count and status

### Model Storage (Parameter Server)

#### GET /models
List models matching criteria.

**Parameters:**
- `num_vars` (query, required): Number of variables
- `width` (query, required): Width parameter
- `tags` (query, optional): Filter by tags (must have all)

**Response:** List of model metadata with download URLs

#### GET /models/latest
Get the most recent model for given parameters.

**Parameters:**
- `num_vars` (query, required): Number of variables
- `width` (query, required): Width parameter

**Response:** Model metadata with download URL

#### GET /models/download/{model_id}
Download a model file (ZIP archive).

**Parameters:**
- `model_id` (path, required): Model ObjectId

**Response:** ZIP file stream

#### POST /models/upload
Upload a new model as ZIP archive.

**Form Data:**
- `file`: ZIP archive containing model
- `num_vars`: Number of variables
- `width`: Width parameter
- `tags`: Comma-separated tags (optional)

**Response:** Upload confirmation with model ID

#### DELETE /models/purge
Purge all models from GridFS storage.

**Response:** Deletion count and freed space

### Health Check

#### GET /health
Service health status.

**Response:**
```json
{
  "status": "healthy"
}
```

## Data Models

### Trajectory Structure

Each trajectory contains:
- `_id`: Unique identifier (UUID)
- `timestamp`: Creation timestamp
- `num_vars`: Number of boolean variables
- `width`: Maximum formula width
- `steps`: Array of trajectory steps

Step format: `[token_type, token_literals, cur_avgQ]`
- `token_type`: Type of gate operation
- `token_literals`: Encoded literals/variables
- `cur_avgQ`: Current average Q value

### Model Metadata

Stored models include:
- `model_id`: GridFS ObjectId
- `filename`: Original filename
- `num_vars`: Model configuration
- `width`: Model configuration
- `tags`: User-defined tags
- `upload_date`: Upload timestamp
- `size`: File size in bytes
- `download_url`: Direct download endpoint

## Usage Examples

### Python Client Example

```python
import httpx
from datetime import datetime

# Create client
client = httpx.Client(base_url="http://localhost:8000")

# Create trajectory
trajectory = {
    "num_vars": 4,
    "width": 3,
    "steps": [
        [1, 5, 0.25],
        [2, 12, 0.45],
        [1, 7, 0.67]
    ]
}
response = client.post("/trajectory", json=trajectory)
traj_id = response.json()["traj_id"]

# Retrieve trajectory
trajectory = client.get(f"/trajectory?traj_id={traj_id}").json()

# Get dataset with filters
params = {
    "num_vars": 4,
    "width": 3,
    "since": "2024-01-01T00:00:00"
}
dataset = client.get("/trajectory/dataset", params=params).json()

# Upload model
with open("model.zip", "rb") as f:
    files = {"file": f}
    data = {"num_vars": 4, "width": 3, "tags": "v1,production"}
    response = client.post("/models/upload", files=files, data=data)

# Download latest model
model_meta = client.get("/models/latest", params={"num_vars": 4, "width": 3}).json()
model_data = client.get(model_meta["download_url"]).content
```

### Using the Longshot Library Client

```python
from longshot.service import WarehouseClient

with WarehouseClient("localhost", 8000) as client:
    # Create trajectory
    traj_id = client.create_trajectory(
        steps=[(1, 5, 0.25), (2, 12, 0.45)],
        num_vars=4,
        width=3
    )

    # Get dataset
    dataset = client.get_trajectory_dataset(
        num_vars=4,
        width=3,
        since=datetime(2024, 1, 1)
    )
```

## Database Schema

### MongoDB Collections

1. **TrajectoryTable**: Stores trajectory documents
   - Indexes: timestamp, compound (num_vars, width, timestamp)

2. **fs.files**: GridFS file metadata
   - Indexes: upload_date, compound (num_vars, width, upload_date)

3. **fs.chunks**: GridFS file chunks (automatic)

## Performance Considerations

1. **Indexing**: Automatic index creation on startup for common queries
2. **Batch Operations**: Use dataset endpoint for bulk retrieval
3. **GridFS**: Efficient for large model files (>16MB)
4. **Connection Pooling**: MongoDB client handles connection pooling
5. **Streaming**: Model downloads use streaming for memory efficiency

## Monitoring

Service logs are written to `warehouse.log` with timestamps and log levels.

Monitor key metrics:
- Database connection status
- Collection sizes
- Query performance
- Storage usage

## Troubleshooting

### Common Issues

1. **MongoDB Connection Failed**
   - Check MongoDB is running: `docker-compose ps`
   - Verify connection string and credentials
   - Check network connectivity

2. **Model Upload Failed**
   - Ensure file is valid ZIP archive
   - Check file size limits
   - Verify num_vars and width parameters

3. **Query Performance**
   - Check indexes are created
   - Use filtering parameters to limit results
   - Consider pagination for large datasets

4. **Storage Issues**
   - Monitor MongoDB disk usage
   - Use purge endpoints to clean old data
   - Consider data retention policies

## Development

### Running Tests

```bash
pytest test/service/test_warehouse.py -v
```

### Extending the Service

1. **New Endpoints**: Add to `main.py` following FastAPI patterns
2. **Data Models**: Update in `longshot.service.api_models`
3. **Indexes**: Add in lifespan function for startup creation
4. **Validation**: Use Pydantic models for request/response validation

## Docker Deployment

Build and run with Docker:

```bash
docker build -t warehouse-service .
docker run -p 8000:8000 --network longshot-net warehouse-service
```

## Configuration

Environment variables (optional):
- `MONGODB_URI`: Override default MongoDB connection
- `WAREHOUSE_PORT`: Change service port (default: 8000)
- `LOG_LEVEL`: Set logging level (INFO, DEBUG, WARNING)

## API Rate Limiting

Currently no rate limiting. For production, consider:
- Using FastAPI middleware for rate limiting
- Implementing API keys for authentication
- Adding request quotas per client

## Backup and Recovery

Regular MongoDB backups recommended:
```bash
mongodump --uri="mongodb://..." --out=backup/
mongorestore --uri="mongodb://..." backup/
```

## Related Services

- **Clusterbomb Service**: Generates trajectories
- **Trainer Service**: Consumes trajectories, produces models
- **Parameter Server Integration**: Model storage and retrieval