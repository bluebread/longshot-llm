# Clusterbomb Service

A microservice for weapon rollout functionality in the gym-longshot system.

## Overview

The Clusterbomb service provides a single endpoint for executing weapon rollout operations. It is designed to integrate with the broader longshot ecosystem and follows the same architectural patterns as other services in the system.

## API Endpoints

### POST /weapon/rollout

Execute a weapon rollout operation.

**Request Body:**
```json
{
  "target": "string",
  "payload": {},
  "config": {}
}
```

**Response:**
```json
{
  "success": true,
  "rollout_id": "rollout_12345_67890",
  "message": "Weapon rollout completed successfully",
  "results": {}
}
```

### GET /health

Health check endpoint for service monitoring.

**Response:**
```json
{
  "status": "healthy",
  "service": "clusterbomb"
}
```

## Development

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
```

The service will be available at `http://localhost:8060`.

### Docker

```bash
# Build the image
docker build -t clusterbomb .

# Run the container
docker run -p 8060:8060 clusterbomb
```

## Configuration

The service uses port 8060 by default and follows the same logging and error handling patterns as other services in the longshot ecosystem.

## Integration

This service is designed to be integrated into the docker-compose infrastructure along with other longshot services (warehouse, ranker, trajproc, etc.).