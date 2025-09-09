# Microservice Documentation

This document outlines the structure and content of the API documentation for the microservices of this project in Version 2. It serves as a guide for developers to understand how to use the simplified API architecture effectively.

## Overview

1. **Warehouse**
    - Manages the storage and retrieval of data with simplified schema. 
    - Contains the following components:
        1. *Trajectory Tables*: Simplified entries with cur_avgQ field stored in MongoDB collections.
        2. *Parameter Server*: Manages model storage and retrieval using MongoDB GridFS for efficient handling of compressed model files. All models are stored as ZIP archives with associated metadata including num_vars, width, and optional tags.

2. **Weapon Services (Job Executors)**
    - Run as independent containerized services to collect trajectories using different strategies.
    - **No public API endpoints** - these services execute trajectory collection jobs autonomously.
    - Services are deployed as standalone containers that continuously collect and store trajectory data.
    - Types of weapon services:
        - **Clusterbomb**: Collects trajectories using MAP-Elites algorithm for quality-diversity optimization. Maintains an archive of diverse, high-performing trajectories to efficiently explore the solution space.
        - **Missile** (future implementation): Will collect trajectories using AI/RL models for guided exploration. Leverages learned policies for more targeted data collection.

## Database Schema

**V2 Schema Changes**: The V2 architecture eliminates linked list structures in trajectory data by embedding complete formula reconstruction information within each trajectory record.

Because MongoDB does not have a native UUID type, we use UUIDs as strings in the database.

### Trajectory Table

**V2 Schema**: Each trajectory contains the COMPLETE formula construction sequence, combining both base formula reconstruction (prefix) and new exploration steps (suffix). This eliminates the linked list structure and enables direct formula reconstruction.

Trajectories are stored in MongoDB with the following schema: 

| Column Name      | Type     | Description                                      |
|------------------|:----------:|--------------------------------------------------|
| traj_id               | UUID     | Primary key                                      |
| timestamp        | datetime | The time when the trajectory was generated       |
| num_vars         | int   | Number of variables in the trajectory               |
| width            | int      | Width of the trajectory                             |
| steps            | list[list] | **V2**: Complete trajectory steps stored as lists `[token_type, token_literals, cur_avgQ]` |
| step[0]          | int      | The type of the token, 0 for ADD, 1 for DELETE    |
| step[1]          | int      | The binary representation for the literals of the token |
| step[2]          | float    | **V2**: Current average Q-value for this specific step |

**V2 Key Changes**:
- **Complete Trajectories**: No more partial/linked trajectories - each record contains full reconstruction sequence
- **Embedded Base Formulas**: Prefix trajectory steps embedded within each trajectory record
- **Direct Reconstruction**: Formula definitions reconstructed directly from trajectory steps without database lookups
- **Consistent avgQ**: All steps include `cur_avgQ` field for trajectory analysis
- **Optimized Storage**: Steps stored as lists `[token_type, token_literals, cur_avgQ]` for 45.9% BSON size reduction
- **Tuple API Format**: API returns steps as tuples `(token_type, token_literals, cur_avgQ)` for efficiency

### Parameter Server 

Models are stored in MongoDB GridFS with the following metadata structure:

| Field Name       | Type     | Description                                      |
|------------------|:--------:|--------------------------------------------------|  
| num_vars         | int      | Number of variables in the model (required)     |
| width            | int      | Width parameter of the model (required)         |
| tags             | list[str]| User-defined tags for categorizing models (optional) |
| upload_date      | datetime | Timestamp when model was uploaded (auto-generated, indexed) |
| content_type     | string   | MIME type, always "application/zip"             |

**Storage Notes**:
- All models stored as ZIP archives for efficient storage
- Automatic indexing on `metadata.upload_date` for latest model retrieval
- Compound index on `(metadata.num_vars, metadata.width, metadata.upload_date)` for query optimization
- Users responsible for decompressing ZIP files after download

## API Endpoints

### Warehouse

The Warehouse is a microservice that manages the storage and retrieval of trajectory data. It abstracts the complexity of data management and provides a simple interface for data access.

#### Data Format

When request is not passed in a correct format, the server will return a `422 Unprocessable Entity` error.

For every token literals in the request body, it should be represented as a 64-bits integer, the first 32 bits for the positive literals and the last 32 bits for the negative literals.

#### `GET /trajectory`
Retrieve a trajectory by its ID.

- **Query Parameters:**  
    - `traj_id` (string, required): Trajectory UUID.

- **Response:**  
    ```json
    {
        "traj_id": "t456",
        "steps": [
            [0, 5, 2.3]
        ],
        "timestamp": "2025-07-21T12:00:00Z"
    }
    ```
    Note: Steps are returned as tuples `[token_type, token_literals, cur_avgQ]` for reduced data redundancy.
- **Status Codes:**  
    - `200 OK`, `404 Not Found`

#### `POST /trajectory`
Add a new trajectory. The fields `traj_id` and `timestamp` are automatically generated by the server.

- **Request Body:**  
    ```json
    {
        "steps": [
            [0, 5, 2.3]
        ]
    }
    ```
    Note: Steps must be provided as tuples `[token_type, token_literals, cur_avgQ]`.
- **Response:**  
    ```json
    {
        "traj_id": "t456"
    }
    ```
- **Status Codes:**  
    - `201 Created`, `422 Unprocessable Entity`

#### `PUT /trajectory`
Update an existing trajectory.

- **Request Body:**  
    ```json
    {
        "traj_id": "t456",
        "steps": [
            [1, 3, 2.8]
        ]
    }
    ```
    Note: Steps must be provided as tuples `[token_type, token_literals, cur_avgQ]`.
- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `422 Unprocessable Entity`, `404 Not Found`

#### `DELETE /trajectory`
Delete a trajectory.

- **Query Parameters:**  
    - `traj_id` (string, required): Trajectory UUID.

- **Response:**  
    - Success message.
- **Status Codes:**  
    - `200 OK`, `404 Not Found`

#### `DELETE /trajectory/purge`
**DESTRUCTIVE OPERATION**: Completely purge all trajectory data from MongoDB. This operation cannot be undone.

- **Request Body:** None
- **Response:**  
    ```json
    {
        "success": true,
        "deleted_count": 1247,
        "message": "Successfully purged 1247 trajectories from MongoDB",
        "timestamp": "2025-01-21T14:30:45.123456"
    }
    ```
- **Status Codes:**  
    - `200 OK`, `500 Internal Server Error`

#### `GET /trajectory/dataset`
Get the complete trajectory dataset with all trajectories using optimized tuple format for steps.

**TODO:** Add an index on the timestamp field in MongoDB for efficient range queries on trajectory timestamps.

- **Query Parameters:**
    - `num_vars` (int, optional): Filter trajectories by number of variables
    - `width` (int, optional): Filter trajectories by width
    - `since` (datetime, optional): Filter trajectories to only include those with timestamp after this date (ISO 8601 format)
    - `until` (datetime, optional): Filter trajectories to only include those with timestamp before this date (ISO 8601 format)

- **Response:**
    ```json
    {
        "trajectories": [
            {
                "traj_id": "t456",
                "timestamp": "2023-01-01T00:00:00Z",
                "num_vars": 3,
                "width": 2,
                "steps": [
                    [0, 5, 0.75],
                    [1, 3, 0.85]
                ]
            }
        ]
    }
    ```
    Note: Each step is represented as a tuple `[token_type, token_literals, cur_avgQ]` for reduced data redundancy.

- **Status Codes:**
    - `200 OK`: Successfully retrieved trajectories (may return empty list if no trajectories match filters or if `since` > `until`)
    - `400 Bad Request`: Invalid date format in `since` or `until` parameters or if `since` timestamp is after `until` timestamp


#### `GET /models`
Retrieve models matching the specified criteria. Returns metadata and download links for all matching models.

- **Query Parameters:**
    - `num_vars` (int, required): Filter by number of variables
    - `width` (int, required): Filter by width parameter
    - `tags` (list[string], optional): Filter by tags (models must contain ALL specified tags)

- **Response:**
    ```json
    {
        "models": [
            {
                "model_id": "507f1f77bcf86cd799439011",
                "filename": "model_v3_2024.zip",
                "num_vars": 4,
                "width": 3,
                "tags": ["production", "optimized"],
                "upload_date": "2024-01-15T10:30:00Z",
                "size": 1048576,
                "download_url": "/models/download/507f1f77bcf86cd799439011"
            }
        ],
        "count": 1
    }
    ```

- **Status Codes:**
    - `200 OK`: Successfully retrieved models (may return empty list if no matches)
    - `400 Bad Request`: Missing required parameters (num_vars or width)

#### `GET /models/latest`
Retrieve the most recently uploaded model for the specified num_vars and width combination.

- **Query Parameters:**
    - `num_vars` (int, required): Number of variables
    - `width` (int, required): Width parameter

- **Response:**
    ```json
    {
        "model_id": "507f1f77bcf86cd799439011",
        "filename": "model_latest.zip",
        "num_vars": 4,
        "width": 3,
        "tags": ["latest", "production"],
        "upload_date": "2024-01-20T14:45:00Z",
        "size": 2097152,
        "download_url": "/models/download/507f1f77bcf86cd799439011"
    }
    ```

- **Status Codes:**
    - `200 OK`: Successfully retrieved the latest model
    - `404 Not Found`: No model found for the specified num_vars and width
    - `400 Bad Request`: Missing required parameters

#### `GET /models/download/{model_id}`
Download a specific model file (ZIP archive). Users are responsible for decompressing the ZIP file after download.

- **Path Parameters:**
    - `model_id` (string, required): GridFS file ID

- **Response:**
    - Binary ZIP file stream with appropriate headers:
        - `Content-Type: application/zip`
        - `Content-Disposition: attachment; filename="model_file.zip"`

- **Status Codes:**
    - `200 OK`: File successfully streamed
    - `404 Not Found`: Model file not found


##### Implementation Notes

- **Tag Filtering**: When filtering by tags, the query uses MongoDB's `$all` operator to ensure models contain ALL specified tags.
- **File Streaming**: Downloads use GridFS streaming to efficiently handle large model files without loading them entirely into memory.
- **Metadata Queries**: All model queries operate on GridFS metadata for efficient filtering before file retrieval.


#### `POST /models/upload`
Upload a new model as a ZIP archive with associated metadata.

- **Request:** Multipart/form-data with the following fields:
    - `file` (binary, required): ZIP archive containing the model
    - `num_vars` (int, required): Number of variables
    - `width` (int, required): Width parameter
    - `tags` (string, optional): Comma-separated list of tags (e.g., "production,optimized,v2")

- **Response:**
    ```json
    {
        "model_id": "507f1f77bcf86cd799439011",
        "filename": "uploaded_model.zip",
        "num_vars": 4,
        "width": 3,
        "tags": ["production", "optimized"],
        "upload_date": "2024-01-20T15:00:00Z",
        "size": 3145728,
        "message": "Model uploaded successfully"
    }
    ```

- **Status Codes:**
    - `201 Created`: Model uploaded successfully
    - `400 Bad Request`: Missing required fields or invalid file format
    - `413 Payload Too Large`: Uploaded file exceeds size limit
    - `422 Unprocessable Entity`: File is not a valid ZIP archive

#### `DELETE /models/purge`
**DESTRUCTIVE OPERATION**: Completely purge all models from GridFS storage. This operation cannot be undone.

- **Request Body:** None
- **Response:**
    ```json
    {
        "success": true,
        "deleted_count": 42,
        "message": "Successfully purged 42 models from GridFS",
        "freed_space": 157286400,
        "timestamp": "2024-01-21T16:30:45.123456"
    }
    ```

- **Status Codes:**
    - `200 OK`: Successfully purged all models
    - `500 Internal Server Error`: Failed to purge models


#### `GET /health`
Check the health status of the warehouse service.

- **Request:** No parameters required
- **Response:**
    ```json
    {
        "status": "healthy"
    }
    ```
- **Status Codes:**
    - `200 OK`: Service is healthy and ready to accept requests
    - `500 Internal Server Error`: Service is unhealthy (e.g., database connection failed)
- **Description:** 
    - This endpoint performs a basic health check on the warehouse service
    - Verifies MongoDB connectivity during service startup
    - Used for monitoring, load balancers, and container orchestration health checks
    - No authentication required
