"""
FastAPI application for the Warehouse microservice.
"""

from fastapi import FastAPI, HTTPException, Query, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from pymongo import MongoClient, ASCENDING, DESCENDING
from gridfs import GridFS
from contextlib import asynccontextmanager
import uuid
from datetime import datetime
import logging
from typing import Optional
import zipfile
from io import BytesIO
from bson import ObjectId

from longshot.service.api_models import (
    QueryTrajectoryInfoResponse,
    CreateTrajectoryRequest,
    UpdateTrajectoryRequest,
    TrajectoryResponse,
    TrajectoryDatasetResponse,
    OptimizedTrajectoryInfo,
    SuccessResponse,
    PurgeResponse,
    ModelMetadata,
    ModelsListResponse,
    ModelUploadResponse,
    ModelsPurgeResponse
)

logging.basicConfig(
    level=logging.INFO, 
    filename="warehouse.log", 
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)

mongo_client = MongoClient("mongodb://haowei:bread861122@mongo-bread:27017")
mongodb = mongo_client["LongshotWarehouse"]
gridfs = GridFS(mongodb)

@asynccontextmanager
async def lifespan(_: FastAPI):
    # Check availability of external services
    try:
        # Test MongoDB connection
        mongodb.command("ping")
    except Exception:
        raise HTTPException(status_code=500, detail="MongoDB connection failed")
    
    # Initialize MongoDB resources
    try:
        # TODO: add validator
        mongodb.create_collection("TrajectoryTable", check_exists=True)
        # TODO: Add index on timestamp field for efficient range queries
        mongodb["TrajectoryTable"].create_index("timestamp")
        
        # Initialize GridFS indexes for Parameter Server
        # Create index on metadata.upload_date for latest model retrieval
        mongodb["fs.files"].create_index([("metadata.upload_date", DESCENDING)])
        # Create compound index for efficient queries
        mongodb["fs.files"].create_index([
            ("metadata.num_vars", ASCENDING),
            ("metadata.width", ASCENDING),
            ("metadata.upload_date", DESCENDING)
        ])
    except Exception:
        pass
    
    # Yield control to the application
    yield
    
    # Cleanup resources
    mongo_client.close()
    
app = FastAPI(
    title="Warehouse API",
    lifespan=lifespan,
)
trajectory_table = mongodb["TrajectoryTable"]

# Trajectory endpoints
@app.get("/trajectory", response_model=QueryTrajectoryInfoResponse)
async def get_trajectory(traj_id: str = Query(..., description="Trajectory UUID")):
    """Retrieve a trajectory by its ID."""
    trajectory_doc = trajectory_table.find_one({"_id": traj_id})
    if trajectory_doc:
        # Steps are now stored as lists in MongoDB
        # Convert to tuples for API response (Pydantic will serialize as JSON arrays)
        steps_as_tuples = [
            tuple(step) 
            for step in trajectory_doc.get("steps", [])
        ]
        return QueryTrajectoryInfoResponse(
            _id=trajectory_doc["_id"],
            timestamp=trajectory_doc["timestamp"],
            steps=steps_as_tuples,
            num_vars=trajectory_doc.get("num_vars"),
            width=trajectory_doc.get("width")
        )
    else:
        raise HTTPException(status_code=404, detail="Trajectory not found")


@app.post("/trajectory", response_model=TrajectoryResponse, status_code=201)
async def create_trajectory(trajectory: CreateTrajectoryRequest):
    """Add a new trajectory with V2 schema including cur_avgQ per step."""
    trajectory_id = str(uuid.uuid4())
    
    # Convert steps from tuples (API format) to lists (MongoDB storage)
    # Each step: (token_type, token_literals, cur_avgQ)
    steps_as_lists = [
        list(step) if isinstance(step, tuple) else step
        for step in trajectory.steps
    ]
    
    trajectory_doc = {
        "_id": trajectory_id,
        "timestamp": datetime.now(),
        "num_vars": trajectory.num_vars,
        "width": trajectory.width,
        "steps": steps_as_lists,  # Store as lists for BSON efficiency
    }
    trajectory_table.insert_one(trajectory_doc)
    
    return TrajectoryResponse(traj_id=trajectory_id)


@app.put("/trajectory", response_model=SuccessResponse)
async def update_trajectory(trajectory: UpdateTrajectoryRequest):
    """Update an existing trajectory."""
    trajectory_id = trajectory.traj_id
    
    # Convert steps from tuples (API format) to lists (MongoDB storage)
    steps_as_lists = [
        list(step) if isinstance(step, tuple) else step
        for step in trajectory.steps
    ]
    
    update_doc = {"steps": steps_as_lists}
    result = trajectory_table.update_one(
        {"_id": trajectory_id}, 
        {"$set": update_doc}
    )

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Trajectory not found")

    return SuccessResponse(message="Trajectory updated successfully")


@app.delete("/trajectory", response_model=SuccessResponse)
async def delete_trajectory(traj_id: str = Query(..., description="Trajectory UUID")):
    """Delete a trajectory."""
    result = trajectory_table.delete_one({"_id": traj_id})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Trajectory not found")

    return SuccessResponse(message="Trajectory deleted successfully")


@app.get("/trajectory/dataset", response_model=TrajectoryDatasetResponse)
async def get_trajectory_dataset(
    num_vars: Optional[int] = Query(None, description="Filter trajectories by number of variables"),
    width: Optional[int] = Query(None, description="Filter trajectories by width"),
    since: Optional[datetime] = Query(None, description="Filter trajectories with timestamp after this date (ISO 8601 format)"),
    until: Optional[datetime] = Query(None, description="Filter trajectories with timestamp before this date (ISO 8601 format)")
):
    """Get the complete trajectory dataset with all trajectories using optimized tuple format.
    """
    
    # Build filter query
    filter_query = {}
    
    if num_vars is not None:
        filter_query["num_vars"] = num_vars
    
    if width is not None:
        filter_query["width"] = width
    
    # Add timestamp range filters
    if since is not None or until is not None:
        # Validate that since is not after until
        if since is not None and until is not None and since > until:
            raise HTTPException(
                status_code=400, 
                detail="Invalid date range: 'since' timestamp cannot be after 'until' timestamp"
            )
        
        timestamp_filter = {}
        if since is not None:
            timestamp_filter["$gte"] = since
        if until is not None:
            timestamp_filter["$lte"] = until
        filter_query["timestamp"] = timestamp_filter
    
    # Get filtered trajectories from MongoDB
    trajectories_cursor = trajectory_table.find(filter_query)
    trajectories = []
    
    for trajectory_doc in trajectories_cursor:
        # Convert steps to tuple format (token_type, token_literals, cur_avgQ)
        optimized_steps = [
            tuple(step)
            for step in trajectory_doc.get("steps", [])
        ]
        
        # Create optimized trajectory info
        optimized_trajectory = OptimizedTrajectoryInfo(
            _id=trajectory_doc["_id"],
            timestamp=trajectory_doc["timestamp"],
            steps=optimized_steps,
            num_vars=trajectory_doc.get("num_vars"),
            width=trajectory_doc.get("width")
        )
        trajectories.append(optimized_trajectory)
    
    return TrajectoryDatasetResponse(trajectories=trajectories)


# Database purge endpoints
@app.delete("/trajectory/purge", response_model=PurgeResponse)
async def purge_trajectories():
    """Completely purge all trajectory data from MongoDB."""
    try:
        # Get count before deletion for reporting
        count_before = trajectory_table.count_documents({})
        
        # Drop the entire trajectory collection
        trajectory_table.drop()
        
        # Recreate the collection to ensure it exists for future operations
        mongodb.create_collection("TrajectoryTable")
        
        return PurgeResponse(
            success=True,
            deleted_count=count_before,
            message=f"Successfully purged {count_before} trajectories from MongoDB",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Failed to purge trajectories: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to purge trajectories: {str(e)}"
        )


# Parameter Server endpoints
@app.get("/models", response_model=ModelsListResponse)
async def get_models(
    num_vars: int = Query(..., ge=1, le=32, description="Number of variables"),
    width: int = Query(..., ge=1, le=32, description="Width parameter"),
    tags: Optional[list[str]] = Query(None, description="Filter by tags (models must contain ALL specified tags)")
):
    """Retrieve models matching the specified criteria."""
    # Build query filter
    query_filter = {
        "metadata.num_vars": num_vars,
        "metadata.width": width
    }
    
    # Add tag filter if specified
    if tags:
        query_filter["metadata.tags"] = {"$all": tags}
    
    # Query GridFS files
    files_cursor = mongodb["fs.files"].find(query_filter).sort("metadata.upload_date", DESCENDING)
    
    models = []
    for file_doc in files_cursor:
        model = ModelMetadata(
            model_id=str(file_doc["_id"]),
            filename=file_doc["filename"],
            num_vars=file_doc["metadata"]["num_vars"],
            width=file_doc["metadata"]["width"],
            tags=file_doc["metadata"].get("tags", []),
            upload_date=file_doc["metadata"]["upload_date"],
            size=file_doc["length"],
            download_url=f"/models/download/{str(file_doc['_id'])}"
        )
        models.append(model)
    
    return ModelsListResponse(models=models, count=len(models))


@app.get("/models/latest", response_model=ModelMetadata)
async def get_latest_model(
    num_vars: int = Query(..., ge=1, le=32, description="Number of variables"),
    width: int = Query(..., ge=1, le=32, description="Width parameter")
):
    """Retrieve the most recently uploaded model for the specified num_vars and width."""
    # Query for the latest model
    query_filter = {
        "metadata.num_vars": num_vars,
        "metadata.width": width
    }
    
    file_doc = mongodb["fs.files"].find_one(
        query_filter,
        sort=[("metadata.upload_date", DESCENDING)]
    )
    
    if not file_doc:
        raise HTTPException(
            status_code=404,
            detail=f"No model found for num_vars={num_vars} and width={width}"
        )
    
    return ModelMetadata(
        model_id=str(file_doc["_id"]),
        filename=file_doc["filename"],
        num_vars=file_doc["metadata"]["num_vars"],
        width=file_doc["metadata"]["width"],
        tags=file_doc["metadata"].get("tags", []),
        upload_date=file_doc["metadata"]["upload_date"],
        size=file_doc["length"],
        download_url=f"/models/download/{str(file_doc['_id'])}"
    )


@app.get("/models/download/{model_id}")
async def download_model(model_id: str):
    """Download a specific model file (ZIP archive)."""
    try:
        # Convert string ID to ObjectId
        file_id = ObjectId(model_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid model ID format")
    
    # Check if file exists
    if not gridfs.exists(file_id):
        raise HTTPException(status_code=404, detail="Model file not found")
    
    # Get the file
    grid_file = gridfs.get(file_id)
    
    # Stream the file content
    def iterfile():
        while True:
            chunk = grid_file.read(8192)  # Read in 8KB chunks
            if not chunk:
                break
            yield chunk
    
    return StreamingResponse(
        iterfile(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{grid_file.filename}"'
        }
    )


@app.post("/models/upload", response_model=ModelUploadResponse, status_code=201)
async def upload_model(
    file: UploadFile = File(..., description="ZIP archive containing the model"),
    num_vars: int = Form(..., ge=1, le=32, description="Number of variables"),
    width: int = Form(..., ge=1, le=32, description="Width parameter"),
    tags: Optional[str] = Form(None, description="Comma-separated list of tags")
):
    """Upload a new model as a ZIP archive with associated metadata."""
    # Validate file is a ZIP
    content = await file.read()
    try:
        with zipfile.ZipFile(BytesIO(content)):
            pass  # Just validate it's a valid ZIP
    except zipfile.BadZipFile:
        raise HTTPException(
            status_code=422,
            detail="File is not a valid ZIP archive"
        )
    
    # Parse tags
    tag_list = []
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    
    # Prepare metadata
    upload_date = datetime.now()
    metadata = {
        "num_vars": num_vars,
        "width": width,
        "tags": tag_list,
        "upload_date": upload_date,
        "content_type": "application/zip"
    }
    
    # Store in GridFS
    file_id = gridfs.put(
        content,
        filename=file.filename,
        metadata=metadata
    )
    
    return ModelUploadResponse(
        model_id=str(file_id),
        filename=file.filename,
        num_vars=num_vars,
        width=width,
        tags=tag_list,
        upload_date=upload_date,
        size=len(content),
        message="Model uploaded successfully"
    )


@app.delete("/models/purge", response_model=ModelsPurgeResponse)
async def purge_models():
    """Completely purge all models from GridFS storage."""
    try:
        # Get count and total size before deletion
        files_cursor = mongodb["fs.files"].find({})
        deleted_count = 0
        freed_space = 0
        
        for file_doc in files_cursor:
            freed_space += file_doc["length"]
            deleted_count += 1
        
        # Delete all files from GridFS
        for file_doc in mongodb["fs.files"].find({}):
            gridfs.delete(file_doc["_id"])
        
        return ModelsPurgeResponse(
            success=True,
            deleted_count=deleted_count,
            message=f"Successfully purged {deleted_count} models from GridFS",
            freed_space=freed_space,
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Failed to purge models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to purge models: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Check the health status of the warehouse service."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)