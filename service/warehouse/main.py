"""
FastAPI application for the Warehouse microservice.
"""

from fastapi import FastAPI, HTTPException, Query
from pymongo import MongoClient
from contextlib import asynccontextmanager
import uuid
from datetime import datetime
import logging
from typing import Optional

from longshot.service.api_models import (
    QueryTrajectoryInfoResponse,
    CreateTrajectoryRequest,
    UpdateTrajectoryRequest,
    TrajectoryResponse,
    TrajectoryDatasetResponse,
    OptimizedTrajectoryInfo,
    SuccessResponse,
    PurgeResponse
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


@app.get("/health")
async def health_check():
    """Check the health status of the warehouse service."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)