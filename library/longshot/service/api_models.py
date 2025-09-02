"""
Pydantic models for the Warehouse API.
"""

from datetime import datetime
from pydantic import BaseModel, Field, model_validator


# Trajectory-related models
# TrajectoryInfoStep is now deprecated - using tuple format (int, int, float) instead
# Tuple format: (token_type, token_literals, cur_avgQ)


class QueryTrajectoryInfoResponse(BaseModel):
    """Trajectory information model."""
    traj_id: str = Field(alias="_id", serialization_alias="traj_id")
    timestamp: datetime
    steps: list[tuple[int, int, float]] = Field(..., description="Steps as tuples of (token_type, token_literals, cur_avgQ)")
    num_vars: int | None = Field(None, description="Number of variables in the formula")
    width: int | None = Field(None, description="Width of the formula")
    


class CreateTrajectoryRequest(BaseModel):
    """Request model for creating a trajectory."""
    steps: list[tuple[int, int, float]] = Field(..., 
                                                 description="Steps as tuples of (token_type, token_literals, cur_avgQ)",
                                                 max_length=10000)  # Reasonable limit to prevent resource exhaustion
    num_vars: int | None = Field(..., 
                                     description="Number of variables in the formula",
                                     ge=1, le=32)  # Valid variable range
    width: int | None = Field(..., 
                                 description="Width of the formula",
                                 ge=1, le=32)  # Valid width range


# Trajectory-related models
class UpdateTrajectoryRequest(BaseModel):
    """Request model for updating a trajectory."""
    traj_id: str
    steps: list[tuple[int, int, float]] | None = Field(None, description="Steps as tuples of (token_type, token_literals, cur_avgQ)")


class TrajectoryResponse(BaseModel):
    """Response model for trajectory creation."""
    traj_id: str



class OptimizedTrajectoryInfo(BaseModel):
    """Optimized trajectory information with tuple-based steps."""
    traj_id: str = Field(alias="_id", serialization_alias="traj_id")
    timestamp: datetime
    steps: list[tuple[int, int, float]] = Field(..., description="Steps as tuples of (token_type, token_literals, cur_avgQ)")
    num_vars: int | None = Field(None, description="Number of variables in the formula")
    width: int | None = Field(None, description="Width of the formula")

class TrajectoryDatasetResponse(BaseModel):
    """Response model for complete trajectory dataset."""
    trajectories: list[OptimizedTrajectoryInfo] = Field(..., description="All trajectories in the dataset")


# Database purge response model
class PurgeResponse(BaseModel):
    """Response model for purge operations."""
    success: bool
    deleted_count: int
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)


# Generic response models
class SuccessResponse(BaseModel):
    """Generic success response."""
    message: str = "Success"
    