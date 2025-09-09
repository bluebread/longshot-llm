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


# Parameter Server models
class ModelMetadata(BaseModel):
    """Model metadata stored in GridFS."""
    model_id: str = Field(..., description="GridFS file ID")
    filename: str = Field(..., description="Original filename")
    num_vars: int = Field(..., ge=1, le=32, description="Number of variables")
    width: int = Field(..., ge=1, le=32, description="Width parameter")
    tags: list[str] = Field(default_factory=list, description="User-defined tags")
    upload_date: datetime = Field(default_factory=datetime.now, description="Upload timestamp")
    size: int = Field(..., description="File size in bytes")
    download_url: str = Field(..., description="Download URL for the model")


class ModelsListResponse(BaseModel):
    """Response model for listing models."""
    models: list[ModelMetadata] = Field(..., description="List of matching models")
    count: int = Field(..., description="Number of models returned")


class ModelUploadResponse(BaseModel):
    """Response model for model upload."""
    model_id: str = Field(..., description="GridFS file ID")
    filename: str = Field(..., description="Original filename")
    num_vars: int = Field(..., description="Number of variables")
    width: int = Field(..., description="Width parameter")
    tags: list[str] = Field(default_factory=list, description="User-defined tags")
    upload_date: datetime = Field(..., description="Upload timestamp")
    size: int = Field(..., description="File size in bytes")
    message: str = Field(default="Model uploaded successfully", description="Success message")


class ModelsPurgeResponse(BaseModel):
    """Response model for models purge operation."""
    success: bool
    deleted_count: int
    message: str
    freed_space: int = Field(..., description="Total freed space in bytes")
    timestamp: datetime = Field(default_factory=datetime.now)


# Generic response models
class SuccessResponse(BaseModel):
    """Generic success response."""
    message: str = "Success"
    