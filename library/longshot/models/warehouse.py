"""
Pydantic models for the Warehouse API.
"""

from datetime import datetime
from pydantic import BaseModel, Field


# Formula-related models
class FormulaInfo(BaseModel):
    """Formula information model."""
    id: str = Field(alias="_id", serialization_alias="id")
    base_formula_id: str
    trajectory_id: str
    avgQ: float
    wl_hash: str
    num_vars: int
    width: int
    size: int
    timestamp: datetime
    node_id: str
    
    class Config:
        validate_by_name = True
        allow_population_by_alias = True
        
class CreateFormulaRequest(BaseModel):
    """Request model for creating a formula."""
    base_formula_id: str | None = None
    trajectory_id: str | None = None
    avgQ: float
    wl_hash: str
    num_vars: int
    width: int
    size: int
    node_id: str


class UpdateFormulaRequest(BaseModel):
    """Request model for updating a formula."""
    id: str
    base_formula_id: str | None = None
    trajectory_id: str | None = None
    avgQ: float | None = None
    wl_hash: str | None = None
    num_vars: int | None = None
    width: int | None = None
    size: int | None = None
    timestamp: datetime | None = None
    node_id: str | None = None


class FormulaResponse(BaseModel):
    """Response model for formula creation."""
    id: str


class LikelyIsomorphicResponse(BaseModel):
    """Response model for likely isomorphic formulas."""
    wl_hash: str
    isomorphic_ids: list[str]


class LikelyIsomorphicRequest(BaseModel):
    """Request model for adding likely isomorphic formula."""
    wl_hash: str
    formula_id: str

    class Config:
        validate_by_name = True


# Trajectory-related models
class TrajectoryStep(BaseModel):
    """A single step in a trajectory."""
    token_type: int
    token_literals: int
    reward: float


class TrajectoryInfo(BaseModel):
    """Trajectory information model."""
    id: str = Field(alias="_id", serialization_alias="id")
    base_formula_id: str
    timestamp: datetime
    steps: list[TrajectoryStep]
    
    class Config:
        validate_by_name = True
        allow_population_by_alias = True


class CreateTrajectoryRequest(BaseModel):
    """Request model for creating a trajectory."""
    base_formula_id: str | None = None
    steps: list[TrajectoryStep]


# Trajectory-related models
class UpdateTrajectoryStep(BaseModel):
    """A single step in a trajectory."""
    order: int
    token_type: int
    token_literals: int
    reward: float

class UpdateTrajectoryRequest(BaseModel):
    """Request model for updating a trajectory."""
    id: str
    base_formula_id: str | None = None
    steps: list[UpdateTrajectoryStep] | None = None


class TrajectoryResponse(BaseModel):
    """Response model for trajectory creation."""
    id: str
    
# Formula definition models
class FormulaDefinition(BaseModel):
    """Formula definition model."""
    id: str
    definition: list[int]


# Generic response models
class SuccessResponse(BaseModel):
    """Generic success response."""
    message: str = "Success"