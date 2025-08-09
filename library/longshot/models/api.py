"""
Pydantic models for the Warehouse API.
"""

from datetime import datetime
from pydantic import BaseModel, Field, model_validator


# Formula-related models
class QueryFormulaInfoResponse(BaseModel):
    """Formula information model."""
    id: str = Field(alias="_id", serialization_alias="id")
    base_formula_id: str | None = None
    trajectory_id: str | None = None
    avgQ: float
    wl_hash: str
    num_vars: int
    width: int
    size: int
    timestamp: datetime
    node_id: str
    
        
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



# Trajectory-related models
class TrajectoryInfoStep(BaseModel):
    """A single step in a trajectory."""
    token_type: int
    token_literals: int
    reward: float


class QueryTrajectoryInfoResponse(BaseModel):
    """Trajectory information model."""
    id: str = Field(alias="_id", serialization_alias="id")
    base_formula_id: str
    timestamp: datetime
    steps: list[TrajectoryInfoStep]
    


class CreateTrajectoryRequest(BaseModel):
    """Request model for creating a trajectory."""
    base_formula_id: str | None = None
    steps: list[TrajectoryInfoStep]


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


# Evolution graph node models
class QueryEvolutionGraphNode(BaseModel):
    """Evolution graph node model."""
    formula_id: str
    avgQ: float
    num_vars: int
    width: int
    size: int
    visited_counter: int
    in_degree: int
    out_degree: int


class CreateNodeRequest(BaseModel):
    """Request model for creating a node."""
    formula_id: str
    avgQ: float
    num_vars: int
    width: int
    size: int
    

class UpdateNodeRequest(BaseModel):
    """Request model for updating a node."""
    formula_id: str
    inc_visited_counter: int | None = None
    visited_counter: int | None = None
    avgQ: float | None = None
    num_vars: int | None = None
    width: int | None = None
    size: int | None = None

    @model_validator(mode='before')
    def check_exclusive_fields(cls, values):
        inc = values.get('inc_visited_counter')
        visited = values.get('visited_counter')
        
        if inc is not None and visited is not None:
            raise ValueError('Only one of inc_visited_counter or visited_counter can be set, not both.')
        
        return values

class NodeResponse(BaseModel):
    """Response model for node creation."""
    formula_id: str


# Formula definition models
class QueryFormulaDefinitionResponse(BaseModel):
    """Formula definition model."""
    id: str
    definition: list[int]


# Path models
class CreateNewPathRequest(BaseModel):
    """Request model for creating a new path."""
    path: list[str] = Field(..., description="List of node IDs representing the path")


# Download Nodes Response
class DownloadNodesResponse(BaseModel):
    """Response model for downloading nodes."""
    nodes: list[QueryEvolutionGraphNode] = Field(..., description="List of evolution graph nodes")


# Download Hypothetical Nodes Response
class QueryHyperNodeInfo(BaseModel):
    """Response model for querying hyper nodes."""
    hnid: int = Field(..., description="Hyper node ID")
    nodes: list[str] = Field(..., description="List of node IDs representing the hyper nodes")
    
class DownloadHyperNodesResponse(BaseModel):
    """Response model for downloading hyper nodes."""
    hypernodes: list[QueryHyperNodeInfo] = Field(..., description="List of hyper nodes with their average Q values and node IDs")


# Generic response models
class SuccessResponse(BaseModel):
    """Generic success response."""
    message: str = "Success"
    

# Arm-related models
class ArmInfo(BaseModel):
    """
    Model representing information about an arm.
    It includes the arm's ID, its value, and any additional metadata.
    """

    formula_id: str
    definition: list[int]

class TopKArmsResponse(BaseModel):
    """
    Response model for the /topk_arms endpoint.
    It defines the structure of the output data.
    """

    top_k_arms: list[ArmInfo]


# Weapon-related models
class WeaponRolloutRequest(BaseModel):
    """Request model for weapon rollout endpoint."""
    num_vars: int = Field(..., description="Number of variables in the formula")
    width: int = Field(..., description="Width of the formula")
    steps_per_trajectory: int  = Field(None, description="Number of steps per trajectory")
    num_trajectories: int  = Field(None, description="Number of trajectories to collect")
    initial_definition: list[int] = Field(..., description="Initial definition of the formula, represented as a list of integers representing gates")
    initial_formula_id: str | None = Field(None, description="ID of the initial formula used as base for trajectories")
    seed: int | None = Field(None, description="Random seed for reproducible trajectory generation. If not provided, randomness will be non-deterministic")

    @model_validator(mode='before')
    def check_exclusive_fields(cls, values):
        num_steps = values.get('num_steps')
        num_trajectories = values.get('num_trajectories')
        
        if num_steps is None and num_trajectories is None:
            raise ValueError('Either num_steps or num_trajectories must be provided')
        
        if num_steps is not None and num_trajectories is not None:
            raise ValueError('Only one of num_steps or num_trajectories can be provided, not both')
        
        return values

class WeaponRolloutResponse(BaseModel):
    """Response model for weapon rollout endpoint."""
    total_steps: int = Field(..., description="Number of steps actually run")
    num_trajectories: int = Field(..., description="Number of trajectories actually collected") 