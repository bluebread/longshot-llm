"""
Pydantic models for the Warehouse API.
"""

from datetime import datetime
from pydantic import BaseModel, Field, model_validator


# Formula-related models are now integrated into Evolution Graph nodes
# in V2 architecture


class LikelyIsomorphicResponse(BaseModel):
    """Response model for likely isomorphic formulas."""
    wl_hash: str
    isomorphic_ids: list[str]


class LikelyIsomorphicRequest(BaseModel):
    """Request model for adding likely isomorphic formula."""
    wl_hash: str
    node_id: str



# Trajectory-related models
class TrajectoryInfoStep(BaseModel):
    """A single step in a trajectory."""
    token_type: int
    token_literals: int
    cur_avgQ: float


class QueryTrajectoryInfoResponse(BaseModel):
    """Trajectory information model."""
    traj_id: str = Field(alias="_id", serialization_alias="traj_id")
    timestamp: datetime
    steps: list[TrajectoryInfoStep]
    


class CreateTrajectoryRequest(BaseModel):
    """Request model for creating a trajectory."""
    steps: list[TrajectoryInfoStep]


# Trajectory-related models
class UpdateTrajectoryStep(BaseModel):
    """A single step in a trajectory."""
    order: int
    token_type: int
    token_literals: int
    cur_avgQ: float

class UpdateTrajectoryRequest(BaseModel):
    """Request model for updating a trajectory."""
    traj_id: str
    steps: list[UpdateTrajectoryStep] | None = None


class TrajectoryResponse(BaseModel):
    """Response model for trajectory creation."""
    traj_id: str


# Evolution graph node models
class QueryEvolutionGraphNode(BaseModel):
    """Evolution graph node model with integrated formula data."""
    node_id: str
    avgQ: float
    num_vars: int
    width: int
    size: int
    in_degree: int
    out_degree: int
    wl_hash: str
    timestamp: datetime
    traj_id: str
    traj_slice: int


class CreateNodeRequest(BaseModel):
    """Request model for creating a node with integrated formula data."""
    node_id: str
    avgQ: float
    num_vars: int
    width: int
    size: int
    wl_hash: str
    traj_id: str
    traj_slice: int
    

class UpdateNodeRequest(BaseModel):
    """Request model for updating a node with integrated formula data."""
    node_id: str
    avgQ: float | None = None
    num_vars: int | None = None
    width: int | None = None
    size: int | None = None
    wl_hash: str | None = None
    traj_id: str | None = None
    traj_slice: int | None = None

class NodeResponse(BaseModel):
    """Response model for node creation."""
    node_id: str


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


# Dataset Response Models
class EvolutionGraphEdge(BaseModel):
    """Edge in the evolution graph."""
    src: str = Field(..., description="Source node ID")
    dst: str = Field(..., description="Target node ID")
    type: str = Field(..., description="Type of edge (EVOLVED_TO or SAME_Q)")

class EvolutionGraphDatasetResponse(BaseModel):
    """Response model for complete evolution graph dataset."""
    nodes: list[dict] = Field(..., description="All nodes in the evolution graph with selected fields")
    edges: list[EvolutionGraphEdge] = Field(..., description="All edges in the evolution graph")

class OptimizedTrajectoryStep(BaseModel):
    """Optimized trajectory step as tuple (type, literals, cur_avgQ)."""
    step: tuple[int, int, float] = Field(..., description="Trajectory step as (token_type, token_literals, cur_avgQ)")

class OptimizedTrajectoryInfo(BaseModel):
    """Optimized trajectory information with tuple-based steps."""
    traj_id: str = Field(alias="_id", serialization_alias="traj_id")
    timestamp: datetime
    steps: list[tuple[int, int, float]] = Field(..., description="Steps as tuples of (token_type, token_literals, cur_avgQ)")

class TrajectoryDatasetResponse(BaseModel):
    """Response model for complete trajectory dataset."""
    trajectories: list[OptimizedTrajectoryInfo] = Field(..., description="All trajectories in the dataset")


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

    node_id: str
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
    size: int = Field(..., description="Size of the formula (number of nodes)")
    steps_per_trajectory: int  = Field(None, description="Number of steps per trajectory")
    num_trajectories: int  = Field(None, description="Number of trajectories to collect")
    initial_definition: list[int] = Field(..., description="Initial definition of the formula, represented as a list of integers representing gates")
    initial_node_id: str | None = Field(None, description="ID of the initial node used as base for trajectories")
    seed: int | None = Field(None, description="Random seed for reproducible trajectory generation. If not provided, randomness will be non-deterministic")


class WeaponRolloutResponse(BaseModel):
    """Response model for weapon rollout endpoint."""
    total_steps: int = Field(..., description="Number of steps actually run")
    num_trajectories: int = Field(..., description="Number of trajectories actually collected") 