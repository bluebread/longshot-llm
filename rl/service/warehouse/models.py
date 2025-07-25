"""
Pydantic models for the Warehouse API.
"""

from typing import List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID


# Formula-related models
class FormulaInfo(BaseModel):
    """Formula information model."""
    id: Optional[str] = None
    base_formula_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    avgQ: Optional[float] = None
    wl_hash: Optional[str] = None
    num_vars: Optional[int] = None
    width: Optional[int] = None
    size: Optional[int] = None
    timestamp: Optional[datetime] = None
    node_id: Optional[str] = None
    full_trajectory_id: Optional[str] = None

    class Config:
        allow_population_by_field_name = True


class CreateFormulaRequest(BaseModel):
    """Request model for creating a formula."""
    base_formula_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    avgQ: float
    wl_hash: str = Field(alias="wl-hash")
    num_vars: int
    width: int
    size: int
    timestamp: Optional[datetime] = None
    node_id: Optional[str] = None
    full_trajectory_id: Optional[str] = None

    class Config:
        allow_population_by_field_name = True


class UpdateFormulaRequest(BaseModel):
    """Request model for updating a formula."""
    id: str
    avgQ: Optional[float] = None
    size: Optional[int] = None


class FormulaResponse(BaseModel):
    """Response model for formula creation."""
    id: str


class LikelyIsomorphicResponse(BaseModel):
    """Response model for likely isomorphic formulas."""
    isomorphic_ids: List[str]


class LikelyIsomorphicRequest(BaseModel):
    """Request model for adding likely isomorphic formula."""
    wl_hash: str = Field(alias="wl-hash")
    formula_id: str

    class Config:
        allow_population_by_field_name = True


# Trajectory-related models
class TrajectoryStep(BaseModel):
    """A single step in a trajectory."""
    order: int
    token_type: int
    token_literals: int
    reward: float


class TrajectoryInfo(BaseModel):
    """Trajectory information model."""
    id: Optional[str] = None
    steps: List[TrajectoryStep]


class CreateTrajectoryRequest(BaseModel):
    """Request model for creating a trajectory."""
    steps: List[TrajectoryStep]


class UpdateTrajectoryRequest(BaseModel):
    """Request model for updating a trajectory."""
    id: str
    steps: List[TrajectoryStep]


class TrajectoryResponse(BaseModel):
    """Response model for trajectory creation."""
    id: str


# Evolution graph node models
class EvolutionGraphNode(BaseModel):
    """Evolution graph node model."""
    formula_id: str
    avgQ: float
    visited_counter: int
    inactive: bool
    in_degree: int
    out_degree: int


class CreateNodeRequest(BaseModel):
    """Request model for creating a node."""
    formula_id: str
    avgQ: float


class UpdateNodeRequest(BaseModel):
    """Request model for updating a node."""
    node_id: str
    inc_visited_counter: Optional[int] = None
    inactive: Optional[bool] = None


class NodeResponse(BaseModel):
    """Response model for node creation."""
    node_id: str


# Evolution graph edge models
class EvolutionGraphEdge(BaseModel):
    """Evolution graph edge model."""
    base_formula_id: str
    new_formula_id: str


class CreateEdgeRequest(BaseModel):
    """Request model for creating an edge."""
    base_formula_id: str
    new_formula_id: str


class EdgeResponse(BaseModel):
    """Response model for edge creation."""
    edge_id: str


# Formula definition models
class FormulaDefinition(BaseModel):
    """Formula definition model."""
    id: str
    definition: List[List[str]]


# Subgraph models
class SubgraphResponse(BaseModel):
    """Response model for subgraph."""
    nodes: List[Any]
    edges: List[Any]


class SubgraphRequest(BaseModel):
    """Request model for adding subgraph."""
    nodes: List[Any]
    edges: List[Any]


# High-level API models
class AddFormulaRequest(BaseModel):
    """Request model for high-level formula addition."""
    base_formula_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    avgQ: float
    wl_hash: str = Field(alias="wl-hash")
    num_vars: int
    width: int
    size: int

    class Config:
        allow_population_by_field_name = True


class ContractEdgeRequest(BaseModel):
    """Request model for contracting an edge."""
    edge_id: str


# Generic response models
class SuccessResponse(BaseModel):
    """Generic success response."""
    message: str = "Success"
