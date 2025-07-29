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
    ids: list[str]


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
    base_formula_id: str
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


# # Evolution graph node models
# class EvolutionGraphNode(BaseModel):
#     """Evolution graph node model."""
#     formula_id: str
#     avgQ: float
#     visited_counter: int
#     inactive: bool
#     in_degree: int
#     out_degree: int


# class CreateNodeRequest(BaseModel):
#     """Request model for creating a node."""
#     num_vars: int
#     width: int
#     formula_id: str
#     avgQ: float


# class UpdateNodeRequest(BaseModel):
#     """Request model for updating a node."""
#     node_id: str
#     inc_visited_counter: int | None = None
#     inactive: bool | None = None


# class NodeResponse(BaseModel):
#     """Response model for node creation."""
#     node_id: str


# # Evolution graph edge models
# class EvolutionGraphEdge(BaseModel):
#     """Evolution graph edge model."""
#     base_formula_id: str
#     new_formula_id: str


# class CreateEdgeRequest(BaseModel):
#     """Request model for creating an edge."""
#     base_formula_id: str
#     new_formula_id: str


# class EdgeResponse(BaseModel):
#     """Response model for edge creation."""
#     edge_id: str


# Formula definition models
class FormulaDefinition(BaseModel):
    """Formula definition model."""
    id: str
    definition: list[int]


# # Subgraph models
# class SubgraphResponse(BaseModel):
#     """Response model for subgraph."""
#     nodes: list
#     edges: list


# class SubgraphRequest(BaseModel):
#     """Request model for adding subgraph."""
#     nodes: list
#     edges: list


# High-level API models
# class AddFormulaRequest(BaseModel):
#     """Request model for high-level formula addition."""
#     base_formula_id: str | None = None
#     trajectory_id: str | None = None
#     avgQ: float
#     wl_hash: str
#     num_vars: int
#     width: int
#     size: int

#     class Config:
#         allow_population_by_field_name = True


# class ContractEdgeRequest(BaseModel):
#     """Request model for contracting an edge."""
#     edge_id: str


# Generic response models
class SuccessResponse(BaseModel):
    """Generic success response."""
    message: str = "Success"