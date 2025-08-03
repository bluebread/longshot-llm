from typing import Any, Dict, List
from pydantic import BaseModel, Field
from lsutils import TrajectoryMessage, TrajectoryStep, TrajectoryStep

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

    top_k_arms: List[ArmInfo] 
    
class Arm(BaseModel):
    """
    Model representing an arm in the context of reinforcement learning.
    It includes the arm's ID, its value, and any additional metadata.
    """

    node_id: str = Field(..., description="Unique identifier for the arm node")
    definition: set[int] = Field(..., description="Set of integers representing the arm's definition")
    avgQ: float = Field(..., description="Average Q-value of the arm")
    visited_counter: int = Field(..., description="Counter for the number of times the arm has been visited")
    in_degree: int = Field(..., description="In-degree of the arm node in the evolution graph")
    out_degree: int = Field(..., description="Out-degree of the arm node in the evolution graph")
