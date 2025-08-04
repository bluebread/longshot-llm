from pydantic import BaseModel, Field
from datetime import datetime

class TrajectoryMessageStep(BaseModel):
    """
    Model representing a single step in a trajectory.
    It includes the arm ID and the reward received for that step.
    """

    order: int = Field(..., description="Order of the step in the trajectory")
    token_type: str = Field(..., description="Type of the token (e.g., 'arm')")
    token_literals: int = Field(..., description="Literal associated with the token")
    reward: float = Field(..., description="Reward received for this step")
    avgQ: float = Field(..., description="Average Q-value for this step")

class TrajectoryMessageMultipleSteps(BaseModel):
    """
    Model representing a trajectory in the context of reinforcement learning.
    It includes the trajectory ID, the arm ID, and the trajectory data.
    """
    
    base_formula_id: str | None = Field(None, description="ID of the base formula for the trajectory")
    steps: list[TrajectoryMessageStep] = Field(..., description="List of steps in the trajectory")

class TrajectoryQueueMessage(BaseModel):
    """
    Model representing a trajectory message.
    It includes the trajectory ID, the arm ID, and the trajectory data.
    """
    
    num_vars: int = Field(..., description="Number of variables in the trajectory")
    width: int = Field(..., description="Width of the trajectory")
    base_size: int = Field(..., description="Size of the base formula")
    timestamp: datetime = Field(..., description="Timestamp of the trajectory")
    trajectory: TrajectoryMessageMultipleSteps = Field(..., description="The trajectory data itself")
