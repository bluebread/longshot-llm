from typing import Any, Dict, List
from pydantic import BaseModel, Field

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