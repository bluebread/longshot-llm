from ..literals import Literals
from pydantic import BaseModel, ConfigDict, Field

class GateToken(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    type: str
    literals: Literals = Field(..., description="List of literals involved in the operation")
    
    @classmethod
    def dim_token(cls, num_vars: int) -> int:
        return 2 * num_vars + 3