"""
This service provides an API to filter and retrieve arm-related data.
"""

from fastapi import FastAPI, HTTPException, Query, Response
from contextlib import asynccontextmanager
import httpx
import logging
from models import TopKArmsResponse


logging.basicConfig(
    level=logging.INFO, 
    filename="warehouse.log", 
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)

warehouse_url = "http://localhost:8000"  # URL of the warehouse service
warehouse = httpx.Client(base_url=warehouse_url)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    warehouse.close()

app = FastAPI(
    title="Arm Filter API",
    lifespan=lifespan,
)

@app.get("/topk_arms", response_model=TopKArmsResponse)
async def topk_arms(
    num_vars: int = Query(..., description="The number of variables in the formula"),
    width: int = Query(..., description="The width of the formula"),
    k: int = Query(..., description="The number of top arms to return"),
    size: int | None = Query(default=None, description="The maximum size of the formula. Default: None"),
):
    """
    Filter arms based on the provided criteria.
    """
    # NOTE: This is a stub implementation.
    return {
        "top_k_arms": [
            {
                "formula_id": "f123",
                "definition": [
                    534, 123, 456, 789, 101112
                ]
            },
            {
                "formula_id": "f124",
                "definition": [
                    213, 321, 654, 987, 111213
                ]
            },
        ]
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
