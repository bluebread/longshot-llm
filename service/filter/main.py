"""
This service provides an API to filter and retrieve arm-related data.
"""

from fastapi import FastAPI, HTTPException, Query, Response
from contextlib import asynccontextmanager
import httpx
import logging
import pathlib
import uuid
from longshot.models import TopKArmsResponse
from longshot.agent import TrajectoryQueueAgent


ranking_dir = pathlib.Path(__file__).parent / "ranking"

logging.basicConfig(
    level=logging.INFO, 
    filename="filter.log", 
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)

warehouse_url = "http://localhost:8000"  # URL of the warehouse service
warehouse = httpx.Client(base_url=warehouse_url)

trajectory_queue = TrajectoryQueueAgent(host="rabbitmq-bread", port=5672)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    warehouse.close()
    trajectory_queue.close()

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
    return TopKArmsResponse(
        arms=[
            {
                "arm_id": str(uuid.uuid4()),
                "score": 0.9,
                "variables": [f"var_{i}" for i in range(num_vars)],
                "width": width,
            }
            for _ in range(k)
        ]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
