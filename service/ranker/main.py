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
import numpy as np
import math


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

class ArmRanker:

    def __init__(self, **config):
        self.wq = config.get("wq", 1.0)  # Weight for average Q value
        self.wvc = config.get("wvc", 1.0)  # Weight

    def rank_arms(self, arms: list[dict]) -> list[int]:
        """
        Ranks the given arms (formulas) based on their performance and potential.

        :param arms: A list of dictionaries, each representing an arm (formula) with its properties.
        :return: A list of indices representing the ranked order of the arms.
        """
        tvc = sum(arm.get("visited_counter", 0) for arm in arms)
        scores = np.array([self.score(arm, tvc) for arm in arms])
        ids = np.array([arm.get("id", i) for i, arm in enumerate(arms)])
        sorted_idx = np.argsort(scores)
        
        return list(zip(ids[sorted_idx].tolist(), scores[sorted_idx].tolist()))

    def score(self, arm: dict, total_visited: int) -> float:
        """
        Scores a single arm (formula) based on its properties.

        :param arm: A dictionary representing an arm (formula) with its properties.
        :param total_visited: The total number of visits across all arms, used for normalization.
        :return: A float score representing the performance and potential of the arm.
        """
        q = arm.get("avgQ", 0)
        tvc = total_visited if total_visited > 0 else 1
        vc = arm.get("visited_counter", 0)
        di = arm.get("in_degree", 0)
        do = arm.get("out_degree", 0)

        # Compute the score using a weighted formula
        score = (
            self.wq * q 
            + self.wvc * math.sqrt(math.log(tvc) / (vc + 1)) 
        )
        return score


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
