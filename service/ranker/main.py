"""
This service provides an API to filter and retrieve arm-related data.
"""

from fastapi import FastAPI, HTTPException, Query, Response
from contextlib import asynccontextmanager
import httpx
import logging
import pathlib
import uuid
import numpy as np
import math
import random
import asyncio
from functools import reduce
from collections import namedtuple
from pydantic import BaseModel
from longshot.models import TopKArmsResponse
from longshot.agent import TrajectoryQueueAgent, AsyncWarehouseAgent
from longshot.utils import to_lambda

config = {
    "eps": 0.1,  # Small value to avoid division by zero
    "wq": 1.0,  # Weight for the average Q value
    "wvc": 2.0,  # Weight for the visited counter
}
config = namedtuple("Config", config.keys())(**config)

logging.basicConfig(
    level=logging.INFO, 
    filename="ranker.log", 
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)

trajectory_queue = TrajectoryQueueAgent(host="rabbitmq-bread", port=5672)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    trajectory_queue.close()

app = FastAPI(
    title="Arm Filter API",
    lifespan=lifespan,
)

class Arm(BaseModel):
    avgQ: float
    visited_counter: int
    in_degree: int
    out_degree: int


def score(arm: Arm, num_vars: int, total_visited: int) -> float:
    """
    Scores a single arm (formula) based on its properties.

    :param arm: A dictionary representing an arm (formula) with its properties.
    :param num_vars: The number of variables in the formula, used for normalization.
    :param total_visited: The total number of visits across all arms, used for normalization.
    :return: A float score representing the performance and potential of the arm.
    """
    lmbd = to_lambda(arm.avgQ, n=num_vars, eps=config.eps)

    # Compute the score using a weighted formula
    score = (
        config.wq * (arm.avgQ + lmbd)
        + config.wvc * math.sqrt(math.log(total_visited) / (arm.visited_counter + 1))
    )
    return score


@app.get("/topk_arms", response_model=TopKArmsResponse)
async def topk_arms(
    num_vars: int = Query(..., description="The number of variables in the formula"),
    width: int = Query(..., description="The width of the formula"),
    k: int = Query(..., description="The number of top arms to return"),
    size_constraint: int | None = Query(default=None, description="The maximum size of the formula. Default: None"),
):
    """
    Filter arms based on the provided criteria.
    """
    async with AsyncWarehouseAgent(host="localhost", port=8000) as warehouse:
        params = {
            "num_vars": num_vars,
            "width": width,
            "size_constraint": size_constraint,
        }
        c1 = warehouse.download_nodes(**params)
        c2 = warehouse.download_hypernodes(**params)

        nodes, hypernodes = await asyncio.gather(c1, c2)
        
    nmap = { node.pop("formula_id"): node for node in nodes }
    hmap = { hn["hnid"]: hn["nodes"] for hn in hypernodes }
    hset = set(reduce(lambda x, y: x + y, hmap.values(), []))
    
    armmap = {
        hnid: Arm(**{
            "avgQ": nmap[nodes[0]]['avgQ'],
            "visited_counter": sum([nmap[nid]['visited_counter'] for nid in nodes]),
            "in_degree": sum([nmap[nid]['in_degree'] for nid in nodes]),
            "out_degree": sum([nmap[nid]['out_degree'] for nid in nodes]),
        })
        for hnid, nodes in hmap.items()
    }
    armmap.update({
        nid: Arm(**node) for nid, node in nmap.items() if nid not in hset
    })
    
    total_visited = sum(arm.visited_counter for arm in armmap.values())
    ranking = sorted([
        (
            score(arm, num_vars, total_visited), 
            aid,
        )
        for aid, arm in armmap.items()
    ], reverse=True)
    
    rng = random.Random()
    selected_fids = [rng.choice(hmap[aid]) if aid in hmap else aid for _, aid in ranking[:k]]
    selected_arms = []
    
    async with AsyncWarehouseAgent(host="localhost", port=8000) as warehouse:
        coroutines = [
            warehouse.get_formula_definition(fid)
            for fid in selected_fids
        ]
        definitions = await asyncio.gather(*coroutines)

    for fid, definition in zip(selected_fids, definitions):
        selected_arms.append({
            "formula_id": fid,
            "definition": definition,
        })

    return TopKArmsResponse(top_k_arms=selected_arms)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
