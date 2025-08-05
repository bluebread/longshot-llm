"""
This service provides an API to filter and retrieve arm-related data.
"""

from fastapi import FastAPI, HTTPException, Query, Response
from contextlib import asynccontextmanager
import httpx
import logging
import threading
import pathlib
import pickle
import uuid
from schedule import every, repeat, run_pending
from longshot.models import TopKArmsResponse
from longshot.agent import TrajectoryQueueAgent

from .processor import TrajectoryProcessor
from .evograph import EvolutionGraphManager
from .ranker import ArmRanker

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

def scheduled_thread():
    """
    A function that runs in a separate thread.
    It can be used to perform operations that should not block the main thread.
    """
    
    processor = TrajectoryProcessor(warehouse=warehouse)
    graph_manager = EvolutionGraphManager()
    ranker = ArmRanker(wq=1.0, wvc=1.0)
    
    def consume_trajectory(trajectory: dict):
        """
        Consume trajectories from the queue and process them.
        """

        data = processor.process_trajectory(trajectory)
        graph_manager.update_graph(**data)
        arms = graph_manager.get_active_nodes()
        ranking = ranker.rank_arms(arms)
        new_ranking = ranking_dir / f"ranking_{uuid.uuid4()}.pkl"
        
        with open(new_ranking, "wb") as f:
            pickle.dump(ranking, f)
        
    trajectory_queue.start_consuming(consume_trajectory)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=scheduled_thread, daemon=True).start()
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
    files = list(ranking_dir.glob("ranking_*.pkl"))
    newest = max(files, key=lambda f: f.stat().st_mtime) if files else None

    if newest:
        with open(newest, "rb") as f:
            ranking = pickle.load(f)
    else:
        ranking = []

    results = []

    with httpx.AsyncClient(base_url=warehouse_url) as client:
        for fid, _ in ranking[:k]: 
            fdef = await client.get(f"/formulas/definition", params={"formula_id": fid})
            
            if fdef.status_code == 200:
                results.append(fdef.json())

    return results



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8050)
