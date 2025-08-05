"""
This service provides an API to filter and retrieve arm-related data.
"""

from fastapi import FastAPI, HTTPException, Query, Response
from contextlib import asynccontextmanager
import httpx
import logging
import threading
from schedule import every, repeat, run_pending
from longshot.models import TopKArmsResponse
from longshot.agent import TrajectoryQueueAgent


logging.basicConfig(
    level=logging.INFO, 
    filename="filter.log", 
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)

warehouse_url = "http://localhost:8000"  # URL of the warehouse service
warehouse = httpx.Client(base_url=warehouse_url)

# trajectory_queue = TrajectoryQueueAgent(host="rabbitmq-bread", port=5672)

# @repeat(every(30).seconds)
# def scheduled_task():
#     """
#     A scheduled task that runs every 30 seconds.
#     It can be used to perform periodic operations, such as cleaning up resources or logging.
#     """
#     try:
#         logger.info("Running scheduled task...")
#         # Here you can add any periodic operations, like fetching data from the warehouse
#     except Exception as e:
#         logger.error(f"Error in scheduled task: {e}")
        

# def scheduled_thread():
#     """
#     A function that runs in a separate thread.
#     It can be used to perform operations that should not block the main thread.
#     """
#     while True:
#         run_pending()  # Run any scheduled tasks


@asynccontextmanager
async def lifespan(app: FastAPI):
    # threading.Thread(target=scheduled_thread, daemon=True).start()
    yield
    warehouse.close()
    # trajectory_queue.close()

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
