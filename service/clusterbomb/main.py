"""
FastAPI application for the Clusterbomb microservice.

This service provides weapon rollout functionality for the longshot system.
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
from longshot.models import WeaponRolloutRequest, WeaponRolloutResponse

logging.basicConfig(
    level=logging.INFO, 
    filename="clusterbomb.log", 
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Clusterbomb service starting up")
    yield
    logger.info("Clusterbomb service shutting down")

app = FastAPI(
    title="Clusterbomb API",
    description="Weapon rollout service for the longshot system",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "clusterbomb"}

@app.post("/weapon/rollout", response_model=WeaponRolloutResponse)
async def weapon_rollout(request: WeaponRolloutRequest):
    """
    Execute a weapon rollout operation.
    
    Collects trajectories from the environment and pushes them to the trajectory queue.
    The request specifies the number of steps to run and the initial formula's definition.
    """
    try:
        logger.info(f"Weapon rollout requested: num_vars={request.num_vars}, width={request.width}")
        
        # TODO: Implement the actual weapon rollout logic here
        # This should:
        # 1. Initialize the RL environment with the given parameters
        # 2. Run the environment for num_steps or until num_trajectories collected
        # 3. Push collected trajectories to the trajectory queue
        # 4. Return actual counts of steps and trajectories
        
        # Placeholder implementation - replace with actual logic
        actual_steps = request.num_steps if request.num_steps else 50  # Default fallback
        actual_trajectories = request.num_trajectories if request.num_trajectories else 5  # Default fallback
        
        # If num_steps was provided, we might collect fewer trajectories
        # If num_trajectories was provided, we might run fewer steps
        if request.num_steps:
            # Simulate collecting some trajectories during the steps
            actual_trajectories = min(actual_steps // 10, 10)  # Rough estimation
        elif request.num_trajectories:
            # Simulate running some steps to collect trajectories
            actual_steps = actual_trajectories * 15  # Rough estimation
        
        logger.info(f"Weapon rollout completed: {actual_steps} steps, {actual_trajectories} trajectories")
        
        return WeaponRolloutResponse(
            num_steps=actual_steps,
            num_trajectories=actual_trajectories
        )
        
    except Exception as e:
        logger.error(f"Error during weapon rollout: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Weapon rollout failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)