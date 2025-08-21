"""
FastAPI application for the Clusterbomb microservice.

This service provides weapon rollout functionality for the longshot system.
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import asyncio
import logging
import random
from datetime import datetime
from longshot.models import WeaponRolloutRequest, WeaponRolloutResponse
from longshot.models.trajectory import TrajectoryQueueMessage, TrajectoryMessageMultipleSteps
from longshot.agent.trajectory_queue import AsyncTrajectoryQueueAgent
from longshot.circuit import FormulaType
from longshot.env import FormulaGame
from longshot.utils import parse_formula_definition, generate_random_token
from longshot.error import LongshotError

trajque_host = 'rabbitmq-bread'
trajque_port = 5672

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
    The request specifies the number of steps per trajectory, the initial formula's definition,
    and optionally a random seed for reproducible results.
    
    If a seed is provided, all random token generation will be deterministic and reproducible.
    Without a seed, the service uses non-deterministic randomness.
    """
    logger.info(f"Weapon rollout requested: num_vars={request.num_vars}, width={request.width}")
    logger.info(f"Initial definition: {request.initial_definition}")
    
    # Create local RNG instance for coroutine safety
    if request.seed is not None:
        rng = random.Random(request.seed)
        logger.info(f"Random seed set to: {request.seed}")
    else:
        rng = random.Random()
        logger.info("No seed provided - using non-deterministic randomness")
    
    # 1. Parse the initial formula definition
    initial_formula = parse_formula_definition(
        request.initial_definition, 
        request.num_vars, 
        FormulaType.Conjunctive  # Assume CNF for now
    )
    
    logger.info(f"Parsed initial formula with {initial_formula.num_gates} gates")
    
    # 2. Initialize the RL environment (FormulaGame) - Validation errors return 422
    try:
        game = FormulaGame(
            initial_formula, 
            width=request.width,
            size=request.size,
            penalty=-1.0
        )
    except LongshotError as game_error:
        logger.warning(f"Invalid game parameters: {str(game_error)}")
        raise HTTPException(
            status_code=422,
            detail=f"Invalid formula parameters: {str(game_error)}. "
                   f"The initial_definition may not be compatible with the specified width ({request.width}) or size ({request.size}) constraints."
        )
    
    # 3. Run the environment simulation - Server errors return 500
    try:
        actual_trajectories = 0
        push_coroutines = []  # Collect coroutines to gather later
        
        # Determine stopping condition and trajectory length
        steps_per_trajectory = request.steps_per_trajectory 
        max_trajectories = request.num_trajectories
        
        # Create queue agent connection once for all pushes
        queue_agent = await AsyncTrajectoryQueueAgent.create(
            host=trajque_host, 
            port=trajque_port,
        )
        
        while actual_trajectories < max_trajectories:
            current_trajectory = []
            
            # Reset game for new trajectory
            if actual_trajectories > 0:
                game.reset()
            
            # Run exactly steps_per_trajectory steps for this trajectory
            for i in range(steps_per_trajectory):
                # Generate random action (token) using local RNG for coroutine safety
                token = generate_random_token(request.num_vars, request.width, rng)
                
                # Take step in environment
                reward = game.step(token)
                
                # Record step (matching TrajectoryMessageStep model)
                step_data = {
                    "order": i,
                    "token_type": token.type,
                    "token_literals": int(token.literals),  # Convert to integer representation
                    "reward": reward,
                    "avgQ": game.cur_avgQ  # Average Q-value for this step
                }
                current_trajectory.append(step_data)
            
            # Push trajectory immediately after completing the round
            queue_message = TrajectoryQueueMessage(
                num_vars=request.num_vars,
                width=request.width,
                base_size=initial_formula.num_gates,
                timestamp=datetime.now(),
                trajectory=TrajectoryMessageMultipleSteps(
                    base_formula_id=request.initial_node_id,  # V2 field name
                    steps=current_trajectory
                )
            )
            
            # Create coroutine for pushing this trajectory and collect it
            push_coroutine = queue_agent.push(queue_message)
            push_coroutines.append(push_coroutine)
            
            actual_trajectories += 1
            logger.info(f"Completed trajectory {actual_trajectories} with {len(current_trajectory)} steps")
        
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )
        
    logger.info(f"Collected {len(push_coroutines)} trajectories for pushing")
    
    # Gather all push coroutines after all rounds are complete
    try:
        # Execute all trajectory pushes concurrently
        await asyncio.gather(*push_coroutines)
        trajectories_pushed = len(push_coroutines)
        
        logger.info(f"Successfully pushed {trajectories_pushed} trajectories to RabbitMQ queue")
        
    except Exception as queue_error:
        logger.error(f"Error pushing trajectories to queue: {str(queue_error)}", exc_info=True)
        # Don't fail the whole request if queue pushing fails - just log it
    finally:
        # Close the queue agent connection after all pushes are complete
        await queue_agent.close()
        
    return WeaponRolloutResponse(
        total_steps=request.steps_per_trajectory * actual_trajectories,
        num_trajectories=actual_trajectories
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)