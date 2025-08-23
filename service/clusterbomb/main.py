"""
FastAPI application for the Clusterbomb microservice.

This service provides weapon rollout functionality for the longshot system.
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import logging
import random
from datetime import datetime
from longshot.models import WeaponRolloutRequest, WeaponRolloutResponse
from longshot.models.api import TrajectoryProcessingContext, TrajectoryInfoStep
from longshot.agent import TrajectoryProcessor, WarehouseAgent
import time
from longshot.circuit import FormulaType
from longshot.env import FormulaGame
from longshot.utils import parse_formula_definition, generate_random_token
from longshot.error import LongshotError

warehouse_host = 'localhost'
warehouse_port = 8000

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
    
    Collects trajectories from the environment and processes them directly using TrajectoryProcessor.
    The request specifies the number of steps per trajectory, the initial formula's definition,
    and optionally a random seed for reproducible results.
    
    If a seed is provided, all random token generation will be deterministic and reproducible.
    Without a seed, the service uses non-deterministic randomness.
    """
    logger.info(f"Weapon rollout requested: num_vars={request.num_vars}, width={request.width}")
    logger.info(f"V2 prefix trajectory with {len(request.prefix_traj)} steps")
    
    # Create local RNG instance for coroutine safety
    if request.seed is not None:
        rng = random.Random(request.seed)
        logger.info(f"Random seed set to: {request.seed}")
    else:
        rng = random.Random()
        logger.info("No seed provided - using non-deterministic randomness")
    
    # 1. V2: Reconstruct base formula from prefix trajectory
    # Initialize temporary processor for base formula reconstruction
    temp_processor = TrajectoryProcessor(None)
    base_formula_graph = temp_processor.reconstruct_base_formula(request.prefix_traj)
    
    # Convert FormulaGraph to NormalFormFormula for game initialization
    # Extract gates from the reconstructed formula graph
    base_gates = list(base_formula_graph.gates) if hasattr(base_formula_graph, 'gates') else []
    initial_formula = parse_formula_definition(
        base_gates,
        request.num_vars,
        FormulaType.Conjunctive
    )
    
    logger.info(f"Reconstructed base formula from prefix trajectory with {initial_formula.num_gates} gates")
    
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
    
    # 3. Initialize warehouse agent and trajectory processor
    warehouse = WarehouseAgent(warehouse_host, warehouse_port)
    processor = TrajectoryProcessor(warehouse)
    
    # V2: Check if base formula already exists in database
    base_formula_exists, base_formula_id = processor.check_base_formula_exists(base_formula_graph)
    logger.info(f"Base formula exists in database: {base_formula_exists}, ID: {base_formula_id}")
    
    # If base formula doesn't exist, save it to warehouse
    if not base_formula_exists:
        logger.info("Base formula not found, creating new node in warehouse")
        
        # Save the complete prefix trajectory
        prefix_traj_id = processor.warehouse.post_trajectory(
            steps=[
                {
                    "token_type": step.token_type,
                    "token_literals": step.token_literals,
                    "cur_avgQ": step.cur_avgQ,
                }
                for step in request.prefix_traj
            ]
        )
        
        # Calculate base formula properties
        base_wl_hash = base_formula_graph.wl_hash(iterations=processor.hash_iterations)
        base_size = sum(1 for step in request.prefix_traj if step.token_type == 0) - sum(1 for step in request.prefix_traj if step.token_type == 1)
        final_avgQ = request.prefix_traj[-1].cur_avgQ if request.prefix_traj else 0.0
        
        # Create evolution graph node for base formula
        base_formula_id = processor.warehouse.post_evolution_graph_node(
            avgQ=final_avgQ,
            num_vars=request.num_vars,
            width=request.width,
            size=base_size,
            wl_hash=base_wl_hash,
            traj_id=prefix_traj_id,
            traj_slice=len(request.prefix_traj) - 1
        )
        
        # Add to isomorphism hash table
        processor.warehouse.post_likely_isomorphic(
            wl_hash=base_wl_hash,
            formula_id=base_formula_id
        )
        
        base_formula_exists = True  # Now it exists
        logger.info(f"Created new base formula node: {base_formula_id}")
        
    # 4. Run the environment simulation - Server errors return 500
    try:
        actual_trajectories = 0
        v2_contexts = []  # V2 processing contexts
        total_processed_formulas = 0
        total_new_nodes = 0
        
        # Determine stopping condition and trajectory length
        steps_per_trajectory = request.steps_per_trajectory 
        max_trajectories = request.num_trajectories
        
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
                
                # Record step for V2 trajectory format
                step_data = {
                    "token_type": token.type if isinstance(token.type, int) else (0 if token.type == 'ADD' else 1),
                    "token_literals": int(token.literals),
                    "cur_avgQ": game.cur_avgQ
                }
                current_trajectory.append(step_data)
            
            # V2: Create TrajectoryProcessingContext with suffix trajectory
            suffix_steps = [
                TrajectoryInfoStep(
                    token_type=step["token_type"],
                    token_literals=step["token_literals"],
                    cur_avgQ=step["cur_avgQ"]
                )
                for step in current_trajectory
            ]
            
            context = TrajectoryProcessingContext(
                prefix_traj=request.prefix_traj,
                suffix_traj=suffix_steps,
                base_formula_hash=None,  # Will be computed during processing
                processing_metadata={
                    "num_vars": request.num_vars,
                    "width": request.width,
                    "size": request.size
                }
            )
            v2_contexts.append(context)
            
            actual_trajectories += 1
            logger.info(f"Completed trajectory {actual_trajectories} with {len(current_trajectory)} steps")
        
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Simulation failed: {str(e)}"
        )
        
    logger.info(f"Collected {len(v2_contexts)} trajectories for processing")
    
    # V2: Process all trajectories using new trajectory processing
    trajectories_processed = 0
    try:
        for context in v2_contexts:
            result = processor.process_trajectory(context)
            total_processed_formulas += result["processed_formulas"]
            total_new_nodes += result["new_nodes_created"]
            trajectories_processed += 1
            logger.info(f"V2 Processed trajectory {trajectories_processed}/{len(v2_contexts)}: {result['new_nodes_created']} new nodes")
        
        logger.info(f"Successfully processed {trajectories_processed} trajectories")
        
    except Exception as process_error:
        logger.error(f"Error processing trajectories: {str(process_error)}", exc_info=True)
        # Don't fail the whole request if processing fails - just log it
        
    return WeaponRolloutResponse(
        total_steps=request.steps_per_trajectory * actual_trajectories,
        num_trajectories=actual_trajectories,
        processed_formulas=total_processed_formulas,
        new_nodes_created=total_new_nodes,
        base_formula_exists=base_formula_exists
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8060)