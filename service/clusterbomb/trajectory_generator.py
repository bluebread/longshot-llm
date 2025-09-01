"""
Internal trajectory generator for MAP-Elites mutations.
"""

import random
from typing import List, Dict, Any, Optional, Tuple
from longshot.formula import FormulaRewardModel, FormulaType
from longshot.utils import (
    parse_formula_definition,
    generate_uniform_token,
    parse_trajectory_to_definition,
    parse_gate_integer_representation
)
from longshot.error import LongshotError
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)


def run_mutations_sync(
    num_vars: int,
    width: int,
    num_trajectories: int,
    steps_per_trajectory: int,
    prefix_traj: List[Tuple[int, int, float]],
    early_stop: bool = True,
    seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Run trajectory mutations synchronously (for use in multiprocessing).
    
    Uses generate_uniform_token for smart token generation that ensures:
    - ADD tokens only for gates not in the formula
    - DELETE tokens only for gates already in the formula
    - Uniform sampling across all possible valid operations
    
    Args:
        num_vars: Number of variables in formulas
        width: Width constraint for formulas
        num_trajectories: Number of trajectories to generate
        steps_per_trajectory: Steps per trajectory
        prefix_traj: Prefix trajectory to start from
        early_stop: Stop when avgQ reaches 0
        seed: Random seed for reproducibility
    
    Returns:
        List of trajectory dictionaries
    """
    # Create local RNG instance
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    trajectories = []
    
    # Reconstruct base formula from prefix trajectory
    base_gates = parse_trajectory_to_definition(prefix_traj)
    try:
        initial_formula = parse_formula_definition(
            base_gates,
            num_vars,
            FormulaType.Conjunctive
        )
    except LongshotError as e:
        logger.warning(f"Failed to parse initial formula: {e}")
        # Start with empty formula if parsing fails
        initial_formula = parse_formula_definition(
            [],
            num_vars,
            FormulaType.Conjunctive
        )
    
    # Generate trajectories
    for traj_idx in range(num_trajectories):
        try:
            # Initialize game environment
            game = FormulaRewardModel(
                initial_formula,
                width=width,
                penalty=-1.0
            )
            
            # Collect trajectory
            trajectory_steps = []
            total_steps = 0
            
            for step_idx in range(steps_per_trajectory):
                # Generate uniform token that's aware of current formula state
                # This ensures valid ADD/DELETE operations
                try:
                    current_gates = game.gates  # Get current gates from the game
                    token = generate_uniform_token(
                        num_vars,
                        width,
                        current_gates,
                        rng
                    )
                except ValueError:
                    break  # No valid token could be generated
                
                if token is None:
                    break  # No valid moves
                
                # Take action using step method (not take_action)
                game.step(token)  # Returns reward but we don't need it
                avgQ = game.cur_avgQ  # Use cur_avgQ property (not avgQ)
                
                # Record step - convert GateToken to our trajectory format
                # token.type is 'ADD' or 'DEL', we need 0 or 1
                token_type = 0 if token.type == 'ADD' else 1
                # Convert Literals to integer representation
                litint = int(token.literals)
                trajectory_steps.append((token_type, litint, avgQ))
                
                total_steps += 1
                
                # Early stopping
                if early_stop and avgQ <= 0:
                    break
            
            # Create trajectory dictionary
            trajectory = {
                "traj_id": str(uuid.uuid4()),
                "steps": trajectory_steps,
                "num_vars": num_vars,
                "width": width,
                "timestamp": datetime.now().isoformat(),
                "prefix_length": len(prefix_traj)
            }
            
            trajectories.append(trajectory)
            
        except Exception as e:
            logger.warning(f"Failed to generate trajectory {traj_idx}: {e}")
            continue
    
    return trajectories


class TrajectoryGenerator:
    """Generator for creating trajectory mutations internally"""
    
    def __init__(self, config: dict):
        """
        Initialize trajectory generator.
        
        Args:
            config: Configuration dictionary with num_vars, width
        """
        self.num_vars = config.get("num_vars", 4)
        self.width = config.get("width", 3)
        self.rng = random.Random()
    
    def generate_initial_trajectories(
        self,
        num_trajectories: int,
        steps_per_trajectory: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Generate initial random trajectories from empty formula.
        
        Args:
            num_trajectories: Number of trajectories to generate
            steps_per_trajectory: Steps per trajectory
        
        Returns:
            List of trajectory dictionaries
        """
        return run_mutations_sync(
            num_vars=self.num_vars,
            width=self.width,
            num_trajectories=num_trajectories,
            steps_per_trajectory=steps_per_trajectory,
            prefix_traj=[],  # Start from empty
            early_stop=True
        )
    
    def mutate_from_prefix(
        self,
        prefix_traj: List[Tuple[int, int, float]],
        num_trajectories: int,
        steps_per_trajectory: int
    ) -> List[Dict[str, Any]]:
        """
        Generate mutations from a prefix trajectory.
        
        Args:
            prefix_traj: Prefix trajectory to start from
            num_trajectories: Number of trajectories to generate
            steps_per_trajectory: Steps per trajectory
        
        Returns:
            List of trajectory dictionaries
        """
        return run_mutations_sync(
            num_vars=self.num_vars,
            width=self.width,
            num_trajectories=num_trajectories,
            steps_per_trajectory=steps_per_trajectory,
            prefix_traj=prefix_traj,
            early_stop=True
        )
    
    def validate_trajectory_prefix(
        self,
        prefix_traj: List[Tuple[int, int, float]]
    ) -> bool:
        """
        Validate that a prefix trajectory produces a valid formula.
        
        Args:
            prefix_traj: Prefix trajectory to validate
        
        Returns:
            True if valid, False otherwise
        """
        formula_gates = []
        current_width = 0
        used_variables = 0
        
        for step in prefix_traj:
            if isinstance(step, (list, tuple)) and len(step) == 3:
                token_type, litint, _ = step
            else:
                continue
            
            lits = parse_gate_integer_representation(litint)
            used_variables |= (lits.pos | lits.neg)
            
            if token_type == 0:  # ADD
                formula_gates.append(litint)
                current_width = max(current_width, lits.width)
            elif token_type == 1:  # DELETE
                if litint in formula_gates:
                    formula_gates.remove(litint)
                    if formula_gates:
                        current_width = max(
                            parse_gate_integer_representation(g).width
                            for g in formula_gates
                        )
                    else:
                        current_width = 0
        
        # Check if resulting formula is valid
        valid = (
            len(formula_gates) > 0 and
            current_width <= self.width and
            used_variables.bit_count() <= self.num_vars
        )
        
        return valid