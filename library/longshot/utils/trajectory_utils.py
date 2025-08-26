"""Utility functions for trajectory processing."""
from ..env import FormulaGraph


def parse_trajectory_to_definition(trajectory_steps: list[tuple[int, int, float]]) -> list[int]:
    """
    Parse a trajectory to extract the formula definition (gates).
    
    This function processes a trajectory and extracts the resulting formula
    definition by applying ADD and DEL operations sequentially.
    
    Args:
        trajectory_steps: List of trajectory steps as tuples (token_type, token_literals, cur_avgQ)
                         where token_type: 0=ADD, 1=DEL, 2=EOS
    
    Returns:
        List of gate integers representing the formula definition
    """
    gates = set()
    
    for step in trajectory_steps:
        token_type = step[0]
        token_literals = step[1]
        
        if token_type == 0:  # ADD
            gates.add(token_literals)
        elif token_type == 1:  # DEL
            gates.discard(token_literals)  # Use discard to avoid KeyError if not present
        elif token_type == 2:  # EOS
            break  # End of sequence
    
    return list(gates)


def reconstruct_formula_from_trajectory(
    trajectory_steps: list[tuple[int, int, float]], 
) -> FormulaGraph:
    """
    Reconstruct a FormulaGraph from a trajectory.
    
    This is a more efficient alternative to using TrajectoryProcessor's
    reconstruct_base_formula when you just need the final formula.
    
    Args:
        trajectory_steps: List of trajectory steps as tuples (token_type, token_literals, cur_avgQ)
        num_vars: Number of variables in the formula
    
    Returns:
        FormulaGraph representing the formula after applying all trajectory steps
    """
    from ..env.formula_graph import FormulaGraph
    
    # Get the formula definition from trajectory
    gates = parse_trajectory_to_definition(trajectory_steps)
    
    # Create and return FormulaGraph
    return FormulaGraph(gates)