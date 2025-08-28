#!/usr/bin/env python3
"""
Script to display formula string representation given either:
1. A node_id from the evolution graph
2. A (traj_id, traj_slice) pair from a trajectory

This script retrieves the formula definition from the warehouse and
displays it in a human-readable string format using NormalFormFormula.
"""

import argparse
import sys
from typing import Optional, List
from longshot.agent import WarehouseAgent
from longshot.circuit.formula import NormalFormFormula, FormulaType
from longshot.utils import parse_formula_definition, parse_trajectory_to_definition


def get_formula_from_node(warehouse: WarehouseAgent, node_id: str) -> Optional[List[int]]:
    """
    Retrieve formula definition from a node_id.
    
    Args:
        warehouse: WarehouseAgent instance
        node_id: Evolution graph node ID
        
    Returns:
        Formula definition as list of integers, or None if not found
    """
    try:
        # Get node information
        node = warehouse.get_evolution_graph_node(node_id)
        if not node:
            print(f"Error: Node {node_id} not found")
            return None
            
        # Get trajectory ID and slice from node
        traj_id = node.get("traj_id")
        traj_slice = node.get("traj_slice")
        
        if not traj_id:
            print(f"Error: Node {node_id} has no associated trajectory")
            return None
            
        print(f"Node {node_id} -> trajectory {traj_id}, slice {traj_slice}")
        
        # Get formula from trajectory
        return get_formula_from_trajectory(warehouse, traj_id, traj_slice)
        
    except Exception as e:
        print(f"Error retrieving node {node_id}: {e}")
        return None


def get_formula_from_trajectory(warehouse: WarehouseAgent, traj_id: str, traj_slice: int) -> Optional[List[int]]:
    """
    Retrieve formula definition from a trajectory at a specific slice.
    
    Args:
        warehouse: WarehouseAgent instance
        traj_id: Trajectory ID
        traj_slice: Slice index in the trajectory
        
    Returns:
        Formula definition as list of integers, or None if not found
    """
    try:
        # Get trajectory
        trajectory = warehouse.get_trajectory(traj_id)
        if not trajectory:
            print(f"Error: Trajectory {traj_id} not found")
            return None
            
        steps = trajectory.get("steps", [])
        
        if traj_slice < 0 or traj_slice >= len(steps):
            print(f"Error: Slice {traj_slice} out of range (trajectory has {len(steps)} steps)")
            return None
            
        # Reconstruct formula up to the specified slice
        prefix_steps = steps[:traj_slice + 1]
        
        # Parse trajectory to get formula definition
        formula_definition = parse_trajectory_to_definition(prefix_steps)
        
        return formula_definition
        
    except Exception as e:
        print(f"Error retrieving trajectory {traj_id}: {e}")
        return None


def display_formula(definition: List[int], num_vars: int = 4, formula_type: FormulaType = FormulaType.Conjunctive):
    """
    Display formula in human-readable string format.
    
    Args:
        definition: Formula definition as list of integers
        num_vars: Number of variables (default 4)
        formula_type: Formula type (default Conjunctive)
    """
    if not definition:
        print("Empty formula (no gates)")
        return
        
    try:
        # Parse formula definition
        formula = parse_formula_definition(definition, num_vars, formula_type)
        
        # Convert to NormalFormFormula for string representation
        if hasattr(formula, 'to_normal_form'):
            nf_formula = formula.to_normal_form()
        else:
            # If it's already a NormalFormFormula
            nf_formula = formula
            
        # Display formula information
        print(f"\nFormula Information:")
        print(f"  Type: {formula_type.name}")
        print(f"  Number of gates: {len(definition)}")
        print(f"  Number of variables: {num_vars}")
        
        # Display string representation
        print(f"\nString representation:")
        print(f"  {nf_formula}")
        
        # Display raw gate definitions
        print(f"\nRaw gate definitions:")
        for i, gate in enumerate(definition):
            print(f"  Gate {i}: {gate} (0x{gate:08x})")
            
    except Exception as e:
        print(f"Error parsing formula: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Display formula string representation from node_id or trajectory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display formula from a node ID
  %(prog)s --node-id abc123-def456-789

  # Display formula from trajectory at specific slice
  %(prog)s --traj-id xyz789-abc123 --traj-slice 42

  # Display with custom number of variables
  %(prog)s --node-id abc123 --num-vars 5

  # Display as DNF formula
  %(prog)s --node-id abc123 --formula-type DNF
        """
    )
    
    # Input options (mutually exclusive group)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--node-id", type=str, help="Evolution graph node ID")
    input_group.add_argument("--traj", nargs=2, metavar=("TRAJ_ID", "SLICE"),
                            help="Trajectory ID and slice index")
    
    # Formula configuration
    parser.add_argument("--num-vars", type=int, default=4,
                       help="Number of variables in the formula (default: 4)")
    parser.add_argument("--formula-type", type=str, default="CNF",
                       choices=["CNF", "DNF"],
                       help="Formula type (default: CNF)")
    
    # Warehouse configuration
    parser.add_argument("--warehouse-host", type=str, default="localhost",
                       help="Warehouse host (default: localhost)")
    parser.add_argument("--warehouse-port", type=int, default=8000,
                       help="Warehouse port (default: 8000)")
    
    args = parser.parse_args()
    
    # Determine formula type
    formula_type = FormulaType.Conjunctive if args.formula_type == "CNF" else FormulaType.Disjunctive
    
    # Initialize warehouse agent
    warehouse = WarehouseAgent(args.warehouse_host, args.warehouse_port)
    
    try:
        # Get formula definition based on input
        if args.node_id:
            print(f"Retrieving formula from node: {args.node_id}")
            definition = get_formula_from_node(warehouse, args.node_id)
        else:
            traj_id, traj_slice = args.traj
            traj_slice = int(traj_slice)
            print(f"Retrieving formula from trajectory: {traj_id}, slice: {traj_slice}")
            definition = get_formula_from_trajectory(warehouse, traj_id, traj_slice)
        
        if definition is None:
            print("Failed to retrieve formula definition")
            sys.exit(1)
            
        # Display the formula
        display_formula(definition, args.num_vars, formula_type)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        warehouse._client.close()


if __name__ == "__main__":
    main()