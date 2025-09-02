#!/usr/bin/env python3
"""
Script to display formula representation given a trajectory ID and slice.

This script retrieves trajectory data from the warehouse and displays
the formula at a specific slice in a human-readable format.

USAGE:
    # Display formula from a trajectory at a specific slice
    python script/show_formula.py --traj xyz789-abc123 42
    
    # Connect to a different warehouse server
    python script/show_formula.py --traj abc123 50 --host remote.server --port 8080

OUTPUT:
    The script will display:
    1. Trajectory metadata (num_vars, width)
    2. Formula reconstruction up to the specified slice
    3. Formula statistics (number of gates, variables used)
    
EXAMPLES:
    # Retrieve formula from MAP-Elites elite stored in archive
    python script/show_formula.py --traj 5f3e4d2c-1a2b-3c4d-5e6f-7a8b9c0d1e2f 127
    
    # Display formula at slice 50 from trajectory
    python script/show_formula.py --traj trajectory_id 50

REQUIREMENTS:
    - Warehouse service must be running (default: localhost:8000)
    - Valid trajectory_id must exist in the warehouse
    - Slice index must be within bounds
"""

import argparse
import sys
from typing import Optional, List, Dict, Any
from longshot.service.warehouse import WarehouseClient
from longshot.utils import parse_trajectory_to_definition, parse_formula_definition
from longshot.literals import Literals
from longshot.formula import FormulaType


def get_trajectory_and_formula(
    warehouse: WarehouseClient, 
    traj_id: str, 
    traj_slice: int
) -> tuple[Optional[Dict[str, Any]], Optional[List[int]], Optional[int], Optional[int]]:
    """
    Retrieve trajectory and reconstruct formula at specified slice.
    
    Args:
        warehouse: WarehouseClient instance
        traj_id: Trajectory ID
        traj_slice: Slice index in the trajectory
        
    Returns:
        Tuple of (trajectory dict, formula gates, num_vars, width), or Nones if not found
    """
    try:
        # Get trajectory
        trajectory = warehouse.get_trajectory(traj_id)
        if not trajectory:
            print(f"Error: Trajectory {traj_id} not found")
            return None, None, None, None
            
        # Extract trajectory components from steps
        steps = trajectory.get("steps", [])
        
        if not steps:
            print(f"Error: Trajectory {traj_id} has no steps")
            return None, None, None, None
            
        if traj_slice < 0 or traj_slice >= len(steps):
            print(f"Error: Slice {traj_slice} out of range (trajectory has {len(steps)} steps)")
            return None, None, None, None
            
        # Steps are in format: [type, litint, avgQ]
        # Convert to tuples for parse_trajectory_to_definition
        trajectory_tuples = [(step[0], step[1], step[2]) for step in steps[:traj_slice + 1]]
        
        # Store full trajectory data for later use
        trajectory["type"] = [step[0] for step in steps]
        trajectory["litint"] = [step[1] for step in steps]
        trajectory["avgQ"] = [step[2] for step in steps]
        
        # Count token types up to the slice
        # Token types: 0=ADD, 1=DEL, 2=EOS
        add_count = sum(1 for step in steps[:traj_slice + 1] if step[0] == 0)
        del_count = sum(1 for step in steps[:traj_slice + 1] if step[0] == 1)
        eos_count = sum(1 for step in steps[:traj_slice + 1] if step[0] == 2)
        trajectory["add_count"] = add_count
        trajectory["del_count"] = del_count
        trajectory["eos_count"] = eos_count
        
        # Get metadata
        num_vars = trajectory.get("num_vars", 4)  # Default to 4 if not found
        width = trajectory.get("width", 3)  # Default to 3 if not found
        
        # Parse trajectory to get formula gates
        formula_gates = parse_trajectory_to_definition(trajectory_tuples)
        
        return trajectory, formula_gates, num_vars, width
        
    except Exception as e:
        print(f"Error retrieving trajectory {traj_id}: {e}")
        return None, None, None, None


def analyze_formula_gates(gates: List[int], num_vars: int) -> Dict[str, Any]:
    """
    Analyze formula gates to extract statistics.
    
    Args:
        gates: List of gate integers
        num_vars: Number of variables
        
    Returns:
        Dictionary with formula statistics
    """
    stats = {
        "num_gates": len(gates),
        "variables_used": set(),
        "max_width": 0,
        "gate_types": {"AND": 0, "OR": 0, "NOT": 0}
    }
    
    for gate in gates:
        # Use Literals to decode the gate
        literals = Literals(gate & 0xFFFFFFFF, (gate >> 32) & 0xFFFFFFFF)
        
        # Get width (number of literals in this gate)
        width = literals.width
        stats["max_width"] = max(stats["max_width"], width)
        
        # Track which variables are used
        for var_idx in range(min(num_vars, 32)):
            if literals.pos & (1 << var_idx):
                stats["variables_used"].add(f"x{var_idx}")
            if literals.neg & (1 << var_idx):
                stats["variables_used"].add(f"¬x{var_idx}")
    
    return stats


def display_formula(
    trajectory: Dict[str, Any],
    gates: List[int], 
    num_vars: int, 
    width: int,
    traj_slice: int
):
    """
    Display formula information and gates.
    
    Args:
        trajectory: Full trajectory data
        gates: Formula gates as list of integers
        num_vars: Number of variables
        width: Formula width
        traj_slice: Slice index that was reconstructed
    """
    if not gates:
        print("Empty formula (no gates)")
        return
        
    # Get formula statistics
    stats = analyze_formula_gates(gates, num_vars)
    
    # Display trajectory information
    print(f"\nTrajectory Information:")
    print(f"  Trajectory ID: {trajectory.get('traj_id', 'unknown')}")
    print(f"  Total steps: {len(trajectory.get('type', []))}")
    print(f"  Slice requested: {traj_slice}")
    print(f"  Number of variables: {num_vars}")
    print(f"  Width constraint: {width}")
    
    # Display token counts
    print(f"\nToken Counts (up to slice {traj_slice}):")
    print(f"  ADD tokens: {trajectory.get('add_count', 0)}")
    print(f"  DEL tokens: {trajectory.get('del_count', 0)}")
    print(f"  EOS tokens: {trajectory.get('eos_count', 0)}")
    
    # Display formula statistics
    print(f"\nFormula Statistics (up to slice {traj_slice}):")
    print(f"  Number of gates: {stats['num_gates']}")
    print(f"  Variables used: {', '.join(sorted(stats['variables_used']))}")
    print(f"  Maximum gate width: {stats['max_width']}")
    
    # Display raw gate definitions
    print(f"\nRaw gate definitions:")
    for i, gate in enumerate(gates):
        # Use Literals to decode the gate
        pos_bits = gate & 0xFFFFFFFF
        neg_bits = (gate >> 32) & 0xFFFFFFFF
        literals_obj = Literals(pos_bits, neg_bits)
        
        # Build string representation
        literals = []
        for var_idx in range(min(num_vars, 32)):
            if literals_obj.pos & (1 << var_idx):
                literals.append(f"x{var_idx}")
            if literals_obj.neg & (1 << var_idx):
                literals.append(f"¬x{var_idx}")
        
        gate_str = " ∨ ".join(literals) if literals else "empty"
        
        # Print gate with binary representations
        print(f"  Gate {i}: ({gate_str}) \t {gate} \t pos=0b{pos_bits:b}, neg=0b{neg_bits:b}")
    
    # Display trajectory rewards if available
    if "avgQ" in trajectory:
        avgQs = trajectory["avgQ"]
        if traj_slice < len(avgQs):
            print(f"\nReward at slice {traj_slice}: {avgQs[traj_slice]:.4f}")
    
    # Parse formula from gates and calculate avgQ
    print(f"\nFormula Analysis:")
    try:
        # Parse the formula definition to get NormalFormFormula
        formula = parse_formula_definition(gates, num_vars, FormulaType.Conjunctive)
        
        # Calculate avgQ from the formula
        calculated_avgQ = formula.avgQ()
        print(f"  Calculated avgQ from formula: {calculated_avgQ:.6f}")
        
        # Compare with trajectory avgQ if available
        if "avgQ" in trajectory and traj_slice < len(trajectory["avgQ"]):
            stored_avgQ = trajectory["avgQ"][traj_slice]
            difference = abs(calculated_avgQ - stored_avgQ)
            match_status = "✓ MATCH" if difference < 1e-6 else "✗ MISMATCH"
            
            print(f"  Stored avgQ in trajectory:    {stored_avgQ:.6f}")
            print(f"  Difference:                   {difference:.6f} {match_status}")
            
            if difference >= 1e-6:
                print(f"\n  WARNING: The calculated avgQ does not match the stored trajectory avgQ!")
    except Exception as e:
        print(f"  Error calculating avgQ: {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Display formula representation from trajectory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Display formula from trajectory at specific slice
  %(prog)s --traj xyz789-abc123 42

  # Connect to different warehouse server
  %(prog)s --traj abc123 50 --host remote.server --port 8080
        """
    )
    
    # Trajectory input
    parser.add_argument("--traj", nargs=2, metavar=("TRAJ_ID", "SLICE"),
                       required=True,
                       help="Trajectory ID and slice index")
    
    # Warehouse configuration
    parser.add_argument("--host", type=str, default="localhost",
                       help="Warehouse host (default: localhost)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Warehouse port (default: 8000)")
    
    args = parser.parse_args()
    
    # Parse trajectory arguments
    traj_id, traj_slice = args.traj
    try:
        traj_slice = int(traj_slice)
    except ValueError:
        print(f"Error: Slice must be an integer, got '{traj_slice}'")
        sys.exit(1)
    
    # Initialize warehouse client
    warehouse = WarehouseClient(host=args.host, port=args.port)
    
    try:
        print(f"Retrieving formula from trajectory: {traj_id}, slice: {traj_slice}")
        
        # Get trajectory and reconstruct formula
        trajectory, gates, num_vars, width = get_trajectory_and_formula(
            warehouse, traj_id, traj_slice
        )
        
        if gates is None:
            print("Failed to retrieve trajectory or reconstruct formula")
            sys.exit(1)
            
        # Display the formula
        display_formula(trajectory, gates, num_vars, width, traj_slice)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        warehouse.close()


if __name__ == "__main__":
    main()