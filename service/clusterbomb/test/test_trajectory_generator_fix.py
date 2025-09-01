#!/usr/bin/env python3
"""
Quick test to verify trajectory generator fixes work correctly.
"""

import sys
import os
# Add parent directory to path to import clusterbomb modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trajectory_generator import run_mutations_sync
from longshot.formula import FormulaRewardModel, FormulaType
from longshot.utils import parse_formula_definition


def test_trajectory_generation():
    """Test that trajectory generation works with the fixes"""
    
    # Test parameters
    num_vars = 4
    width = 3
    
    # Generate trajectories from empty formula
    trajectories = run_mutations_sync(
        num_vars=num_vars,
        width=width,
        num_trajectories=2,
        steps_per_trajectory=10,
        prefix_traj=[],
        early_stop=True,
        seed=42
    )
    
    print(f"Generated {len(trajectories)} trajectories")
    
    for i, traj in enumerate(trajectories):
        print(f"\nTrajectory {i+1}:")
        print(f"  ID: {traj['traj_id']}")
        print(f"  Steps: {len(traj['steps'])}")
        print(f"  First 3 steps:")
        for j, step in enumerate(traj['steps'][:3]):
            token_type, litint, avgQ = step
            token_name = "ADD" if token_type == 0 else "DELETE"
            print(f"    Step {j+1}: {token_name} gate {litint}, avgQ={avgQ:.4f}")
    
    # Test with prefix trajectory
    prefix = [(0, 5, 0.5), (0, 12, 0.6)]
    trajectories_with_prefix = run_mutations_sync(
        num_vars=num_vars,
        width=width,
        num_trajectories=1,
        steps_per_trajectory=5,
        prefix_traj=prefix,
        early_stop=True,
        seed=43
    )
    
    print(f"\nGenerated {len(trajectories_with_prefix)} trajectories with prefix")
    print(f"Prefix length: {trajectories_with_prefix[0].get('prefix_length', 0)}")
    
    # Test that game.gates property works
    formula = parse_formula_definition([], num_vars, FormulaType.Conjunctive)
    game = FormulaRewardModel(formula, width=width)
    
    print(f"\nInitial gates in game: {game.gates}")
    assert isinstance(game.gates, set), "gates property should return a set"
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    test_trajectory_generation()