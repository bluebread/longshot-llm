"""Test trajectory utility functions."""

import pytest
from longshot.utils.trajectory_utils import parse_trajectory_to_definition, reconstruct_formula_from_trajectory
from longshot.env.formula_graph import FormulaGraph


class TestTrajectoryUtils:
    """Test trajectory utility functions."""
    
    def test_parse_trajectory_to_definition_empty(self):
        """Test parsing empty trajectory."""
        trajectory = []
        definition = parse_trajectory_to_definition(trajectory)
        assert definition == []
    
    def test_parse_trajectory_to_definition_add_only(self):
        """Test parsing trajectory with only ADD operations."""
        trajectory = [
            (0, 5, 1.0),   # ADD gate 5
            (0, 10, 1.5),  # ADD gate 10
            (0, 15, 2.0),  # ADD gate 15
        ]
        definition = parse_trajectory_to_definition(trajectory)
        assert set(definition) == {5, 10, 15}
    
    def test_parse_trajectory_to_definition_add_del(self):
        """Test parsing trajectory with ADD and DEL operations."""
        trajectory = [
            (0, 5, 1.0),   # ADD gate 5
            (0, 10, 1.5),  # ADD gate 10
            (1, 5, 1.2),   # DEL gate 5
            (0, 15, 2.0),  # ADD gate 15
        ]
        definition = parse_trajectory_to_definition(trajectory)
        assert set(definition) == {10, 15}
    
    def test_parse_trajectory_to_definition_duplicate_add(self):
        """Test that duplicate ADDs don't create duplicates in definition."""
        trajectory = [
            (0, 5, 1.0),   # ADD gate 5
            (0, 5, 1.5),   # ADD gate 5 again (duplicate)
            (0, 10, 2.0),  # ADD gate 10
        ]
        definition = parse_trajectory_to_definition(trajectory)
        assert set(definition) == {5, 10}
        assert len([x for x in definition if x == 5]) == 1  # Only one instance of 5
    
    def test_parse_trajectory_to_definition_del_nonexistent(self):
        """Test that DEL of non-existent gate is safe."""
        trajectory = [
            (1, 5, 1.0),   # DEL gate 5 (doesn't exist)
            (0, 10, 1.5),  # ADD gate 10
        ]
        definition = parse_trajectory_to_definition(trajectory)
        assert set(definition) == {10}
    
    def test_parse_trajectory_to_definition_with_eos(self):
        """Test that EOS token stops processing."""
        trajectory = [
            (0, 5, 1.0),   # ADD gate 5
            (0, 10, 1.5),  # ADD gate 10
            (2, 0, 2.0),   # EOS
            (0, 15, 2.5),  # ADD gate 15 (should be ignored after EOS)
        ]
        definition = parse_trajectory_to_definition(trajectory)
        assert set(definition) == {5, 10}  # Gate 15 not included
    
    def test_reconstruct_formula_from_trajectory(self):
        """Test reconstructing FormulaGraph from trajectory."""
        trajectory = [
            (0, 5, 1.0),   # ADD gate 5
            (0, 10, 1.5),  # ADD gate 10
            (0, 15, 2.0),  # ADD gate 15
        ]
        num_vars = 4
        
        formula_graph = reconstruct_formula_from_trajectory(trajectory, num_vars)
        
        assert isinstance(formula_graph, FormulaGraph)
        assert set(formula_graph.gates) == {5, 10, 15}
    
    def test_reconstruct_formula_with_operations(self):
        """Test reconstructing FormulaGraph with mixed operations."""
        trajectory = [
            (0, 5, 1.0),   # ADD gate 5
            (0, 10, 1.5),  # ADD gate 10
            (1, 5, 1.2),   # DEL gate 5
            (0, 15, 2.0),  # ADD gate 15
        ]
        num_vars = 4
        
        formula_graph = reconstruct_formula_from_trajectory(trajectory, num_vars)
        
        assert isinstance(formula_graph, FormulaGraph)
        assert set(formula_graph.gates) == {10, 15}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])