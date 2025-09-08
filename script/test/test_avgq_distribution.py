#!/usr/bin/env python3
"""
Unit tests for avgQ distribution analysis script.
"""

import unittest
import json
import tempfile
from pathlib import Path
from collections import Counter
import sys

# Add script directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "script"))

from avgq_distribution import (
    validate_trajectory_data,
    extract_num_vars_width,
    calculate_avgq_distributions,
    calculate_bar_width,
    load_trajectories_from_file
)


class TestAvgQDistribution(unittest.TestCase):
    """Test cases for avgQ distribution analysis."""
    
    def setUp(self):
        """Set up test data."""
        self.valid_data = {
            "metadata": {
                "num_vars": 3,
                "width": 2,
                "threshold": 2.25
            },
            "trajectories": [
                {
                    "traj_id": "test-1",
                    "steps": [
                        [0, 4294967300, 1.5],
                        [0, 6, 1.75],
                        [0, 3, 2.0]
                    ]
                },
                {
                    "traj_id": "test-2",
                    "steps": [
                        [0, 12884901888, 2.25],
                        [0, 8589934596, 2.0],
                        [1, 8589934596, 2.25]
                    ]
                }
            ]
        }
    
    def test_validate_trajectory_data_valid(self):
        """Test validation with valid data."""
        # Should not raise any exception
        validate_trajectory_data(self.valid_data)
    
    def test_validate_trajectory_data_invalid_type(self):
        """Test validation with invalid data type."""
        with self.assertRaises(ValueError) as context:
            validate_trajectory_data([])
        self.assertIn("Data must be a dictionary", str(context.exception))
    
    def test_validate_trajectory_data_no_trajectories(self):
        """Test validation with no trajectories."""
        data = {"metadata": {}}
        with self.assertRaises(ValueError) as context:
            validate_trajectory_data(data)
        self.assertIn("No trajectories found", str(context.exception))
    
    def test_validate_trajectory_data_invalid_trajectory_structure(self):
        """Test validation with invalid trajectory structure."""
        data = {
            "trajectories": [
                {"invalid": "structure"}  # Missing "steps"
            ]
        }
        with self.assertRaises(ValueError) as context:
            validate_trajectory_data(data)
        self.assertIn("Invalid trajectory structure", str(context.exception))
    
    def test_validate_trajectory_data_invalid_step_format(self):
        """Test validation with invalid step format."""
        data = {
            "trajectories": [
                {
                    "traj_id": "test",
                    "steps": [
                        [0, 1]  # Missing avgQ value
                    ]
                }
            ]
        }
        with self.assertRaises(ValueError) as context:
            validate_trajectory_data(data)
        self.assertIn("Invalid step format", str(context.exception))
    
    def test_validate_trajectory_data_invalid_avgq_value(self):
        """Test validation with non-numeric avgQ value."""
        data = {
            "trajectories": [
                {
                    "traj_id": "test",
                    "steps": [
                        [0, 1, "invalid"]  # Non-numeric avgQ
                    ]
                }
            ]
        }
        with self.assertRaises(ValueError) as context:
            validate_trajectory_data(data)
        self.assertIn("Invalid avgQ value", str(context.exception))
    
    def test_extract_num_vars_width(self):
        """Test extraction of num_vars and width from metadata."""
        num_vars, width = extract_num_vars_width(self.valid_data)
        self.assertEqual(num_vars, 3)
        self.assertEqual(width, 2)
    
    def test_extract_num_vars_width_missing(self):
        """Test extraction when metadata is missing."""
        data = {"trajectories": []}
        num_vars, width = extract_num_vars_width(data)
        self.assertIsNone(num_vars)
        self.assertIsNone(width)
    
    def test_calculate_avgq_distributions(self):
        """Test calculation of avgQ distributions."""
        traj_dist, step_dist = calculate_avgq_distributions(self.valid_data)
        
        # Check trajectory distribution (uses max avgQ per trajectory)
        self.assertEqual(len(traj_dist), 2)  # Two trajectories
        self.assertEqual(traj_dist[2.0], 1)  # First trajectory max is 2.0
        self.assertEqual(traj_dist[2.25], 1)  # Second trajectory max is 2.25
        
        # Check step distribution
        self.assertEqual(sum(step_dist.values()), 6)  # Total 6 steps
        self.assertEqual(step_dist[1.5], 1)
        self.assertEqual(step_dist[1.75], 1)
        self.assertEqual(step_dist[2.0], 2)
        self.assertEqual(step_dist[2.25], 2)
    
    def test_calculate_avgq_distributions_empty(self):
        """Test calculation with empty trajectories."""
        data = {"trajectories": []}
        traj_dist, step_dist = calculate_avgq_distributions(data)
        
        self.assertEqual(len(traj_dist), 0)
        self.assertEqual(len(step_dist), 0)
    
    def test_calculate_bar_width(self):
        """Test bar width calculation."""
        # Single value
        width = calculate_bar_width([1.0])
        self.assertEqual(width, 0.01)  # Default width
        
        # Multiple values
        values = [1.0, 1.5, 2.0, 2.5, 3.0]
        width = calculate_bar_width(values)
        expected = (3.0 - 1.0) / (5 * 10)
        self.assertEqual(width, expected)
        
        # Very small range
        values = [1.0, 1.0001]
        width = calculate_bar_width(values)
        self.assertGreaterEqual(width, 0.001)  # Minimum width
    
    def test_load_trajectories_from_file(self):
        """Test loading trajectories from a JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.valid_data, f)
            temp_path = Path(f.name)
        
        try:
            data = load_trajectories_from_file(temp_path)
            self.assertEqual(data["metadata"]["num_vars"], 3)
            self.assertEqual(len(data["trajectories"]), 2)
        finally:
            temp_path.unlink()
    
    def test_load_trajectories_from_file_not_found(self):
        """Test loading from non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_trajectories_from_file(Path("nonexistent.json"))
    
    def test_avgq_values_preserved(self):
        """Test that avgQ values are preserved without rounding."""
        data = {
            "trajectories": [
                {
                    "traj_id": "test",
                    "steps": [
                        [0, 1, 1.123456789],
                        [0, 2, 2.987654321]
                    ]
                }
            ]
        }
        
        traj_dist, step_dist = calculate_avgq_distributions(data)
        
        # Check that values are preserved exactly
        avgq_values = list(step_dist.keys())
        self.assertIn(1.123456789, avgq_values)
        self.assertIn(2.987654321, avgq_values)
        
        # Check that max value is used for trajectory
        self.assertIn(2.987654321, traj_dist.keys())


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def test_end_to_end_with_file(self):
        """Test complete workflow with file input."""
        # Create test data file
        test_data = {
            "metadata": {
                "num_vars": 2,
                "width": 3
            },
            "trajectories": [
                {
                    "traj_id": f"traj-{i}",
                    "steps": [
                        [0, 100 + i, 1.0 + i * 0.25],
                        [1, 200 + i, 1.5 + i * 0.25],
                        [0, 300 + i, 2.0 + i * 0.25]
                    ]
                }
                for i in range(10)
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data, f)
            temp_path = Path(f.name)
        
        try:
            # Load and validate
            data = load_trajectories_from_file(temp_path)
            validate_trajectory_data(data)
            
            # Extract parameters
            num_vars, width = extract_num_vars_width(data)
            self.assertEqual(num_vars, 2)
            self.assertEqual(width, 3)
            
            # Calculate distributions
            traj_dist, step_dist = calculate_avgq_distributions(data)
            
            # Verify results
            self.assertEqual(sum(traj_dist.values()), 10)  # 10 trajectories
            self.assertEqual(sum(step_dist.values()), 30)  # 30 steps total
            
        finally:
            temp_path.unlink()


if __name__ == "__main__":
    unittest.main()