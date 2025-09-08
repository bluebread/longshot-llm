#!/usr/bin/env python3
"""
Unit tests for TrajectoryDataset with local file support.
"""

import unittest
import json
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset import TrajectoryDataset


class TestTrajectoryDataset(unittest.TestCase):
    """Test cases for TrajectoryDataset with local file support."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = {
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
                        [1, 3, 2.0]
                    ]
                },
                {
                    "traj_id": "test-2", 
                    "steps": [
                        [0, 12884901888, 2.25],
                        [1, 8589934596, 2.0],
                        [0, 8589934596, 2.25]
                    ]
                }
            ]
        }
    
    def test_load_from_local_file(self):
        """Test loading dataset from a local JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_path = f.name
        
        try:
            # Load dataset from local file
            dataset = TrajectoryDataset(local_file=temp_path)
            
            # Check dataset properties
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.num_vars, 3)
            self.assertEqual(dataset.width, 2)
            
            # Check first sample
            sample = dataset[0]
            self.assertIn('input_ids', sample)
            self.assertIn('attention_mask', sample)
            self.assertIn('labels', sample)
            
            # Verify data transformation
            expected_input_ids = [4294967300, 6, -3]  # [0,l1], [0,l2], [1,-l3]
            self.assertEqual(sample['input_ids'], expected_input_ids)
            self.assertEqual(sample['labels'], [1.5, 1.75, 2.0])
            self.assertEqual(len(sample['attention_mask']), 3)
            
        finally:
            os.unlink(temp_path)
    
    def test_load_from_local_file_without_metadata(self):
        """Test loading dataset from file without metadata."""
        # Create data without metadata - just trajectories
        test_data_no_metadata = {
            "trajectories": [
                {
                    "steps": [
                        [0, 100, 1.0],
                        [1, 200, 1.5]
                    ]
                }
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data_no_metadata, f)
            temp_path = f.name
        
        try:
            dataset = TrajectoryDataset(local_file=temp_path)
            self.assertEqual(len(dataset), 1)
            
            sample = dataset[0]
            self.assertEqual(sample['input_ids'], [100, -200])
            self.assertEqual(sample['labels'], [1.0, 1.5])
            
        finally:
            os.unlink(temp_path)
    
    def test_direct_steps_format(self):
        """Test loading dataset with direct steps format."""
        # Direct list of trajectories (steps)
        test_data_direct = [
            [[0, 10, 0.5], [1, 20, 1.0]],
            [[0, 30, 1.5], [1, 40, 2.0]]
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(test_data_direct, f)
            temp_path = f.name
        
        try:
            dataset = TrajectoryDataset(local_file=temp_path)
            self.assertEqual(len(dataset), 2)
            
            # Check first trajectory
            sample1 = dataset[0]
            self.assertEqual(sample1['input_ids'], [10, -20])
            self.assertEqual(sample1['labels'], [0.5, 1.0])
            
            # Check second trajectory
            sample2 = dataset[1]
            self.assertEqual(sample2['input_ids'], [30, -40])
            self.assertEqual(sample2['labels'], [1.5, 2.0])
            
        finally:
            os.unlink(temp_path)
    
    def test_local_file_not_found(self):
        """Test error handling for non-existent local file."""
        with self.assertRaises(FileNotFoundError):
            TrajectoryDataset(local_file="nonexistent_file.json")
    
    def test_missing_parameters_for_warehouse(self):
        """Test that num_vars and width are required when not using local_file."""
        with self.assertRaises(ValueError) as context:
            TrajectoryDataset()  # No local_file, num_vars, or width
        self.assertIn("num_vars and width are required", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            TrajectoryDataset(num_vars=3)  # Missing width
        self.assertIn("num_vars and width are required", str(context.exception))
    
    def test_override_metadata_values(self):
        """Test that provided num_vars/width can override metadata values."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.test_data, f)
            temp_path = f.name
        
        try:
            # Load with override values
            dataset = TrajectoryDataset(local_file=temp_path, num_vars=5, width=4)
            
            # Should use metadata values from file
            self.assertEqual(dataset.num_vars, 3)  # From metadata
            self.assertEqual(dataset.width, 2)     # From metadata
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    unittest.main()