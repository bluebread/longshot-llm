#!/usr/bin/env python3
"""
Unit tests for the trajectory data cleaning script.
"""

import json
import logging
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

# Import data_cleaning module from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data_cleaning


class TestDataCleaningFunctions(unittest.TestCase):
    """Test the core data cleaning functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = logging.getLogger('test')
        self.logger.setLevel(logging.CRITICAL)  # Suppress output during tests
        
        # Sample trajectory data
        self.sample_trajectory = {
            "traj_id": "test-traj-123",
            "num_vars": 3,
            "width": 2,
            "timestamp": "2025-01-01T12:00:00Z",
            "steps": [
                [0, 5, 1.0],   # ADD gate 5, avgQ 1.0
                [1, 3, 1.5],   # DEL gate 3, avgQ 1.5
                [0, 7, 2.5],   # ADD gate 7, avgQ 2.5 (max)
                [1, 5, 1.8]    # DEL gate 5, avgQ 1.8
            ]
        }
        
        # Sample trajectories list
        self.sample_trajectories = [
            {
                "traj_id": "traj-1",
                "steps": [[0, 1, 0.5], [1, 2, 1.0], [0, 3, 0.8]]
            },
            {
                "traj_id": "traj-2", 
                "steps": [[0, 4, 2.0], [1, 5, 3.0]]  # max 3.0
            },
            {
                "traj_id": "traj-3",
                "steps": [[0, 6, 0.3], [1, 7, 0.4]]  # max 0.4
            }
        ]
    
    def test_get_max_avgq(self):
        """Test get_max_avgq function."""
        # Normal case
        steps = [[0, 5, 1.0], [1, 3, 2.5], [0, 7, 1.8]]
        result = data_cleaning.get_max_avgq(steps)
        self.assertEqual(result, 2.5)
        
        # Empty steps
        result = data_cleaning.get_max_avgq([])
        self.assertEqual(result, 0.0)
        
        # Single step
        result = data_cleaning.get_max_avgq([[0, 1, 1.5]])
        self.assertEqual(result, 1.5)
        
        # All same values
        result = data_cleaning.get_max_avgq([[0, 1, 2.0], [1, 2, 2.0]])
        self.assertEqual(result, 2.0)
    
    def test_calculate_max_avgq_in_dataset(self):
        """Test calculate_max_avgq_in_dataset function."""
        # Normal case with multiple trajectories
        trajectories = [
            {"traj_id": "1", "steps": [[0, 1, 1.0], [1, 2, 2.5], [0, 3, 1.8]]},  # max 2.5
            {"traj_id": "2", "steps": [[0, 4, 3.2], [1, 5, 1.0]]},              # max 3.2
            {"traj_id": "3", "steps": [[0, 6, 0.5], [1, 7, 0.8]]}               # max 0.8
        ]
        result = data_cleaning.calculate_max_avgq_in_dataset(trajectories)
        self.assertEqual(result, 3.2)
        
        # Empty trajectories list
        result = data_cleaning.calculate_max_avgq_in_dataset([])
        self.assertEqual(result, 0.0)
        
        # Trajectories with empty steps
        trajectories = [
            {"traj_id": "1", "steps": []},
            {"traj_id": "2", "steps": []}
        ]
        result = data_cleaning.calculate_max_avgq_in_dataset(trajectories)
        self.assertEqual(result, 0.0)
        
        # Mix of empty and non-empty trajectories
        trajectories = [
            {"traj_id": "1", "steps": []},
            {"traj_id": "2", "steps": [[0, 1, 2.7]]},
            {"traj_id": "3", "steps": []}
        ]
        result = data_cleaning.calculate_max_avgq_in_dataset(trajectories)
        self.assertEqual(result, 2.7)
        
        # Single trajectory
        trajectories = [
            {"traj_id": "1", "steps": [[0, 1, 1.5], [1, 2, 0.8]]}
        ]
        result = data_cleaning.calculate_max_avgq_in_dataset(trajectories)
        self.assertEqual(result, 1.5)
    
    def test_validate_trajectory_format(self):
        """Test trajectory format validation."""
        # Valid trajectory
        valid_traj = {
            "traj_id": "test-123",
            "steps": [[0, 5, 1.0], [1, 3, 2.5]]
        }
        self.assertTrue(data_cleaning.validate_trajectory_format(valid_traj, self.logger))
        
        # Missing traj_id
        invalid_traj = {"steps": [[0, 5, 1.0]]}
        self.assertFalse(data_cleaning.validate_trajectory_format(invalid_traj, self.logger))
        
        # Missing steps
        invalid_traj = {"traj_id": "test"}
        self.assertFalse(data_cleaning.validate_trajectory_format(invalid_traj, self.logger))
        
        # Steps not a list
        invalid_traj = {"traj_id": "test", "steps": "not a list"}
        self.assertFalse(data_cleaning.validate_trajectory_format(invalid_traj, self.logger))
        
        # Invalid step format - not tuple/list
        invalid_traj = {"traj_id": "test", "steps": ["not a tuple"]}
        self.assertFalse(data_cleaning.validate_trajectory_format(invalid_traj, self.logger))
        
        # Invalid step format - wrong length
        invalid_traj = {"traj_id": "test", "steps": [[0, 5]]}
        self.assertFalse(data_cleaning.validate_trajectory_format(invalid_traj, self.logger))
        
        # Invalid step format - wrong types
        invalid_traj = {"traj_id": "test", "steps": [["not_int", 5, 1.0]]}
        self.assertFalse(data_cleaning.validate_trajectory_format(invalid_traj, self.logger))
        
        invalid_traj = {"traj_id": "test", "steps": [[0, "not_int", 1.0]]}
        self.assertFalse(data_cleaning.validate_trajectory_format(invalid_traj, self.logger))
        
        invalid_traj = {"traj_id": "test", "steps": [[0, 5, "not_number"]]}
        self.assertFalse(data_cleaning.validate_trajectory_format(invalid_traj, self.logger))
        
        # Valid with different numeric types
        valid_traj = {"traj_id": "test", "steps": [[0, 5, 1], [1, 3, 2.5]]}
        self.assertTrue(data_cleaning.validate_trajectory_format(valid_traj, self.logger))
    
    def test_truncate_trajectory_to_max_avgq(self):
        """Test trajectory truncation to max avgQ."""
        # Normal case - max in middle
        steps = [[0, 1, 1.0], [1, 2, 2.5], [0, 3, 1.8], [1, 4, 1.0]]
        result = data_cleaning.truncate_trajectory_to_max_avgq(steps)
        expected = [[0, 1, 1.0], [1, 2, 2.5]]
        self.assertEqual(result, expected)
        
        # Max at end
        steps = [[0, 1, 1.0], [1, 2, 1.5], [0, 3, 2.5]]
        result = data_cleaning.truncate_trajectory_to_max_avgq(steps)
        self.assertEqual(result, steps)  # Should be unchanged
        
        # Max at beginning
        steps = [[0, 1, 2.5], [1, 2, 1.0], [0, 3, 1.8]]
        result = data_cleaning.truncate_trajectory_to_max_avgq(steps)
        expected = [[0, 1, 2.5]]
        self.assertEqual(result, expected)
        
        # Multiple max values - should keep to last one
        steps = [[0, 1, 1.0], [1, 2, 2.5], [0, 3, 1.8], [1, 4, 2.5], [0, 5, 1.0]]
        result = data_cleaning.truncate_trajectory_to_max_avgq(steps)
        expected = [[0, 1, 1.0], [1, 2, 2.5], [0, 3, 1.8], [1, 4, 2.5]]
        self.assertEqual(result, expected)
        
        # Empty steps
        result = data_cleaning.truncate_trajectory_to_max_avgq([])
        self.assertEqual(result, [])
        
        # Single step
        steps = [[0, 1, 1.5]]
        result = data_cleaning.truncate_trajectory_to_max_avgq(steps)
        self.assertEqual(result, steps)
        
        # Test float precision handling
        steps = [[0, 1, 1.0], [1, 2, 2.5], [0, 3, 2.5000000001]]
        result = data_cleaning.truncate_trajectory_to_max_avgq(steps)
        # Should treat both values as approximately equal and truncate to the last one
        expected = [[0, 1, 1.0], [1, 2, 2.5], [0, 3, 2.5000000001]]
        self.assertEqual(result, expected)
    
    def test_filter_trajectories_by_threshold(self):
        """Test trajectory filtering by avgQ threshold."""
        trajectories = [
            {"traj_id": "1", "steps": [[0, 1, 0.5], [1, 2, 1.0]]},  # max 1.0
            {"traj_id": "2", "steps": [[0, 3, 2.0], [1, 4, 3.0]]},  # max 3.0
            {"traj_id": "3", "steps": [[0, 5, 0.3], [1, 6, 0.4]]}   # max 0.4
        ]
        
        # Filter with threshold 1.0 - should keep trajectories 1 and 2
        result = data_cleaning.filter_trajectories_by_threshold(trajectories, 1.0, self.logger)
        self.assertEqual(len(result), 2)
        self.assertIn(trajectories[0], result)
        self.assertIn(trajectories[1], result)
        
        # Filter with threshold 2.0 - should keep only trajectory 2
        result = data_cleaning.filter_trajectories_by_threshold(trajectories, 2.0, self.logger)
        self.assertEqual(len(result), 1)
        self.assertIn(trajectories[1], result)
        
        # Filter with threshold 5.0 - should keep none
        result = data_cleaning.filter_trajectories_by_threshold(trajectories, 5.0, self.logger)
        self.assertEqual(len(result), 0)
        
        # Empty input
        result = data_cleaning.filter_trajectories_by_threshold([], 1.0, self.logger)
        self.assertEqual(len(result), 0)
    
    def test_process_trajectories(self):
        """Test trajectory processing (truncation)."""
        trajectories = [
            {
                "traj_id": "1",
                "num_vars": 3,
                "width": 2,
                "steps": [[0, 1, 1.0], [1, 2, 2.0], [0, 3, 1.5]]
            },
            {
                "traj_id": "2", 
                "num_vars": 4,
                "width": 3,
                "steps": [[0, 4, 0.5], [1, 5, 1.0]]
            }
        ]
        
        result = data_cleaning.process_trajectories(trajectories, self.logger)
        
        # Should have same number of trajectories
        self.assertEqual(len(result), 2)
        
        # First trajectory should be truncated to max avgQ step (2.0)
        self.assertEqual(result[0]["traj_id"], "1")
        expected_steps = [[0, 1, 1.0], [1, 2, 2.0]]
        self.assertEqual(result[0]["steps"], expected_steps)
        # num_vars and width should not be in individual trajectories (only in metadata)
        self.assertNotIn("num_vars", result[0])
        self.assertNotIn("width", result[0])
        
        # Second trajectory should be unchanged (max at end)
        self.assertEqual(result[1]["traj_id"], "2")
        self.assertEqual(result[1]["steps"], trajectories[1]["steps"])
        # num_vars and width should not be in individual trajectories (only in metadata)
        self.assertNotIn("num_vars", result[1])
        self.assertNotIn("width", result[1])


class TestDataCleaningIntegration(unittest.TestCase):
    """Integration tests for the data cleaning script."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.temp_dir, "test_output.json")
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('data_cleaning.WarehouseClient')
    def test_download_trajectories_success(self, mock_warehouse_client):
        """Test successful trajectory download."""
        # Mock the warehouse client
        mock_client_instance = MagicMock()
        mock_warehouse_client.return_value.__enter__.return_value = mock_client_instance
        
        # Mock the dataset response
        mock_dataset = {
            "trajectories": [
                {"traj_id": "test-1", "steps": [[0, 1, 1.0]]},
                {"traj_id": "test-2", "steps": [[1, 2, 2.0]]}
            ]
        }
        mock_client_instance.get_trajectory_dataset.return_value = mock_dataset
        
        logger = logging.getLogger('test')
        logger.setLevel(logging.CRITICAL)
        
        result = data_cleaning.download_trajectories(
            host="localhost",
            port=8000,
            num_vars=3,
            width=2,
            timeout=30.0,
            logger=logger
        )
        
        # Verify the result
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["traj_id"], "test-1")
        self.assertEqual(result[1]["traj_id"], "test-2")
        
        # Verify the client was called correctly
        mock_warehouse_client.assert_called_once_with("localhost", 8000)
        mock_client_instance.get_trajectory_dataset.assert_called_once_with(
            num_vars=3, width=2
        )
    
    def test_save_dataset_to_file(self):
        """Test saving dataset to JSON file."""
        dataset = {
            "metadata": {"num_vars": 3, "width": 2},
            "trajectories": [{"traj_id": "test", "steps": [[0, 1, 1.0]]}]
        }
        
        logger = logging.getLogger('test')
        logger.setLevel(logging.CRITICAL)
        
        data_cleaning.save_dataset_to_file(dataset, self.output_file, False, logger)
        
        # Verify file was created and contains correct data
        self.assertTrue(os.path.exists(self.output_file))
        
        with open(self.output_file, 'r') as f:
            saved_data = json.load(f)
        
        self.assertEqual(saved_data, dataset)
    
    def test_save_dataset_to_file_compact(self):
        """Test saving dataset in compact format."""
        dataset = {
            "metadata": {"num_vars": 3, "width": 2},
            "trajectories": [{"traj_id": "test", "steps": [[0, 1, 1.0]]}]
        }
        
        logger = logging.getLogger('test')
        logger.setLevel(logging.CRITICAL)
        
        # Test compact format
        compact_file = os.path.join(self.temp_dir, "compact_output.json")
        data_cleaning.save_dataset_to_file(dataset, compact_file, True, logger)
        
        # Test non-compact format
        regular_file = os.path.join(self.temp_dir, "regular_output.json")
        data_cleaning.save_dataset_to_file(dataset, regular_file, False, logger)
        
        # Read both files as text to compare formatting
        with open(compact_file, 'r') as f:
            compact_content = f.read()
        
        with open(regular_file, 'r') as f:
            regular_content = f.read()
        
        # Compact format should have no unnecessary whitespace
        self.assertNotIn('  ', compact_content)  # No double spaces
        self.assertNotIn('\n', compact_content)  # No newlines
        
        # Regular format should have indentation
        self.assertIn('  ', regular_content)  # Has indentation
        self.assertIn('\n', regular_content)  # Has newlines
        
        # Both should contain same data when parsed
        with open(compact_file, 'r') as f:
            compact_data = json.load(f)
        
        self.assertEqual(compact_data, dataset)
    
    def test_create_output_dataset_with_max_avgq(self):
        """Test that create_output_dataset includes max_avgq in metadata."""
        trajectories = [
            {"traj_id": "1", "steps": [[0, 1, 1.5], [1, 2, 2.8]]},  # max 2.8
            {"traj_id": "2", "steps": [[0, 3, 3.5], [1, 4, 1.0]]},  # max 3.5
            {"traj_id": "3", "steps": [[0, 5, 0.7]]}                # max 0.7
        ]
        
        dataset = data_cleaning.create_output_dataset(
            trajectories=trajectories,
            num_vars=4,
            width=3,
            threshold=1.0,
            host="localhost",
            port=8000,
            original_count=100,
            filtered_count=50
        )
        
        # Verify metadata structure
        self.assertIn("metadata", dataset)
        self.assertIn("trajectories", dataset)
        
        metadata = dataset["metadata"]
        self.assertEqual(metadata["num_vars"], 4)
        self.assertEqual(metadata["width"], 3)
        self.assertEqual(metadata["threshold"], 1.0)
        self.assertEqual(metadata["warehouse_host"], "localhost")
        self.assertEqual(metadata["warehouse_port"], 8000)
        self.assertEqual(metadata["downloaded_count"], 100)
        self.assertEqual(metadata["filtered_count"], 50)
        self.assertEqual(metadata["removed_count"], 50)
        self.assertEqual(metadata["processed_count"], 3)
        
        # Verify max_avgq is calculated correctly
        self.assertIn("max_avgq", metadata)
        self.assertEqual(metadata["max_avgq"], 3.5)
        
        # Verify trajectory data
        self.assertEqual(len(dataset["trajectories"]), 3)
        self.assertEqual(dataset["trajectories"], trajectories)
    
    def test_create_output_dataset_empty_trajectories(self):
        """Test create_output_dataset with empty trajectories list."""
        dataset = data_cleaning.create_output_dataset(
            trajectories=[],
            num_vars=3,
            width=2,
            threshold=0.5,
            host="localhost",
            port=8000,
            original_count=10,
            filtered_count=0
        )
        
        metadata = dataset["metadata"]
        self.assertEqual(metadata["max_avgq"], 0.0)
        self.assertEqual(metadata["processed_count"], 0)
        self.assertEqual(len(dataset["trajectories"]), 0)


class TestArgumentHandling(unittest.TestCase):
    """Test argument parsing and validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_validate_arguments_with_output(self):
        """Test argument validation with --output specified."""
        import argparse
        
        # Create a valid output file path
        output_file = os.path.join(self.temp_dir, "test_output.json")
        
        args = argparse.Namespace(
            num_vars=3,
            width=2,
            port=8000,
            timeout=30.0,
            output=output_file,
            output_dir=None
        )
        
        # Should not raise any exception
        data_cleaning.validate_arguments(args)
    
    def test_validate_arguments_with_output_dir(self):
        """Test argument validation with --output-dir specified."""
        import argparse
        
        args = argparse.Namespace(
            num_vars=3,
            width=2,
            port=8000,
            timeout=30.0,
            output=None,
            output_dir=self.temp_dir
        )
        
        # Should not raise any exception
        data_cleaning.validate_arguments(args)
    
    def test_validate_arguments_missing_both(self):
        """Test argument validation with neither --output nor --output-dir specified."""
        import argparse
        
        args = argparse.Namespace(
            num_vars=3,
            width=2,
            port=8000,
            timeout=30.0,
            output=None,
            output_dir=None
        )
        
        # Should raise ValueError
        with self.assertRaises(ValueError) as context:
            data_cleaning.validate_arguments(args)
        
        self.assertIn("Either --output or --output-dir must be specified", str(context.exception))
    
    def test_validate_arguments_invalid_num_vars(self):
        """Test argument validation with invalid num_vars."""
        import argparse
        
        args = argparse.Namespace(
            num_vars=0,
            width=2,
            port=8000,
            timeout=30.0,
            output=None,
            output_dir=self.temp_dir
        )
        
        with self.assertRaises(ValueError) as context:
            data_cleaning.validate_arguments(args)
        
        self.assertIn("num_vars must be positive", str(context.exception))
    
    def test_get_output_file_path_with_output(self):
        """Test get_output_file_path when --output is specified."""
        import argparse
        
        output_file = os.path.join(self.temp_dir, "custom_output.json")
        args = argparse.Namespace(
            output=output_file,
            output_dir=None,
            num_vars=3,
            width=2
        )
        
        result = data_cleaning.get_output_file_path(args)
        self.assertEqual(result, output_file)
    
    def test_get_output_file_path_with_output_dir(self):
        """Test get_output_file_path when --output-dir is specified."""
        import argparse
        
        args = argparse.Namespace(
            output=None,
            output_dir=self.temp_dir,
            num_vars=4,
            width=3
        )
        
        result = data_cleaning.get_output_file_path(args)
        expected = os.path.join(self.temp_dir, "n4w3.json")
        self.assertEqual(result, expected)
    
    def test_get_output_file_path_filename_format(self):
        """Test that generated filenames follow the correct format."""
        import argparse
        
        test_cases = [
            (3, 2, "n3w2.json"),
            (10, 5, "n10w5.json"),
            (1, 1, "n1w1.json")
        ]
        
        for num_vars, width, expected_filename in test_cases:
            args = argparse.Namespace(
                output=None,
                output_dir=self.temp_dir,
                num_vars=num_vars,
                width=width
            )
            
            result = data_cleaning.get_output_file_path(args)
            expected_path = os.path.join(self.temp_dir, expected_filename)
            self.assertEqual(result, expected_path)


if __name__ == "__main__":
    # Set up test logging to reduce noise
    logging.basicConfig(level=logging.CRITICAL)
    
    # Run the tests
    unittest.main(verbosity=2)