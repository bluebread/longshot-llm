import pytest
import torch
import tempfile
import json
import os
import sys
from unittest.mock import MagicMock, patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import (
    extract_avgq_from_trajectory,
    extract_avgq_batch,
    generate_sequences,
    analyze_avgq_improvement
)
from model import GPT2ForLongshot, GPT2ForLongshotConfig
from transformers import GPT2Config


@pytest.fixture
def sample_model():
    """Create a sample GPT2ForLongshot model for testing."""
    config = GPT2ForLongshotConfig(
        num_vars=3,
        width=2,
        n_embed_lit=8,
        ub_q=3.0,
        alpha=1.0,
        beta=20.0,
        gamma=1.0,
        share_semantic=False,
        universal=False,
        gpt2_config=GPT2Config(
            vocab_size=1,
            n_positions=64,
            n_embd=32,
            n_layer=2,
            n_head=2,
        )
    )
    model = GPT2ForLongshot(config)
    model.eval()
    return model


@pytest.fixture
def sample_trajectories():
    """Create sample trajectory data for testing."""
    return [
        [1, -2, 3, -4, 5],
        [10, -20, 30],
        [-100, 200, -300, 400, -500, 600]
    ]


class TestExtractAvgQ:
    """Test the extract_avgq_from_trajectory function."""

    def test_extract_avgq_single_trajectory(self, sample_model):
        """Test extracting avgQ from a single trajectory."""
        trajectory = [1, -2, 3, -4, 5]

        result = extract_avgq_from_trajectory(trajectory, sample_model, device="cpu")

        assert isinstance(result, float)
        assert result >= 0  # avgQ values should be non-negative

    def test_extract_avgq_empty_trajectory(self, sample_model):
        """Test that empty trajectory raises ValueError."""
        with pytest.raises(ValueError, match="Trajectory cannot be empty"):
            extract_avgq_from_trajectory([], sample_model)

    def test_extract_avgq_long_trajectory_warning(self, sample_model, capsys):
        """Test that long trajectories produce a warning."""
        # Use valid encoded gate tokens instead of sequential numbers
        # These are smaller values that won't cause overflow
        long_trajectory = [i % 100 - 50 for i in range(600)]

        try:
            _ = extract_avgq_from_trajectory(long_trajectory, sample_model)
        except RuntimeError:
            # It's ok if the model fails on very long sequences
            pass

        captured = capsys.readouterr()
        assert "Warning: Trajectory length 600 may exceed model capacity" in captured.out

    def test_extract_avgq_device_handling(self, sample_model):
        """Test that device parameter is properly handled."""
        trajectory = [1, -2, 3]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda":
            sample_model = sample_model.to(device)

        result = extract_avgq_from_trajectory(trajectory, sample_model, device=device)

        assert isinstance(result, float)

    def test_extract_avgq_error_handling(self, sample_model):
        """Test that extraction errors are properly caught and re-raised."""
        # Create a mock model that will raise an error
        mock_model = MagicMock()
        mock_model._decode_input_to_embedding.side_effect = RuntimeError("Test error")

        with pytest.raises(RuntimeError, match="Failed to extract avgQ from trajectory"):
            extract_avgq_from_trajectory([1, 2, 3], mock_model)


class TestExtractAvgQBatch:
    """Test the extract_avgq_batch function."""

    def test_batch_extraction(self, sample_model, sample_trajectories):
        """Test batch extraction of avgQ values."""
        results = extract_avgq_batch(sample_trajectories, sample_model, device="cpu")

        assert isinstance(results, list)
        assert len(results) == len(sample_trajectories)
        assert all(isinstance(r, float) for r in results)

    def test_batch_extraction_empty(self, sample_model):
        """Test batch extraction with empty list."""
        results = extract_avgq_batch([], sample_model)

        assert results == []

    def test_batch_extraction_consistency(self, sample_model):
        """Test that batch extraction matches individual extraction."""
        trajectories = [[1, -2, 3], [10, -20, 30]]

        batch_results = extract_avgq_batch(trajectories, sample_model)
        individual_results = [
            extract_avgq_from_trajectory(traj, sample_model)
            for traj in trajectories
        ]

        # Results should be close (accounting for floating point differences)
        for batch_val, individual_val in zip(batch_results, individual_results):
            assert abs(batch_val - individual_val) < 1e-5


class TestGenerateSequences:
    """Test the generate_sequences function."""

    def test_generate_sequences_basic(self, sample_model):
        """Test basic sequence generation."""
        prompt = torch.tensor([[1, -2]], dtype=torch.long)

        sequences, avgq_values = generate_sequences(
            sample_model,
            prompt,
            num_sequences=2,
            max_length=10,
            temperature=1.0,
            device="cpu"
        )

        assert sequences.shape[0] == 2  # num_sequences
        assert sequences.shape[1] <= 10  # max_length
        assert len(avgq_values) == 2
        assert all(isinstance(v, float) for v in avgq_values)

    def test_generate_sequences_parameters(self, sample_model):
        """Test sequence generation with different parameters."""
        prompt = torch.tensor([[1]], dtype=torch.long)

        # Test with different sampling parameters
        sequences, avgq_values = generate_sequences(
            sample_model,
            prompt,
            num_sequences=1,
            max_length=5,
            temperature=0.5,
            top_k=10,
            top_p=0.9,
            device="cpu"
        )

        assert sequences.shape[0] == 1
        assert sequences.shape[1] <= 5
        assert len(avgq_values) == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_generate_sequences_cuda(self, sample_model):
        """Test sequence generation on CUDA device."""
        sample_model = sample_model.to("cuda")
        prompt = torch.tensor([[1, -2]], dtype=torch.long)

        sequences, avgq_values = generate_sequences(
            sample_model,
            prompt,
            num_sequences=1,
            max_length=8,
            device="cuda"
        )

        assert sequences.device.type == "cuda"
        assert len(avgq_values) == 1


class TestAnalyzeAvgQImprovement:
    """Test the analyze_avgq_improvement function."""

    def test_analyze_improvement_basic(self):
        """Test basic improvement analysis."""
        prompt_avgq = [1.0, 2.0, 3.0]
        generated_avgq = [1.5, 2.5, 3.5, 4.0]

        analysis = analyze_avgq_improvement(prompt_avgq, generated_avgq)

        assert "prompt_mean" in analysis
        assert "generated_mean" in analysis
        assert "improvement_mean" in analysis
        assert "improvement_percentage" in analysis

        assert analysis["prompt_mean"] == 2.0
        assert analysis["generated_mean"] == 2.875
        assert analysis["improvement_count"] == 3

    def test_analyze_improvement_empty(self):
        """Test analysis with empty lists."""
        analysis = analyze_avgq_improvement([], [])
        assert analysis == {}

        analysis = analyze_avgq_improvement([1.0], [])
        assert analysis == {}

        analysis = analyze_avgq_improvement([], [1.0])
        assert analysis == {}

    def test_analyze_improvement_statistics(self):
        """Test statistical calculations in improvement analysis."""
        prompt_avgq = [1.0, 2.0, 3.0, 4.0, 5.0]
        generated_avgq = [2.0, 3.0, 4.0, 5.0, 6.0]

        analysis = analyze_avgq_improvement(prompt_avgq, generated_avgq)

        assert analysis["prompt_mean"] == 3.0
        assert analysis["generated_mean"] == 4.0
        assert analysis["improvement_mean"] == 1.0
        assert analysis["improvement_percentage"] == 100.0
        assert analysis["prompt_max"] == 5.0
        assert analysis["generated_max"] == 6.0

    def test_analyze_improvement_mixed_results(self):
        """Test analysis with mixed improvements and regressions."""
        prompt_avgq = [2.0, 3.0, 4.0]
        generated_avgq = [1.5, 3.5, 4.5]  # One worse, two better

        analysis = analyze_avgq_improvement(prompt_avgq, generated_avgq)

        assert analysis["improvement_count"] == 2
        assert abs(analysis["improvement_percentage"] - 66.67) < 0.1


class TestModelIntegration:
    """Integration tests with model save/load."""

    def test_model_save_load_evaluation(self, sample_model):
        """Test that saved and loaded models produce consistent evaluations."""
        trajectory = [1, -2, 3, -4, 5]

        # Get avgQ from original model
        original_avgq = extract_avgq_from_trajectory(trajectory, sample_model)

        # Save and reload model
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_model.save_pretrained(tmpdir)
            loaded_model = GPT2ForLongshot.from_pretrained(tmpdir)
            loaded_model.eval()

            # Get avgQ from loaded model
            loaded_avgq = extract_avgq_from_trajectory(trajectory, loaded_model)

            # Results should be identical
            assert abs(original_avgq - loaded_avgq) < 1e-6


class TestCommandLineInterface:
    """Test the command-line interface functionality."""

    @patch('eval.extract_avgq_from_trajectory')
    @patch('eval.TrajectoryDataset')
    @patch('eval.GPT2ForLongshot.from_pretrained')
    def test_main_function_mock(self, mock_from_pretrained, mock_dataset, mock_extract_avgq):
        """Test the main function with mocked components."""
        # Create mock model
        mock_model = MagicMock()
        mock_model.num_vars = 3
        mock_model.width = 2
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3]]))
        mock_from_pretrained.return_value = mock_model

        # Mock avgQ extraction to return a fixed value
        mock_extract_avgq.return_value = 2.5

        # Create mock dataset
        mock_dataset_instance = MagicMock()
        mock_dataset_instance.__len__ = MagicMock(return_value=10)
        mock_dataset_instance.__getitem__ = MagicMock(return_value={
            "input_ids": [1, -2, 3, -4, 5],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [1.0, 1.5, 2.0, 2.5, 3.0]
        })
        mock_dataset.return_value = mock_dataset_instance

        # Test with minimal arguments
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model")
            os.makedirs(model_path)

            # Create a mock config file
            config = {
                "num_vars": 3,
                "width": 2,
                "n_embed_lit": 8,
                "ub_q": 3.0,
                "alpha": 1.0,
                "beta": 20.0,
                "gamma": 1.0,
                "share_semantic": False,
                "universal": False,
                "gpt2_config": {
                    "vocab_size": 1,
                    "n_positions": 64,
                    "n_embd": 32,
                    "n_layer": 2,
                    "n_head": 2,
                }
            }
            with open(os.path.join(model_path, "config.json"), "w") as f:
                json.dump(config, f)

            # Run main with test arguments
            test_args = [
                "eval.py",
                "--model-path", model_path,
                "--num-prompts", "1",
                "--num-sequences", "1",
                "--prompt-length", "3",
                "--max-length", "5",
                "--seed", "42"
            ]

            with patch('sys.argv', test_args):
                from eval import main

                # Should run without errors
                try:
                    main()
                except SystemExit:
                    pass  # argparse may call sys.exit(0) on success


if __name__ == "__main__":
    pytest.main([__file__, "-v"])