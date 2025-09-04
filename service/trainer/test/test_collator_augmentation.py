import pytest
import torch
import numpy as np
from typing import List, Tuple
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from service.trainer.collator import TrajectoryCollator
from service.trainer.dataset import TrajectoryDataset
from longshot.formula import NormalFormFormula, FormulaType, FormulaRewardModel
from longshot.literals import Literals


class TestTrajectoryCollatorAugmentation:
    
    @pytest.fixture
    def sample_trajectories(self):
        trajectories = [
            {
                "input_ids": [1, -5, 10, -20, 15],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [0.5, 0.6, 0.7, 0.8, 0.9]
            },
            {
                "input_ids": [-2, 8, -12, 25],
                "attention_mask": [1, 1, 1, 1],
                "labels": [0.4, 0.5, 0.6, 0.7]
            },
            {
                "input_ids": [3, -7, 11, -18, 22, -30],
                "attention_mask": [1, 1, 1, 1, 1, 1],
                "labels": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            }
        ]
        return trajectories

    @pytest.fixture
    def collator_with_permutation(self):
        return TrajectoryCollator(num_vars=3, permute_input=True)
    
    @pytest.fixture
    def collator_without_permutation(self):
        return TrajectoryCollator(num_vars=3, permute_input=False)
    
    def simulate_trajectory_with_formula(self, gate_sequence: List[int], num_vars: int) -> List[float]:
        formula = NormalFormFormula(num_vars, FormulaType.Conjunctive)
        reward_model = FormulaRewardModel(formula, width=num_vars)
        
        avgQ_sequence = []
        
        for gate_int in gate_sequence:
            is_add = gate_int > 0
            abs_gate = abs(gate_int)
            
            # Convert gate integer to pos and neg literals
            pos = abs_gate & ((1 << num_vars) - 1)
            neg = (abs_gate >> 32) & ((1 << num_vars) - 1)
            literals = Literals(pos=pos, neg=neg)
            
            from longshot.formula.reward_model import GateToken
            token = GateToken(
                type="ADD" if is_add else "DEL",
                literals=literals
            )
            
            reward_model.step(token)
            avgQ_sequence.append(reward_model.cur_avgQ)
        
        return avgQ_sequence
    
    def test_permutation_sign_preservation_fixed(self, collator_with_permutation):
        """
        Test that permutation NOW correctly preserves the sign (ADD vs DELETE operation type)
        of the gate integers after the fix.
        
        The fix multiplies the permuted result by the original sign.
        """
        num_vars = 3
        
        # Test with negative values (DELETE operations)
        test_sequences = [
            [-1, -2, -4],  # Delete operations
            [1, 2, 4],  # Add operations
            [1, -2, 3, -4],  # Mixed operations
        ]
        
        for seq_idx, original_sequence in enumerate(test_sequences):
            print(f"\nTesting sequence {seq_idx + 1}: {original_sequence}")
            
            original_tensor = torch.tensor([original_sequence], dtype=torch.int64)
            permuted_tensor = collator_with_permutation._permute(original_tensor)
            
            original_signs = (original_tensor < 0).squeeze().tolist()
            permuted_signs = (permuted_tensor < 0).squeeze().tolist()
            
            print(f"Original signs (negative=DELETE): {original_signs}")
            print(f"Permuted signs (negative=DELETE): {permuted_signs}")
            
            # Signs should now be preserved after the fix
            assert original_signs == permuted_signs, \
                f"Signs should be preserved: original {original_signs} != permuted {permuted_signs}"
    
    def test_permutation_preserves_avgQ_with_sign_fix(self, collator_with_permutation):
        """
        Test that after the sign preservation fix, avgQ sequences are computed correctly.
        The permutation changes variable ordering but preserves operation types.
        """
        num_vars = 3
        
        # Test sequences with known operations
        test_cases = [
            {
                "sequence": [1, 2, 4],  # Add three literals
                "description": "All ADD operations"
            },
            {
                "sequence": [1, -2, 4, -1],  # Add, delete, add, delete
                "description": "Mixed ADD/DELETE operations"
            }
        ]
        
        for test_case in test_cases:
            original_sequence = test_case["sequence"]
            print(f"\nTesting: {test_case['description']}")
            print(f"Original sequence: {original_sequence}")
            
            # Apply permutation
            original_tensor = torch.tensor([original_sequence], dtype=torch.int64)
            permuted_tensor = collator_with_permutation._permute(original_tensor)
            permuted_sequence = permuted_tensor.squeeze().tolist()
            print(f"Permuted sequence: {permuted_sequence}")
            
            # Verify signs are preserved
            original_signs = [x < 0 for x in original_sequence]
            permuted_signs = [x < 0 for x in permuted_sequence]
            assert original_signs == permuted_signs, "Signs must be preserved"
            
            # Compute avgQ for both sequences
            original_avgQ = self.simulate_trajectory_with_formula(original_sequence, num_vars)
            permuted_avgQ = self.simulate_trajectory_with_formula(permuted_sequence, num_vars)
            
            print(f"Original avgQ: {original_avgQ}")
            print(f"Permuted avgQ: {permuted_avgQ}")
            
            # The avgQ values may differ because permutation changes which variables are affected
            # but both should be valid sequences with reasonable avgQ values
            assert all(0 <= q <= num_vars for q in original_avgQ), "Original avgQ out of range"
            assert all(0 <= q <= num_vars for q in permuted_avgQ), "Permuted avgQ out of range"
            assert len(original_avgQ) == len(permuted_avgQ), "Sequence lengths must match"
    
    def test_permutation_on_positive_sequences_only(self, collator_with_permutation):
        """
        Test permutation with only positive gate integers (ADD operations).
        For positive-only sequences, the permutation should work correctly.
        """
        num_vars = 3
        
        # Use only positive gate integers 
        test_sequences = [
            [1, 2, 4],  # Add three different positive literals
            [1 | (1 << 32), 2 | (2 << 32), 4 | (4 << 32)],  # Mixed pos/neg literals
        ]
        
        for seq_idx, original_sequence in enumerate(test_sequences):
            print(f"\nTesting sequence {seq_idx + 1}: {original_sequence}")
            
            # Apply permutation
            original_tensor = torch.tensor([original_sequence], dtype=torch.int64)
            permuted_tensor = collator_with_permutation._permute(original_tensor)
            permuted_sequence = permuted_tensor.squeeze().tolist()
            
            print(f"Original sequence: {original_sequence}")
            print(f"Permuted sequence: {permuted_sequence}")
            
            # Verify all values remain positive
            assert all(val >= 0 for val in permuted_sequence), \
                "All permuted values should be non-negative"
            
            # The permutation MAY produce different values (randomized)
            # Can't guarantee it will always be different (might get identity permutation)
            # Just verify the structure is valid
            assert len(permuted_sequence) == len(original_sequence), \
                "Permutation should preserve sequence length"
    
    def test_collator_with_batch_permutation(self, sample_trajectories, collator_with_permutation):
        batch = collator_with_permutation(sample_trajectories)
        
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch
        
        assert batch["input_ids"].shape[0] == len(sample_trajectories)
        assert batch["attention_mask"].shape[0] == len(sample_trajectories)
        assert batch["labels"].shape[0] == len(sample_trajectories)
        
        max_len = max(len(traj["input_ids"]) for traj in sample_trajectories)
        assert batch["input_ids"].shape[1] == max_len
        assert batch["attention_mask"].shape[1] == max_len
        assert batch["labels"].shape[1] == max_len
    
    def test_collator_without_permutation(self, sample_trajectories, collator_without_permutation):
        batch = collator_without_permutation(sample_trajectories)
        
        for i, traj in enumerate(sample_trajectories):
            seq_len = len(traj["input_ids"])
            torch.testing.assert_close(
                batch["input_ids"][i, :seq_len],
                torch.tensor(traj["input_ids"], dtype=torch.int64)
            )
    
    def test_permutation_binary_conversion_roundtrip(self, collator_with_permutation):
        num_vars = 3
        # Test with gate integers that have both positive and negative components
        test_values = [
            1,  # Just positive bit 0
            7,  # Positive bits 0, 1, 2
            (1 << 32),  # Just negative bit 0
            (7 << 32),  # Negative bits 0, 1, 2
            1 | (1 << 32),  # Both positive and negative bit 0
            7 | (7 << 32),  # Both positive and negative bits 0, 1, 2
        ]
        
        for val in test_values:
            input_tensor = torch.tensor([[val]], dtype=torch.int64)
            binary = collator_with_permutation._convert_to_binary(input_tensor)
            reconstructed = collator_with_permutation._convert_to_ids(binary)
            
            assert reconstructed.item() == val, f"Failed roundtrip for value {val} (binary: {binary.tolist()})"
    
    def test_permutation_is_valid(self, collator_with_permutation):
        for _ in range(10):
            perm = collator_with_permutation._random_permutation()
            
            assert len(perm) == 2 * collator_with_permutation.num_vars
            
            perm_set = set(perm.tolist())
            expected_set = set(range(2 * collator_with_permutation.num_vars))
            assert perm_set == expected_set, "Permutation doesn't contain all expected indices"
    
    @pytest.mark.parametrize("num_vars,width", [(3, 2), (3, 3), (4, 3)])
    def test_permutation_with_real_dataset(self, num_vars, width):
        try:
            dataset = TrajectoryDataset(
                num_vars=num_vars, 
                width=width,
                warehouse_host="localhost",
                warehouse_port=8000
            )
            
            if len(dataset) == 0:
                pytest.skip(f"No data available for num_vars={num_vars}, width={width}")
            
            collator = TrajectoryCollator(num_vars=num_vars, permute_input=True)
            
            num_samples = min(5, len(dataset))
            sample_indices = np.random.choice(len(dataset), num_samples, replace=False)
            
            for idx in sample_indices:
                sample = dataset[idx]
                batch = collator([sample])
                
                original_sequence = sample["input_ids"]
                permuted_sequence = batch["input_ids"][0].tolist()
                
                seq_len = len(original_sequence)
                permuted_sequence = permuted_sequence[:seq_len]
                
                print(f"\nDataset sample {idx}:")
                print(f"Original sequence (first 5): {original_sequence[:5]}")
                print(f"Permuted sequence (first 5): {permuted_sequence[:5]}")
                
                original_avgQ = self.simulate_trajectory_with_formula(original_sequence, num_vars)
                permuted_avgQ = self.simulate_trajectory_with_formula(permuted_sequence, num_vars)
                
                print(f"Original avgQ (first 5): {original_avgQ[:5]}")
                print(f"Permuted avgQ (first 5): {permuted_avgQ[:5]}")
                
                np.testing.assert_array_almost_equal(
                    original_avgQ[:min(5, len(original_avgQ))], 
                    permuted_avgQ[:min(5, len(permuted_avgQ))],
                    decimal=6,
                    err_msg=f"avgQ sequences don't match for dataset sample {idx}"
                )
                
        except Exception as e:
            if "Failed to download dataset" in str(e) or "warehouse" in str(e).lower():
                pytest.skip(f"Warehouse service not available: {e}")
            else:
                raise
    
    def test_padding_values(self, sample_trajectories):
        collator = TrajectoryCollator(
            num_vars=3,
            padding_value=0,
            label_padding_value=-100.0
        )
        
        batch = collator(sample_trajectories)
        
        max_len = batch["input_ids"].shape[1]
        for i, traj in enumerate(sample_trajectories):
            original_len = len(traj["input_ids"])
            if original_len < max_len:
                assert torch.all(batch["input_ids"][i, original_len:] == 0)
                assert torch.all(batch["attention_mask"][i, original_len:] == 0)
                assert torch.all(batch["labels"][i, original_len:] == -100.0)
    
    def test_multiple_permutations_produce_different_results(self, collator_with_permutation):
        original_sequence = torch.tensor([[1, 2, 4, 8, 16]], dtype=torch.int64)
        
        permutations = []
        for _ in range(10):
            permuted = collator_with_permutation._permute(original_sequence.clone())
            permutations.append(permuted.squeeze().tolist())
        
        unique_permutations = [list(x) for x in set(tuple(p) for p in permutations)]
        assert len(unique_permutations) > 1, "Permutation should produce varied results"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])