"""Test formula utility functions."""

import pytest
import random
from longshot.utils.formula import (
    generate_random_token,
    generate_uniform_token,
    parse_gate_integer_representation
)
from longshot.literals import Literals


class TestFormulaUtils:
    """Test formula utility functions."""
    
    def test_generate_uniform_token_empty_formula(self):
        """Test generating token when formula is empty - should always ADD."""
        current_gates = set()  # Empty formula
        
        # Test multiple times to ensure consistency
        for _ in range(10):
            token = generate_uniform_token(
                num_vars=4,
                width=2,
                current_gates=current_gates,
                rng=random.Random(42)
            )
            # When formula is empty, should always ADD
            assert token.type == 'ADD'
            assert token.literals.width == 2
    
    def test_generate_uniform_token_with_existing_gate(self):
        """Test generating token when the exact gate exists - should DELETE."""
        # Create a specific gate (variables 0 and 1, both positive)
        pos_bits = (1 << 0) | (1 << 1)  # bits 0 and 1 set
        neg_bits = 0
        gate_int = pos_bits | (neg_bits << 32)
        current_gates = {gate_int}
        
        # Use a fixed seed to generate the same gate
        rng = random.Random(12345)
        
        # Generate tokens until we hit the existing gate
        found_delete = False
        for _ in range(100):
            token = generate_uniform_token(
                num_vars=2,
                width=2,
                current_gates=current_gates,
                rng=rng
            )
            
            # Convert token to gate int to check
            token_gate_int = token.literals.pos | (token.literals.neg << 32)
            
            if token_gate_int == gate_int:
                # If we generated the same gate, it should be DELETE
                assert token.type == 'DEL'
                found_delete = True
                break
            else:
                # If different gate, should be ADD
                assert token.type == 'ADD'
        
        # We should have found at least one DELETE in 100 tries
        assert found_delete, "Should have generated DELETE for existing gate"
    
    def test_generate_uniform_token_mixed_formula(self):
        """Test with a formula containing multiple gates."""
        # Create a formula with several gates
        current_gates = {
            0b0011,  # Gates with first two variables
            0b0101,  # Gates with alternating variables
            0b1111,  # Gate with all four variables (for num_vars=4)
        }
        
        rng = random.Random(999)
        add_count = 0
        del_count = 0
        
        # Generate many tokens to check distribution
        for _ in range(1000):
            token = generate_uniform_token(
                num_vars=4,
                width=2,
                current_gates=current_gates,
                rng=rng
            )
            
            if token.type == 'ADD':
                add_count += 1
                # Verify it's not adding an existing gate
                token_gate_int = token.literals.pos | (token.literals.neg << 32)
                assert token_gate_int not in current_gates
            else:
                del_count += 1
                # Verify it's deleting an existing gate
                token_gate_int = token.literals.pos | (token.literals.neg << 32)
                assert token_gate_int in current_gates
        
        # Should have both ADDs and DELs
        assert add_count > 0
        assert del_count > 0
        
        # DEL probability should be roughly: len(current_gates) / total_possible_gates
        # For width=2, num_vars=4: C(4,2) * 2^2 = 6 * 4 = 24 possible gates
        # With 3 gates in formula: expected DEL ratio = 3/24 = 0.125
        del_ratio = del_count / (add_count + del_count)
        expected_ratio = 3 / 24
        
        # Allow some variance due to randomness (within 5% absolute difference)
        assert abs(del_ratio - expected_ratio) < 0.05, \
            f"DEL ratio {del_ratio:.3f} too far from expected {expected_ratio:.3f}"
    
    def test_generate_uniform_token_validates_inputs(self):
        """Test input validation."""
        current_gates = set()
        
        # Test invalid width
        with pytest.raises(ValueError, match="Width must be positive"):
            generate_uniform_token(
                num_vars=4,
                width=0,
                current_gates=current_gates
            )
        
        # Test invalid num_vars
        with pytest.raises(ValueError, match="num_vars must be positive"):
            generate_uniform_token(
                num_vars=0,
                width=2,
                current_gates=current_gates
            )
    
    def test_generate_uniform_token_deterministic_with_seed(self):
        """Test that same seed produces same results."""
        current_gates = {1, 2, 3}
        
        # Generate sequence with one seed
        rng1 = random.Random(12345)
        tokens1 = [
            generate_uniform_token(
                num_vars=4,
                width=2,
                current_gates=current_gates,
                rng=rng1
            )
            for _ in range(10)
        ]
        
        # Generate sequence with same seed
        rng2 = random.Random(12345)
        tokens2 = [
            generate_uniform_token(
                num_vars=4,
                width=2,
                current_gates=current_gates,
                rng=rng2
            )
            for _ in range(10)
        ]
        
        # Should produce identical sequences
        for t1, t2 in zip(tokens1, tokens2):
            assert t1.type == t2.type
            assert t1.literals.pos == t2.literals.pos
            assert t1.literals.neg == t2.literals.neg
    
    def test_generate_uniform_token_width_constraint(self):
        """Test that generated tokens respect width constraint."""
        current_gates = set()
        rng = random.Random(777)
        
        for width in [1, 2, 3, 4]:
            for _ in range(20):
                token = generate_uniform_token(
                    num_vars=5,
                    width=width,
                    current_gates=current_gates,
                    rng=rng
                )
                
                # Check that exactly 'width' variables are used
                assert token.literals.width == width
    
    def test_generate_uniform_token_with_width_exceeding_vars(self):
        """Test behavior when width > num_vars."""
        current_gates = set()
        
        # Request width=5 but only 3 variables available
        token = generate_uniform_token(
            num_vars=3,
            width=5,
            current_gates=current_gates,
            rng=random.Random(111)
        )
        
        # Should use min(width, num_vars) = 3
        assert token.literals.width == 3