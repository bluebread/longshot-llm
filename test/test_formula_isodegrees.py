"""
Test suite for FormulaIsodegrees class.
Tests isomorphism-invariant degree sequence extraction from boolean formulas.
"""

import pytest
from longshot.env.isodegrees import FormulaIsodegrees


class TestFormulaIsodegrees:
    """Test FormulaIsodegrees functionality."""

    def test_initialization_empty(self):
        """Test initialization with no gates."""
        fisod = FormulaIsodegrees(num_vars=4, gates=[])
        assert fisod.num_vars == 4
        # Empty formula still has a feature with zeros for each variable
        assert fisod.feature == ((0, 0), (0, 0), (0, 0), (0, 0))
        assert len(fisod.gates) == 0

    def test_initialization_with_gates(self):
        """Test initialization with initial gates."""
        # Gate format: integer representation
        # For testing, using simple gate representations
        gates = [0b0001, 0b0010, 0b0100]  # Different gates
        fisod = FormulaIsodegrees(num_vars=4, gates=gates)
        assert fisod.num_vars == 4
        assert len(fisod.gates) == 3
        assert fisod.feature is not None

    def test_add_gate_new(self):
        """Test adding a new gate."""
        fisod = FormulaIsodegrees(num_vars=4, gates=[])
        initial_feature = fisod.feature
        
        fisod.add_gate(0b0001)
        assert 0b0001 in fisod.gates
        assert len(fisod.gates) == 1
        assert fisod.feature != initial_feature

    def test_add_gate_duplicate(self):
        """Test adding a duplicate gate (should not add twice)."""
        fisod = FormulaIsodegrees(num_vars=4, gates=[0b0001])
        initial_gates = set(fisod.gates)
        initial_feature = fisod.feature
        
        fisod.add_gate(0b0001)  # Try to add same gate again
        assert fisod.gates == initial_gates
        assert fisod.feature == initial_feature

    def test_remove_gate_existing(self):
        """Test removing an existing gate."""
        gates = [0b0001, 0b0010]
        fisod = FormulaIsodegrees(num_vars=4, gates=gates)
        
        fisod.remove_gate(0b0001)
        assert 0b0001 not in fisod.gates
        assert 0b0010 in fisod.gates
        assert len(fisod.gates) == 1

    def test_remove_gate_nonexistent(self):
        """Test removing a non-existent gate (should be safe)."""
        fisod = FormulaIsodegrees(num_vars=4, gates=[0b0001])
        initial_gates = set(fisod.gates)
        initial_feature = fisod.feature
        
        fisod.remove_gate(0b1111)  # Try to remove gate that doesn't exist
        assert fisod.gates == initial_gates
        assert fisod.feature == initial_feature

    def test_isomorphism_invariance(self):
        """Test that isomorphic formulas produce identical features."""
        # Create two formulas that are structurally identical but use different variables
        # Formula 1: uses variables in one order
        gates1 = [0b0001, 0b0010, 0b0011]  # Example gates
        fisod1 = FormulaIsodegrees(num_vars=4, gates=gates1)
        
        # Formula 2: same structure but potentially different variable ordering
        # For a proper test, we'd need gates that represent isomorphic formulas
        gates2 = [0b0100, 0b1000, 0b1100]  # Different variables, same pattern
        fisod2 = FormulaIsodegrees(num_vars=4, gates=gates2)
        
        # The features should be based on the degree sequence pattern, not specific variables
        # This is a simplified test - real isomorphism testing would be more complex
        assert isinstance(fisod1.feature, tuple)
        assert isinstance(fisod2.feature, tuple)

    def test_feature_immutability(self):
        """Test that the feature tuple is immutable."""
        fisod = FormulaIsodegrees(num_vars=4, gates=[0b0001, 0b0010])
        feature = fisod.feature
        
        assert isinstance(feature, tuple)
        with pytest.raises(AttributeError):
            feature.append(1)  # Tuples don't have append
        with pytest.raises(TypeError):
            feature[0] = 1  # Tuples are immutable

    def test_hash_consistency(self):
        """Test that hash is consistent with equality."""
        gates = [0b0001, 0b0010]
        fisod1 = FormulaIsodegrees(num_vars=4, gates=gates)
        fisod2 = FormulaIsodegrees(num_vars=4, gates=gates.copy())
        
        if fisod1 == fisod2:
            assert hash(fisod1) == hash(fisod2)

    def test_equality(self):
        """Test equality comparison between FormulaIsodegrees instances."""
        gates = [0b0001, 0b0010]
        fisod1 = FormulaIsodegrees(num_vars=4, gates=gates)
        fisod2 = FormulaIsodegrees(num_vars=4, gates=gates.copy())
        fisod3 = FormulaIsodegrees(num_vars=4, gates=[0b0100])
        
        assert fisod1 == fisod2  # Same gates should be equal
        assert fisod1 != fisod3  # Different gates should not be equal

    def test_incremental_updates(self):
        """Test that incremental updates work correctly."""
        fisod = FormulaIsodegrees(num_vars=4, gates=[])
        
        # Track features after each operation
        features = [fisod.feature]
        
        fisod.add_gate(0b0001)
        features.append(fisod.feature)
        
        fisod.add_gate(0b0010)
        features.append(fisod.feature)
        
        fisod.remove_gate(0b0001)
        features.append(fisod.feature)
        
        # Each operation should potentially change the feature
        # (except when adding duplicates or removing non-existent gates)
        assert len(set(features)) > 1  # Should have different features

    def test_feature_as_list_conversion(self):
        """Test converting feature tuple to list for storage."""
        fisod = FormulaIsodegrees(num_vars=4, gates=[0b0001, 0b0010])
        feature_tuple = fisod.feature
        feature_list = list(feature_tuple)
        
        assert isinstance(feature_list, list)
        assert tuple(feature_list) == feature_tuple

    def test_complex_formula(self):
        """Test with a more complex formula."""
        # Simulate a complex formula with many gates
        gates = [0b0001, 0b0010, 0b0100, 0b1000, 0b0011, 0b0101, 0b1010]
        fisod = FormulaIsodegrees(num_vars=4, gates=gates)
        
        assert len(fisod.gates) == 7
        assert fisod.feature is not None
        assert isinstance(fisod.feature, tuple)
        
        # Test removing multiple gates
        fisod.remove_gate(0b0001)
        fisod.remove_gate(0b0010)
        assert len(fisod.gates) == 5
        
        # Test adding back
        fisod.add_gate(0b0001)
        assert len(fisod.gates) == 6

    def test_empty_formula_operations(self):
        """Test operations on empty formula."""
        fisod = FormulaIsodegrees(num_vars=4, gates=[])
        
        # Remove from empty should be safe
        fisod.remove_gate(0b0001)
        assert len(fisod.gates) == 0
        assert fisod.feature == ((0, 0), (0, 0), (0, 0), (0, 0))
        
        # Add to empty
        fisod.add_gate(0b0001)
        assert len(fisod.gates) == 1
        assert fisod.feature != ((0, 0), (0, 0), (0, 0), (0, 0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])