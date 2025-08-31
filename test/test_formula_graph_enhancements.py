"""
Tests for FormulaGraph enhancements including actual variable tracking and width tracking.
"""

import pytest
from longshot.formula import FormulaGraph
from longshot.utils.formula import parse_gate_integer_representation


class TestFormulaGraphVariableTracking:
    """Test actual variable usage tracking in FormulaGraph."""
    
    def test_num_vars_empty_formula(self):
        """Test that empty formula has 0 variables."""
        fg = FormulaGraph()
        assert fg.num_vars == 0
        assert fg.width == 0
    
    def test_num_vars_actual_count(self):
        """Test that num_vars returns the actual count of used variables."""
        # Create formula using x0, x1, x4 (3 variables)
        # Gate with x0 and x1: 0b11 = 3 (positive literals)
        gate1 = 0x00000003  # x0 AND x1
        # Gate with x1 and x4: 0b10010 = 18 (positive literals)
        gate2 = 0x00000012  # x1 AND x4
        
        fg = FormulaGraph([gate1, gate2])
        
        # Should count 3 actual variables (x0, x1, x4), not 5
        assert fg.num_vars == 3
        assert sorted(fg._used_variables) == [0, 1, 4]
    
    def test_num_vars_single_variable(self):
        """Test formula with single variable."""
        # Gate with just x2: 0b100 = 4
        gate = 0x00000004  # x2
        fg = FormulaGraph([gate])
        
        assert fg.num_vars == 1
        assert list(fg._used_variables) == [2]
    
    def test_num_vars_with_negation(self):
        """Test that negated variables are counted correctly."""
        # Gate with x0 positive and x1 negative
        # Positive: 0b1 = 1, Negative: 0b10 = 2
        gate = (2 << 32) | 1  # NOT x1 AND x0
        
        fg = FormulaGraph([gate])
        assert fg.num_vars == 2
        assert sorted(fg._used_variables) == [0, 1]


class TestFormulaGraphWidthTracking:
    """Test width tracking functionality in FormulaGraph."""
    
    def test_width_empty_formula(self):
        """Test that empty formula has width 0."""
        fg = FormulaGraph()
        assert fg.width == 0
        assert fg.width_counter == {}
    
    def test_width_single_gate(self):
        """Test width tracking with a single gate."""
        # Gate with 3 literals: x0, x1, x2
        gate = 0x00000007  # 0b111 - three variables
        fg = FormulaGraph([gate])
        
        assert fg.width == 3
        assert fg.width_counter == {3: 1}
    
    def test_width_multiple_gates_same_width(self):
        """Test width tracking with multiple gates of same width."""
        # Two gates, both with width 2
        gate1 = 0x00000003  # x0, x1
        gate2 = 0x0000000C  # x2, x3
        
        fg = FormulaGraph([gate1, gate2])
        assert fg.width == 2
        assert fg.width_counter == {2: 2}
    
    def test_width_multiple_gates_different_widths(self):
        """Test width tracking with gates of different widths."""
        # Width 1: just x0
        gate1 = 0x00000001
        # Width 2: x1, x2
        gate2 = 0x00000006
        # Width 3: x0, x1, x2
        gate3 = 0x00000007
        
        fg = FormulaGraph([gate1, gate2, gate3])
        
        assert fg.width == 3  # Maximum width
        assert fg.width_counter == {1: 1, 2: 1, 3: 1}
    
    def test_width_with_negations(self):
        """Test that width counts both positive and negative literals."""
        # Gate with x0 positive, x1 negative (width = 2)
        gate = (0x00000002 << 32) | 0x00000001
        
        fg = FormulaGraph([gate])
        
        # Parse to verify
        literals = parse_gate_integer_representation(gate)
        assert literals.width == 2
        
        assert fg.width == 2
        assert fg.width_counter == {2: 1}
    
    def test_width_add_remove_gates(self):
        """Test width tracking when adding and removing gates."""
        fg = FormulaGraph()
        
        # Add gate with width 2
        gate1 = 0x00000003  # x0, x1
        fg.add_gate(gate1)
        assert fg.width == 2
        assert fg.width_counter == {2: 1}
        
        # Add gate with width 3
        gate2 = 0x00000007  # x0, x1, x2
        fg.add_gate(gate2)
        assert fg.width == 3
        assert fg.width_counter == {2: 1, 3: 1}
        
        # Remove the width-3 gate
        fg.remove_gate(gate2)
        assert fg.width == 2  # Back to max width of 2
        assert fg.width_counter == {2: 1}
        
        # Remove the width-2 gate
        fg.remove_gate(gate1)
        assert fg.width == 0
        assert fg.width_counter == {}
    
    def test_width_from_definition(self):
        """Test that from_definition properly initializes width tracking."""
        gates = [
            0x00000001,  # Width 1
            0x00000003,  # Width 2
            0x00000007,  # Width 3
            0x0000000F,  # Width 4
        ]
        
        fg = FormulaGraph(gates)
        assert fg.width == 4
        assert fg.width_counter == {1: 1, 2: 1, 3: 1, 4: 1}
    
    def test_width_clear(self):
        """Test that clear() resets width tracking."""
        fg = FormulaGraph([0x00000007])  # Width 3
        assert fg.width == 3
        assert fg.width_counter == {3: 1}
        
        fg.clear()
        assert fg.width == 0
        assert fg.width_counter == {}
    
    def test_width_copy(self):
        """Test that copy preserves width tracking."""
        fg = FormulaGraph([0x00000003, 0x00000007])
        fg_copy = fg.copy()
        
        assert fg_copy.width == fg.width
        assert fg_copy.width_counter == fg.width_counter
        assert fg_copy.width_counter is not fg.width_counter  # Different object


class TestFormulaGraphStatistics:
    """Test that get_statistics includes new width information."""
    
    def test_statistics_includes_width(self):
        """Test that statistics include width and width_counter."""
        gates = [0x00000003, 0x00000007]  # Width 2 and 3
        fg = FormulaGraph(gates)
        
        stats = fg.get_statistics()
        assert "width" in stats
        assert stats["width"] == 3
        assert "width_counter" in stats
        assert stats["width_counter"] == {2: 1, 3: 1}
    
    def test_statistics_empty_formula(self):
        """Test statistics for empty formula."""
        fg = FormulaGraph()
        stats = fg.get_statistics()
        
        assert stats["width"] == 0
        assert stats["width_counter"] == {}
        assert stats["num_variables"] == 0


class TestFormulaGraphIntegration:
    """Integration tests for FormulaGraph changes."""
    
    def test_complex_formula(self):
        """Test a complex formula with various properties."""
        # Create a formula with:
        # - Variables x0, x1, x2, x5, x7 (5 variables, not 8)
        # - Different widths
        
        gate1 = 0x00000001  # x0 only (width 1)
        gate2 = 0x00000003  # x0, x1 (width 2)
        gate3 = 0x00000007  # x0, x1, x2 (width 3)
        gate4 = 0x00000020  # x5 only (width 1)
        gate5 = 0x00000080  # x7 only (width 1)
        gate6 = (0x00000002 << 32) | 0x00000081  # x0, NOT x1, x7 (width 3)
        
        fg = FormulaGraph([gate1, gate2, gate3, gate4, gate5, gate6])
        
        # Check num_vars counts actual variables used
        assert fg.num_vars == 5  # x0, x1, x2, x5, x7
        assert sorted(fg._used_variables) == [0, 1, 2, 5, 7]
        
        # Check width is maximum width
        assert fg.width == 3
        
        # Check width counter
        assert fg.width_counter == {1: 3, 2: 1, 3: 2}
        
        # Check size (number of gates)
        assert fg.size == 6
    
    def test_add_remove_sequence(self):
        """Test that add/remove sequence maintains correct tracking."""
        fg = FormulaGraph()
        
        # Start empty
        assert fg.num_vars == 0
        assert fg.width == 0
        
        # Add first gate
        gate1 = 0x00000003  # x0, x1 (width 2)
        fg.add_gate(gate1)
        assert fg.num_vars == 2
        assert fg.width == 2
        
        # Add second gate with new variables
        gate2 = 0x00000014  # x2, x4 (width 2)
        fg.add_gate(gate2)
        assert fg.num_vars == 4  # x0, x1, x2, x4 -> 4 vars total
        assert fg.width == 2
        
        # Add gate with higher width
        gate3 = 0x0000001F  # x0-x4 (width 5)
        fg.add_gate(gate3)
        assert fg.num_vars == 5  # Now using all x0-x4
        assert fg.width == 5
        
        # Remove the high-width gate
        fg.remove_gate(gate3)
        assert fg.num_vars == 4  # Back to x0, x1, x2, x4
        assert fg.width == 2  # Back to max width 2
        
        # Remove all gates
        fg.remove_gate(gate1)
        fg.remove_gate(gate2)
        assert fg.num_vars == 0
        assert fg.width == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])