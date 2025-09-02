"""Test FormulaRewardModel class."""

import pytest
from longshot.formula import NormalFormFormula, FormulaRewardModel, GateToken
from longshot.literals import Literals
from longshot.error import LongshotError


class TestFormulaRewardModel:
    """Test FormulaRewardModel functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a simple formula for testing with 4 variables
        self.formula = NormalFormFormula(num_vars=4)
        self.formula.toggle(Literals(pos=0b0011, neg=0))  # x0 & x1
        self.formula.toggle(Literals(pos=0b0100, neg=0))  # x2
        
    def test_init_with_default_parameters(self):
        """Test initialization with default parameters."""
        model = FormulaRewardModel(self.formula)
        
        # Check default values
        assert model._num_vars == self.formula.num_vars
        assert model._width == self.formula.num_vars
        assert model._size is None  # No size constraint by default
        assert model._eps == 1 / self.formula.num_vars
        
    def test_init_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        model = FormulaRewardModel(
            self.formula,
            width=2,
            size=10,
            eps=0.1,
            penalty=-2.0
        )
        
        assert model._width == 2
        assert model._size == 10
        assert model._eps == 0.1
        assert model._kwargs.get('penalty') == -2.0
        
    def test_init_with_no_size_constraint(self):
        """Test initialization with None size (no constraint)."""
        model = FormulaRewardModel(self.formula, size=None)
        assert model._size is None
        
        # Should not raise error even with large formula
        large_formula = NormalFormFormula(num_vars=10)
        for i in range(100):
            large_formula.toggle(Literals(pos=1 << (i % 10), neg=0))
        
        # This should work with no size constraint
        model2 = FormulaRewardModel(large_formula, size=None)
        assert model2._size is None
        
    def test_init_raises_error_for_invalid_formula(self):
        """Test that initialization raises error for None or invalid formula."""
        with pytest.raises(LongshotError, match="Formula must be an instance"):
            FormulaRewardModel(None)
            
    def test_init_raises_error_for_size_violation(self):
        """Test that initialization raises error when formula exceeds size limit."""
        # Create a formula with 3 gates
        formula = NormalFormFormula(num_vars=4)
        formula.toggle(Literals(pos=0b0001, neg=0))
        formula.toggle(Literals(pos=0b0010, neg=0))
        formula.toggle(Literals(pos=0b0100, neg=0))
        
        # Should raise error with size limit of 2
        with pytest.raises(LongshotError, match="gates greater than"):
            FormulaRewardModel(formula, size=2)
            
    def test_init_raises_error_for_width_violation(self):
        """Test that initialization raises error when formula exceeds width limit."""
        # Create a formula with width 3
        formula = NormalFormFormula(num_vars=4)
        formula.toggle(Literals(pos=0b0111, neg=0))  # width = 3
        
        # Should raise error with width limit of 2
        with pytest.raises(LongshotError, match="width .* greater than"):
            FormulaRewardModel(formula, width=2)
            
    def test_step_valid_add_operation(self):
        """Test step with valid ADD operation."""
        model = FormulaRewardModel(self.formula)
        
        # Add a new gate that doesn't exist
        new_gate = Literals(pos=0b1000, neg=0)  # x3
        token = GateToken(type='ADD', literals=new_gate)
        
        reward = model.step(token)
        
        # Should get positive reward for valid operation
        assert reward > 0
        assert new_gate in model._cur_f
        
    def test_step_valid_del_operation(self):
        """Test step with valid DELETE operation."""
        model = FormulaRewardModel(self.formula)
        
        # Delete an existing gate
        existing_gate = Literals(pos=0b0011, neg=0)  # x0 & x1
        token = GateToken(type='DEL', literals=existing_gate)
        
        reward = model.step(token)
        
        # Should get reward for valid operation
        assert reward != model._kwargs.get('penalty', -1.0)
        assert existing_gate not in model._cur_f
        
    def test_step_invalid_add_existing_gate(self):
        """Test step with invalid ADD of existing gate."""
        model = FormulaRewardModel(self.formula)
        
        # Try to add an existing gate
        existing_gate = Literals(pos=0b0011, neg=0)  # Already in formula
        token = GateToken(type='ADD', literals=existing_gate)
        
        reward = model.step(token)
        
        # Should get penalty
        assert reward == model._kwargs.get('penalty', -1.0)
        
    def test_step_invalid_del_nonexistent_gate(self):
        """Test step with invalid DELETE of non-existent gate."""
        model = FormulaRewardModel(self.formula)
        
        # Try to delete a non-existent gate
        nonexistent_gate = Literals(pos=0b10000, neg=0)
        token = GateToken(type='DEL', literals=nonexistent_gate)
        
        reward = model.step(token)
        
        # Should get penalty
        assert reward == model._kwargs.get('penalty', -1.0)
        
    def test_step_width_violation(self):
        """Test step with width constraint violation."""
        model = FormulaRewardModel(self.formula, width=2)
        
        # Try to add a gate with width 3
        wide_gate = Literals(pos=0b0111, neg=0)  # width = 3
        token = GateToken(type='ADD', literals=wide_gate)
        
        reward = model.step(token)
        
        # Should get penalty for width violation
        assert reward == model._kwargs.get('penalty', -1.0)
        
    def test_step_size_violation_with_limit(self):
        """Test step with size constraint violation when limit exists."""
        # Create model with size limit of 3 (formula already has 2 gates)
        model = FormulaRewardModel(self.formula, size=3)
        
        # Add one gate (should work, total becomes 3)
        gate1 = Literals(pos=0b1000, neg=0)
        token1 = GateToken(type='ADD', literals=gate1)
        reward1 = model.step(token1)
        assert reward1 > 0  # Should succeed
        
        # Try to add another gate (should fail, would exceed size 3)
        gate2 = Literals(pos=0b10000, neg=0)
        token2 = GateToken(type='ADD', literals=gate2)
        reward2 = model.step(token2)
        
        # Should get penalty for size violation
        assert reward2 == model._kwargs.get('penalty', -1.0)
        
    def test_step_no_size_violation_without_limit(self):
        """Test step with no size constraint (size=None)."""
        model = FormulaRewardModel(self.formula, size=None)
        
        # Add many gates - should all succeed
        for i in range(10):
            gate = Literals(pos=1 << (i + 4), neg=0)  # Start from x4
            token = GateToken(type='ADD', literals=gate)
            reward = model.step(token)
            
            # Should not get penalty (no size limit)
            assert reward != model._kwargs.get('penalty', -1.0)
            
    def test_step_constant_literals(self):
        """Test step with constant literals (invalid)."""
        model = FormulaRewardModel(self.formula)
        
        # Try to add constant literals (all zeros)
        constant_gate = Literals(pos=0, neg=0)
        token = GateToken(type='ADD', literals=constant_gate)
        
        reward = model.step(token)
        
        # Should get penalty for constant literals
        assert reward == model._kwargs.get('penalty', -1.0)
        
    def test_reset(self):
        """Test reset functionality."""
        model = FormulaRewardModel(self.formula)
        
        # Modify the formula
        new_gate = Literals(pos=0b1000, neg=0)
        token = GateToken(type='ADD', literals=new_gate)
        model.step(token)
        
        # Verify formula was modified
        assert new_gate in model._cur_f
        
        # Reset
        model.reset()
        
        # Should be back to initial state
        assert new_gate not in model._cur_f
        assert model._cur_f.num_gates == self.formula.num_gates
        assert model._cur_avgQ == model._init_avgQ
        
    def test_custom_penalty(self):
        """Test custom penalty value."""
        model = FormulaRewardModel(self.formula, penalty=-5.0)
        
        # Try invalid operation
        existing_gate = Literals(pos=0b0011, neg=0)
        token = GateToken(type='ADD', literals=existing_gate)
        
        reward = model.step(token)
        
        # Should get custom penalty
        assert reward == -5.0
        
    def test_reward_calculation(self):
        """Test that reward calculation follows the formula."""
        model = FormulaRewardModel(self.formula)
        
        # Perform a valid operation and check reward
        new_gate = Literals(pos=0b1000, neg=0)
        token = GateToken(type='ADD', literals=new_gate)
        
        reward = model.step(token)
        
        # Reward should be q + lambda where lambda = 1/(1-(q-eps)/n)
        q = model._cur_avgQ
        n = model._num_vars
        eps = model._eps
        expected_lambda = 1 / (1 - (q - eps) / n)
        expected_reward = q + expected_lambda
        
        assert abs(reward - expected_reward) < 1e-10
        
    def test_multiple_steps(self):
        """Test multiple sequential steps."""
        model = FormulaRewardModel(self.formula, size=5)
        
        # Add a gate
        gate1 = Literals(pos=0b1000, neg=0)
        token1 = GateToken(type='ADD', literals=gate1)
        reward1 = model.step(token1)
        assert reward1 > 0
        
        # Delete a gate
        gate2 = Literals(pos=0b0011, neg=0)
        token2 = GateToken(type='DEL', literals=gate2)
        reward2 = model.step(token2)
        assert reward2 != model._kwargs.get('penalty', -1.0)
        
        # Add another gate
        gate3 = Literals(pos=0b10000, neg=0)
        token3 = GateToken(type='ADD', literals=gate3)
        reward3 = model.step(token3)
        assert reward3 > 0
        
        # Verify final state
        assert gate1 in model._cur_f
        assert gate2 not in model._cur_f
        assert gate3 in model._cur_f