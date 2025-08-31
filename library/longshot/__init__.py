"""
Longshot: An RL-based framework for logical formula generation and filtering.

This package contains tools for constructing, evaluating, and evolving propositional logic formulas
within reinforcement learning environments. Major components include:

- agent: RL agents and training loop
- circuit: representations and utilities for logic circuits
- error: custom exceptions and error handling
- models: neural network architectures used in the system
- utils: general utility functions
"""
import longshot.literals
import longshot.error
import longshot.models
import longshot.utils
import longshot.formula