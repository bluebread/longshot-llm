from functools import reduce
import random
import networkx as nx
from ..literals import Clause, Term, Literals
from ..formula import NormalFormFormula, FormulaType, GateToken

def to_lambda(avgQ: float, *, n: int, eps: float) -> float:
    """
    Convert average query complexity to lambda using a specific formula.
    
    :param avgQ: The average query complexity.
    :param n: The total number of arms.
    :param eps: A small epsilon value to avoid division by zero.
    :return: The lambda value derived from the average query complexity. If avgQ is greater than n, returns None.
    """
    return 1 / (1 - (avgQ - eps) / n) if avgQ <= n else None   
    
    
def parse_formula_definition(
    definition: list[int], 
    num_vars: int, 
    formula_type: FormulaType = FormulaType.Conjunctive
) -> NormalFormFormula:
    """
    Parse formula definition into a NormalFormFormula object.
    
    Args:
        definition: 1D list of integers, each representing a gate (literals)
        num_vars: Number of variables in the formula
        formula_type: Type of formula (CNF or DNF)
        
    Returns:
        NormalFormFormula: The constructed formula
    """
    formula = NormalFormFormula(num_vars, formula_type)
    
    for literal_int in definition:
        # Lower 32 bits are positive literals, upper 32 bits are negative literals
        pos_bits = literal_int & 0xFFFFFFFF          # Extract lower 32 bits (0-31)
        neg_bits = (literal_int >> 32) & 0xFFFFFFFF  # Extract upper 32 bits (32-63)
        
        # Create Literals object and add to formula
        if pos_bits != 0 or neg_bits != 0:  # Skip empty literals
            # Add appropriate gate based on formula type
            if formula_type == FormulaType.Conjunctive:
                clause = Clause(pos=pos_bits, neg=neg_bits)
                formula.toggle(clause)
            else:
                term = Term(pos=pos_bits, neg=neg_bits)
                formula.toggle(term)
    
    return formula


def parse_gate_integer_representation(gate_int: int) -> Literals:
    """
    Parse a gate integer representation into a Literals object.
    
    Args:
        gate_int: Integer representation where lower 32 bits are positive literals
                 and upper 32 bits are negative literals
                 
    Returns:
        Literals: Literals object created from the gate integer representation
    """
    # Lower 32 bits are positive literals, upper 32 bits are negative literals  
    pos_bits = gate_int & 0xFFFFFFFF          # Extract lower 32 bits (0-31)
    neg_bits = (gate_int >> 32) & 0xFFFFFFFF  # Extract upper 32 bits (32-63)
    
    # Create and return Literals object using the two-argument constructor
    return Literals(pos_bits, neg_bits)


def generate_random_token(num_vars: int, width: int, rng: random.Random = None) -> GateToken:
    """
    Generate a random GateToken for the RL environment.
    
    Args:
        num_vars: Number of variables in the formula
        width: Fixed width (exact number of literals to generate)
        rng: Optional random number generator instance. If None, uses global random module
        
    Returns:
        GateToken: Random token for the environment
        
    Raises:
        ValueError: If width or num_vars are invalid
    """
    # Validate inputs
    if width <= 0:
        raise ValueError(f"Width must be positive, got {width}")
    if num_vars <= 0:
        raise ValueError(f"num_vars must be positive, got {num_vars}")
    
    # Use provided RNG or fall back to global random module
    if rng is None:
        rng = random
    
    # Randomly choose ADD or DELETE operation
    token_type = rng.choice(['ADD', 'DEL'])
    
    # Generate fixed-width literals within constraint
    effective_width = min(width, num_vars)
    selected_vars = rng.sample(range(num_vars), effective_width)
    
    pos_bits = 0
    neg_bits = 0
    
    for var in selected_vars:
        # Randomly assign positive or negative
        if rng.random() < 0.5:
            pos_bits |= (1 << var)
        else:
            neg_bits |= (1 << var)
    
    literals = Literals(pos=pos_bits, neg=neg_bits)
    return GateToken(type=token_type, literals=literals)


def generate_uniform_token(num_vars: int, width: int, current_gates: set, rng: random.Random = None) -> GateToken:
    """
    Generate a random GateToken using uniform gate sampling with formula-aware ADD/DELETE.
    
    This function uniformly samples a gate from all possible gates, then:
    - If the gate is NOT in the formula, generates an ADD token
    - If the gate IS in the formula, generates a DELETE token
    
    This ensures:
    1. Every generated token is valid (no ADD of existing gates, no DELETE of non-existent gates)
    2. DELETE probability = (formula size) / (# possible gates)
    
    Args:
        num_vars: Number of variables in the formula
        width: Fixed width (exact number of literals to generate)
        current_gates: Set of gate integers currently in the formula
        rng: Optional random number generator instance. If None, uses global random module
        
    Returns:
        GateToken: Token with ADD/DELETE type based on whether gate exists in formula
        
    Raises:
        ValueError: If width or num_vars are invalid
    """
    # Validate inputs
    if width <= 0:
        raise ValueError(f"Width must be positive, got {width}")
    if num_vars <= 0:
        raise ValueError(f"num_vars must be positive, got {num_vars}")
    
    # Use provided RNG or fall back to global random module
    if rng is None:
        rng = random
    
    # Generate a random gate uniformly from all possible gates
    effective_width = min(width, num_vars)
    selected_vars = rng.sample(range(num_vars), effective_width)
    
    pos_bits = 0
    neg_bits = 0
    
    for var in selected_vars:
        # Randomly assign positive or negative
        if rng.random() < 0.5:
            pos_bits |= (1 << var)
        else:
            neg_bits |= (1 << var)
    
    literals = Literals(pos=pos_bits, neg=neg_bits)
    
    # Convert to gate integer representation to check if it exists
    gate_int = pos_bits | (neg_bits << 32)
    
    # Determine token type based on whether gate exists in formula
    if gate_int in current_gates:
        token_type = 'DEL'  # Gate exists, so we DELETE it
    else:
        token_type = 'ADD'  # Gate doesn't exist, so we ADD it
    
    return GateToken(type=token_type, literals=literals)
