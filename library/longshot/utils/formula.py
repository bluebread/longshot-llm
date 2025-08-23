from functools import reduce
import random
import networkx as nx
from ..circuit import NormalFormFormula, FormulaType, Clause, Term, Literals
from ..models import GateToken

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
        width: Maximum width constraint
        rng: Optional random number generator instance. If None, uses global random module
        
    Returns:
        GateToken: Random token for the environment
    """
    # Use provided RNG or fall back to global random module
    if rng is None:
        rng = random
    
    # Randomly choose ADD or DELETE operation
    token_type = rng.choice(['ADD', 'DEL'])
    
    # Generate random literals within width constraint
    num_literals = rng.randint(1, min(width, num_vars))
    selected_vars = rng.sample(range(num_vars), num_literals)
    
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
