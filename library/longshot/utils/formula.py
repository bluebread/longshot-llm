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
    

def add_gate_to_graph(graph: nx.Graph, gate: int) -> None:
    """
    Adds a gate to the graph with its positive and negative literals.
    
    Parameters:
        graph (nx.Graph): The graph to which the gate will be added.
        gate (int): The gate identifier.
    """
    graph.add_node(gate, label="gate")
    pl = gate & 0xFFFFFFFF
    nl = (gate >> 32) & 0xFFFFFFFF
    vl = pl | nl

    vs = [i for i in range(32) if (vl & (1 << i)) != 0]
    pv = [i for i in range(32) if (pl & (1 << i)) != 0]
    nv = [i for i in range(32) if (nl & (1 << i)) != 0]
    
    # Add variables and literals that are not already in the graph
    new_vs = [f"x{i}" for i in vs if f"x{i}" not in graph]
    new_pv = [f"+x{i}" for i in pv if f"+x{i}" not in graph]
    new_nv = [f"-x{i}" for i in nv if f"-x{i}" not in graph]
    new_pv_edges = [(f"+x{i}", f"x{i}") for i in pv if f"+x{i}" not in graph]
    new_nv_edges = [(f"-x{i}", f"x{i}") for i in nv if f"-x{i}" not in graph]
    graph.add_nodes_from(new_vs, label="variable")
    graph.add_nodes_from(new_pv + new_nv, label="literal")
    graph.add_edges_from(new_pv_edges + new_nv_edges)
    
    # Add edges from literals to the gate
    edges = [(f"+x{i}", gate) for i in pv]
    edges += [(f"-x{i}", gate) for i in nv]
    graph.add_edges_from(edges)


def del_gate_from_graph(graph: nx.Graph, gate: int) -> None:
    """
    Removes a gate from the graph along with its associated literals.
    
    Parameters:
        graph (nx.Graph): The graph from which the gate will be removed.
        gate (int): The gate identifier to remove.
    """
    if gate in graph:
        graph.remove_node(gate)
    # otherwise, ignore as it is not present in the graph
    

def definition_to_graph(definition: list) -> nx.Graph:
    """
    Converts a formula definition to a networkx graph.
    
    Parameters:
        definition (list): The formula definition containing variables and their connections.
    
    Returns:
        nx.Graph: A graph representation of the formula.
    """
    vars: int = reduce(lambda x, y: x | y, definition, 0)
    pos_vars = vars & 0xFFFFFFFF
    neg_vars = (vars >> 32) & 0xFFFFFFFF
    vars = pos_vars | neg_vars
    
    if vars == 0:
        return nx.Graph()
    
    num_vars = vars.bit_length()
    formula_graph = nx.Graph()
    
    vars = [i for i in range(num_vars) if (vars & (1 << i)) != 0]
    pos_vars = [i for i in range(num_vars) if (pos_vars & (1 << i)) != 0]
    neg_vars = [i for i in range(num_vars) if (neg_vars & (1 << i)) != 0]
    
    var_nodes = [f"x{i}" for i in vars]
    pos_nodes = [f"+x{i}" for i in pos_vars]
    neg_nodes = [f"-x{i}" for i in neg_vars]

    formula_graph.add_nodes_from(var_nodes, label="variable")
    formula_graph.add_nodes_from(pos_nodes + neg_nodes, label="literal")
    formula_graph.add_edges_from([(f"+x{i}", f"x{i}") for i in pos_vars])
    formula_graph.add_edges_from([(f"-x{i}", f"x{i}") for i in neg_vars])
    
    for gate in definition:
        add_gate_to_graph(formula_graph, gate)

    return formula_graph


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
    token_type = rng.choice(['ADD', 'DELETE'])
    
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
