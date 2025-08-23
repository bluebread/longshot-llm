"""
Formula graph representation and manipulation utilities.

This module provides the FormulaGraph class for working with graph representations
of boolean formulas, including methods for adding/removing gates, converting
between definitions and graphs, and checking for isomorphism.
"""

from functools import reduce
from typing import Optional, List, Dict, Any
import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from networkx.algorithms.isomorphism import vf2pp_is_isomorphic


class FormulaGraph:
    """
    A graph representation of a boolean formula.
    
    This class maintains a NetworkX graph and provides methods for:
    - Adding and removing gates
    - Converting between definitions and graphs
    - Computing hashes and checking isomorphism
    - Tracking formula properties
    """
    
    def __init__(self, definition: Optional[List[int]] = None):
        """
        Initialize a FormulaGraph.
        
        Args:
            definition: Optional list of gate integers to initialize the graph
        """
        self.graph = nx.Graph()
        self._used_variables = set()  # Track which variables are actually used
        self._size = 0  # Number of gates
        
        if definition:
            self.from_definition(definition)
    
    def add_gate(self, gate: int) -> None:
        """
        Add a gate to the graph with its positive and negative literals.
        
        Args:
            gate: The gate identifier (encoded as integer with positive and negative literals)
        """
        if gate in self.graph:
            return
        
        self.graph.add_node(gate, label="gate")
        self._size += 1
        
        # Extract positive and negative literals
        pos_literals = gate & 0xFFFFFFFF
        neg_literals = (gate >> 32) & 0xFFFFFFFF
        all_literals = pos_literals | neg_literals
        
        # Find all variables involved
        variables = [i for i in range(32) if (all_literals & (1 << i)) != 0]
        pos_vars = [i for i in range(32) if (pos_literals & (1 << i)) != 0]
        neg_vars = [i for i in range(32) if (neg_literals & (1 << i)) != 0]
        
        # Update the set of used variables
        for var_idx in variables:
            self._used_variables.add(var_idx)
        
        # Add variable nodes if not present
        new_vars = [f"x{i}" for i in variables if f"x{i}" not in self.graph]
        new_pos_lits = [f"+x{i}" for i in pos_vars if f"+x{i}" not in self.graph]
        new_neg_lits = [f"-x{i}" for i in neg_vars if f"-x{i}" not in self.graph]
        
        # Add nodes
        self.graph.add_nodes_from(new_vars, label="variable")
        self.graph.add_nodes_from(new_pos_lits + new_neg_lits, label="literal")
        
        # Add edges from literals to variables
        pos_var_edges = [(f"+x{i}", f"x{i}") for i in pos_vars]
        neg_var_edges = [(f"-x{i}", f"x{i}") for i in neg_vars]
        self.graph.add_edges_from(pos_var_edges + neg_var_edges)
        
        # Add edges from literals to the gate
        gate_edges = [(f"+x{i}", gate) for i in pos_vars]
        gate_edges += [(f"-x{i}", gate) for i in neg_vars]
        self.graph.add_edges_from(gate_edges)
    
    def remove_gate(self, gate: int) -> None:
        """
        Remove a gate from the graph.
        
        Args:
            gate: The gate identifier to remove
        """
        if gate not in self.graph:
            return
            
        # Find variables used by this gate
        gate_vars = set()
        for neighbor in self.graph.neighbors(gate):
            if isinstance(neighbor, str) and (neighbor.startswith('+x') or neighbor.startswith('-x')):
                var_idx = int(neighbor[2:])
                gate_vars.add(var_idx)
        
        # Remove the gate
        self.graph.remove_node(gate)
        self._size -= 1
        
        # Check if any literals or variables are now unused
        for var_idx in gate_vars:
            var_node = f"x{var_idx}"
            pos_lit = f"+x{var_idx}"
            neg_lit = f"-x{var_idx}"
            
            # Check if positive literal is still used by any gate
            pos_lit_used = False
            if pos_lit in self.graph:
                for neighbor in self.graph.neighbors(pos_lit):
                    if isinstance(neighbor, int):  # It's a gate
                        pos_lit_used = True
                        break
            
            # Check if negative literal is still used by any gate
            neg_lit_used = False
            if neg_lit in self.graph:
                for neighbor in self.graph.neighbors(neg_lit):
                    if isinstance(neighbor, int):  # It's a gate
                        neg_lit_used = True
                        break
            
            # Remove unused positive literal
            if not pos_lit_used and pos_lit in self.graph:
                self.graph.remove_node(pos_lit)
            
            # Remove unused negative literal
            if not neg_lit_used and neg_lit in self.graph:
                self.graph.remove_node(neg_lit)
            
            # Remove variable node if neither literal is used
            if not pos_lit_used and not neg_lit_used:
                self._used_variables.discard(var_idx)
                if var_node in self.graph:
                    self.graph.remove_node(var_node)
    
    def from_definition(self, definition: List[int]) -> None:
        """
        Initialize the graph from a formula definition.
        
        Args:
            definition: List of gate integers representing the formula
        """
        # Clear existing graph
        self.clear()
        
        if not definition:
            return
        
        # Combine all gates to find all variables
        all_vars = reduce(lambda x, y: x | y, definition, 0)
        pos_vars = all_vars & 0xFFFFFFFF
        neg_vars = (all_vars >> 32) & 0xFFFFFFFF
        all_vars = pos_vars | neg_vars
        
        if all_vars == 0:
            return
        
        max_bit = all_vars.bit_length()
        
        # Extract variable indices and update used_variables set
        var_indices = [i for i in range(max_bit) if (all_vars & (1 << i)) != 0]
        pos_var_indices = [i for i in range(max_bit) if (pos_vars & (1 << i)) != 0]
        neg_var_indices = [i for i in range(max_bit) if (neg_vars & (1 << i)) != 0]
        
        # Update the set of used variables
        self._used_variables = set(var_indices)
        
        # Create nodes
        var_nodes = [f"x{i}" for i in var_indices]
        pos_literal_nodes = [f"+x{i}" for i in pos_var_indices]
        neg_literal_nodes = [f"-x{i}" for i in neg_var_indices]
        
        # Add nodes to graph
        self.graph.add_nodes_from(var_nodes, label="variable")
        self.graph.add_nodes_from(pos_literal_nodes + neg_literal_nodes, label="literal")
        
        # Add edges from literals to variables
        self.graph.add_edges_from([(f"+x{i}", f"x{i}") for i in pos_var_indices])
        self.graph.add_edges_from([(f"-x{i}", f"x{i}") for i in neg_var_indices])
        
        # Add all gates
        for gate in definition:
            self.add_gate(gate)
    
    def to_definition(self) -> List[int]:
        """
        Convert the graph to a formula definition (list of gate integers).
        
        Returns:
            List of gate integers representing the formula
        """
        gates = []
        
        # Find all gate nodes
        for node in self.graph.nodes():
            if isinstance(node, int) and self.graph.nodes[node].get("label") == "gate":
                gates.append(node)
        
        return sorted(gates)
    
    def wl_hash(self, iterations: int = 5) -> str:
        """
        Calculate the Weisfeiler-Lehman hash of the formula graph.
        
        Args:
            iterations: Number of WL iterations (default: 5)
            
        Returns:
            Weisfeiler-Lehman hash string
        """
        # Even an empty graph should have a proper WL hash
        return weisfeiler_lehman_graph_hash(
            self.graph,
            node_attr="label",
            iterations=iterations
        )
    
    def is_isomorphic_to(self, other: 'FormulaGraph') -> bool:
        """
        Check if this graph is isomorphic to another FormulaGraph.
        
        Args:
            other: Another FormulaGraph instance
            
        Returns:
            True if graphs are isomorphic, False otherwise
        """
        if self.graph.number_of_nodes() != other.graph.number_of_nodes():
            return False
        
        if self.graph.number_of_edges() != other.graph.number_of_edges():
            return False
        
        # Empty graphs are isomorphic
        if self.graph.number_of_nodes() == 0:
            return True
        
        # Quick check using WL hash
        if self.wl_hash() != other.wl_hash():
            return False
        
        # Detailed isomorphism check
        return vf2pp_is_isomorphic(
            self.graph, 
            other.graph,
            node_label="label"
        )
    
    def clear(self) -> None:
        """Clear the graph and reset all attributes."""
        self.graph.clear()
        self._used_variables.clear()
        self._size = 0
    
    def copy(self) -> 'FormulaGraph':
        """
        Create a deep copy of this FormulaGraph.
        
        Returns:
            A new FormulaGraph instance with the same graph structure
        """
        new_graph = FormulaGraph()
        new_graph.graph = self.graph.copy()
        new_graph._used_variables = self._used_variables.copy()
        new_graph._size = self._size
        return new_graph
    
    @property
    def num_vars(self) -> int:
        """Get the number of variables in the formula."""
        # Return the maximum variable index + 1, or 0 if no variables
        return max(self._used_variables, default=-1) + 1
    
    @property
    def size(self) -> int:
        """Get the number of gates in the formula."""
        return self._size
    
    @property
    def num_nodes(self) -> int:
        """Get the total number of nodes in the graph."""
        return self.graph.number_of_nodes()
    
    @property
    def num_edges(self) -> int:
        """Get the total number of edges in the graph."""
        return self.graph.number_of_edges()
    
    @property
    def is_empty(self) -> bool:
        """Check if the graph is empty."""
        return self.graph.number_of_nodes() == 0
    
    @property
    def gates(self) -> List[int]:
        """Get all gates in the formula as a list of integers."""
        return [
            node for node in self.graph.nodes() 
            if isinstance(node, int) and self.graph.nodes[node].get("label") == "gate"
        ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the formula graph.
        
        Returns:
            Dictionary with graph statistics
        """
        if self.is_empty:
            return {
                "num_nodes": 0,
                "num_edges": 0,
                "num_variables": 0,
                "used_variables": [],
                "num_literals": 0,
                "num_gates": 0,
                "size": 0,
                "density": 0.0,
                "is_connected": False,
                "wl_hash": self.wl_hash()
            }
        
        # Count node types
        num_var_nodes = sum(1 for n in self.graph.nodes() 
                            if self.graph.nodes[n].get("label") == "variable")
        num_literal_nodes = sum(1 for n in self.graph.nodes() 
                               if self.graph.nodes[n].get("label") == "literal")
        
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "num_variables": self.num_vars,
            "used_variables": sorted(self._used_variables),
            "num_variable_nodes": num_var_nodes,
            "num_literal_nodes": num_literal_nodes,
            "num_gates": self.size,
            "size": self.size,
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph),
            "wl_hash": self.wl_hash()
        }
    
    def __repr__(self) -> str:
        """String representation of the FormulaGraph."""
        return (f"FormulaGraph(num_vars={self.num_vars}, size={self.size}, "
                f"nodes={self.num_nodes}, edges={self.num_edges})")
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        if self.is_empty:
            return "Empty FormulaGraph"
        return (f"FormulaGraph with {self.num_vars} variables, {self.size} gates, "
                f"{self.num_nodes} nodes, {self.num_edges} edges")


# Backward compatibility: Static methods for direct use
def create_from_definition(definition: List[int]) -> FormulaGraph:
    """
    Create a FormulaGraph from a definition.
    
    Args:
        definition: List of gate integers
        
    Returns:
        A new FormulaGraph instance
    """
    return FormulaGraph(definition)


def add_gate_to_graph(graph: nx.Graph, gate: int) -> None:
    """
    Legacy function: Add a gate to a NetworkX graph directly.
    
    Args:
        graph: NetworkX graph to modify
        gate: Gate integer to add
    """
    # Create a temporary FormulaGraph wrapper
    fg = FormulaGraph()
    fg.graph = graph
    fg.add_gate(gate)


def del_gate_from_graph(graph: nx.Graph, gate: int) -> None:
    """
    Legacy function: Remove a gate from a NetworkX graph directly.
    
    Args:
        graph: NetworkX graph to modify
        gate: Gate integer to remove
    """
    if gate in graph:
        graph.remove_node(gate)


def definition_to_graph(definition: list) -> nx.Graph:
    """
    Legacy function: Convert a definition to a NetworkX graph directly.
    
    Args:
        definition: List of gate integers
        
    Returns:
        NetworkX graph representation
    """
    fg = FormulaGraph(definition)
    return fg.graph