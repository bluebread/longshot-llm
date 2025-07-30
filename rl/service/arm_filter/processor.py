from functools import reduce
import itertools
import httpx
import networkx as nx
import numpy as np
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from networkx.algorithms.isomorphism import vf2pp_is_isomorphic

from longshot.circuit import DNF
from models import TrajectoryMessage

class TrajectoryProcessor:
    def __init__(self, warehouse: httpx.Client, **config):
        self.warehouse = warehouse
        self.config = config
        # Default number of iterations for WL hash
        self.hash_iterations: int = config.get("iterations", 5)  
        # Default granularity for Q-value discretization
        self.traj_granularity: int = config.get("granularity", 20)  
        # Default number of summits to consider
        # `traj_num_summits` should be no more than `traj_granularity`
        self.traj_num_summits: int = config.get("num_summits", 5)  

    def retrieve_definition(self, formula_id: str) -> dict:
        """
        Retrieves the formula definition from the warehouse.
        
        Parameters:
            warehouse (httpx.Client): The HTTP client to interact with the warehouse.
            formula_id (str): The identifier of the formula to retrieve.
        
        Returns:
            dict: The formula definition.
        """
        response = self.warehouse.get(f"/formula/definition", params={"id": formula_id})
        
        if response.status_code != 200:
            raise ValueError(f"Failed to retrieve formula definition for ID {formula_id}: {response.text}")
        
        return response.json()['definition']

    @staticmethod
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
        
        edges = [(f"+x{i}", gate) for i in range(32) if (i & pl) != 0]
        edges += [(f"-x{i}", gate) for i in range(32) if (i & nl) != 0]
        
        graph.add_edges_from(edges)

    @staticmethod
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
        

    @staticmethod
    def definition_to_graph(definition: dict) -> nx.Graph:
        """
        Converts a formula definition to a networkx graph.
        
        Parameters:
            definition (dict): The formula definition containing variables and their connections.
        
        Returns:
            nx.Graph: A graph representation of the formula.
        """
        vars: int = reduce(lambda x, y: x | y, definition, initial=0)
        pos_vars = vars & 0xFFFFFFFF
        neg_vars = (vars >> 32) & 0xFFFFFFFF
        vars = pos_vars | neg_vars
        
        if vars == 0:
            return nx.Graph()
        
        num_vars = vars.bit_length() - 1
        formula_graph = nx.Graph()
        
        vars = [(1 << i) for i in range(num_vars) if (vars & (1 << i)) != 0]
        pos_vars = [(1 << i) for i in range(num_vars) if (pos_vars & (1 << i)) != 0]
        neg_vars = [(1 << i) for i in range(num_vars) if (neg_vars & (1 << i)) != 0]
        
        var_nodes = [f"x{i}" for i in vars]
        pos_nodes = [f"+x{i}" for i in pos_vars]
        neg_nodes = [f"-x{i}" for i in neg_vars]

        formula_graph.add_nodes_from(var_nodes, label="variable")
        formula_graph.add_nodes_from(pos_nodes + neg_nodes, label="literal")
        formula_graph.add_edges_from([(f"+x{i}", f"x{i}") for i in pos_vars])
        formula_graph.add_edges_from([(f"-x{i}", f"x{i}") for i in neg_vars])
        
        for gate in definition:
            TrajectoryProcessor.add_gate_to_graph(formula_graph, gate)

        return formula_graph

    def check_if_duplicate(self, formula_graph: nx.Graph) -> bool:
        """
        Checks if a given formula's graph is isomorphic to any existing formula in the warehouse.
        This method uses the Weisfeiler-Lehman hash to determine if the formula is a duplicate.

        Parameters:
            formula_graph (networkx.Graph): The graph representation of the formula to check for isomorphism.

        Returns:
            bool: True if the formula is a duplicate, False otherwise.
        """
        # Implementation of the duplicate check using Weisfeiler-Lehman hash
        formula_hash = weisfeiler_lehman_graph_hash(formula_graph, iterations=self.hash_iterations, node_attr="label")
        response = self.warehouse.get("/formula/likely_isomorphic", params={"wl_hash": formula_hash})
        
        if response.status_code == 404:
            return False  # No existing formula with this hash, so it's not a duplicate
        elif response.status_code != 200:
            raise ValueError(f"Failed to check for isomorphic formulas: {response.text}")
        
        isomorphic_ids: list[str] = response.json().get("isomorphic_ids", [])
        
        for fid in isomorphic_ids:
            fdef = self.retrieve_definition(fid)
            fg = self.definition_to_graph(fdef)
            
            if vf2pp_is_isomorphic(formula_graph, fg, node_label="label"):
                return True
            
        return False
    
    def process_trajectory(self, msg: TrajectoryMessage) -> None:
        avgQs = [step.avgQ for step in msg.trajectory.steps]
        
        if len(avgQs) == 0:
            return
        
        ns = self.traj_num_summits
        granu = self.traj_granularity
        pieces: list[tuple[int,int]] = []
        
        if len(avgQs) > ns:
            top_ns = sorted(list(np.argpartition(avgQs, ns)[-ns:])) + [len(avgQs)]
            s = 0
            
            for t in top_ns:
                l = t - s
                pn = (l + granu - 1) // granu
                
                for i in range(pn):
                    pieces.append((s + i * granu, min(s + (i + 1) * granu, t)))
                
                s = t
        else:
            summit = np.argmax(avgQs)
            pieces = [(0, summit), (summit, len(avgQs))]
        
        base_fdef = self.retrieve_definition(msg.trajectory.base_formula_id)
        fg = self.definition_to_graph(base_fdef)
        prev_formula_id: str = msg.trajectory.base_formula_id
        
        for s, t in pieces:
            # Process each piece of the trajectory
            # Each piece is [s, t) and represents a segment of the trajectory,
            # where `s` is the start index and `t` is the end index.
            response = self.warehouse.post("/trajectory", json={
                "base_formula_id": prev_formula_id,
                "steps": [
                    {
                        "token_type": step.token_type,
                        "token_literal": step.token_literals,
                        "reward": step.reward,
                    }
                    for step in msg.trajectory.steps[s:t]
                ]
            })
            
            if response.status_code != 200:
                raise ValueError(f"Failed to process trajectory piece {s}:{t}: {response.text}")
            
            traj_id = response.json().get("id")
            
            for step in msg.trajectory.steps[s:t]:
                if step.token_type == 'ADD':
                    self.add_gate_to_graph(fg, step.token_literals)
                elif step.token_type == 'DEL':
                    self.del_gate_from_graph(fg, step.token_literals)
                elif step.token_type == 'EOS':
                    pass
                else:
                    raise ValueError(f"Unknown token type: {step.token_type}")
                
            wl_hash = weisfeiler_lehman_graph_hash(fg, iterations=self.hash_iterations, node_attr="label")
            
            response = self.warehouse.post("/formula", json={
                "base_formula_id": prev_formula_id,
                "trajectory_id": traj_id,
                "avgQ": msg.trajectory.steps[t-1].avgQ,
                "wl_hash": wl_hash,
                "num_vars": msg.num_vars,
                "width": msg.width,
                "size": msg.size,
                "node_id": "" # Unused in this context, but required by the API
            })
            
            if response.status_code != 200:
                raise ValueError(f"Failed to store formula for trajectory piece {s}:{t}: {response.text}")
            
            prev_formula_id = response.json().get("id")
        