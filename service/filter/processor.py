from functools import reduce
import httpx
import networkx as nx
import numpy as np
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash
from networkx.algorithms.isomorphism import vf2pp_is_isomorphic
from typing import Any
from longshot.models import TrajectoryQueueMessage, TrajectoryInfoStep

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

    def retrieve_definition(self, formula_id: str | None) -> list:
        """
        Retrieves the formula definition from the warehouse.
        
        Parameters:
            formula_id (str): The identifier of the formula to retrieve.
        
        Returns:
            list: The formula definition.
        """
        if formula_id is None:
            return []
        
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
            TrajectoryProcessor.add_gate_to_graph(formula_graph, gate)

        return formula_graph

    def isomorphic_to(self, formula_graph: nx.Graph, wl_hash: str | None = None) -> str | None:
        """
        Returns the ID of an existing formula in the warehouse that is isomorphic to the given formula graph. If no isomorphic formula is found, it returns `None`. 
        This method uses the Weisfeiler-Lehman hash to determine if the formula is a duplicate. If the `wl_hash` is provided, it will be used to check for isomorphism; otherwise, the hash will be computed from the `formula_graph`.

        :parameters:
            formula_graph (networkx.Graph): The graph representation of the formula to check for isomorphism.
            wl_hash (str | None): The precomputed Weisfeiler-Lehman hash of the formula graph. If `None`, it will be computed.
        :returns:
            str | None: The ID of the isomorphic formula if found, otherwise `None`.
        """
        # Implementation of the duplicate check using Weisfeiler-Lehman hash
        if wl_hash is None:
            wl_hash = weisfeiler_lehman_graph_hash(formula_graph, iterations=self.hash_iterations, node_attr="label")
        response = self.warehouse.get("/formula/likely_isomorphic", params={"wl_hash": wl_hash})
        
        if response.status_code == 404:
            return None  # No existing formula with this hash, so it's not a duplicate
        elif response.status_code != 200:
            raise ValueError(f"Failed to check for isomorphic formulas: {response.text}")
        
        isomorphic_ids: list[str] = response.json().get("isomorphic_ids", [])
        
        for fid in isomorphic_ids:
            fdef = self.retrieve_definition(fid)
            fg = self.definition_to_graph(fdef)
            
            if vf2pp_is_isomorphic(formula_graph, fg, node_label="label"):
                return fid  
            
        return None
    
    def process_trajectory(self, msg: TrajectoryQueueMessage) -> dict[str, Any]:
        """
        Processes a single trajectory and updates the evolution graph accordingly. This method is called when a new trajectory is received from the trajectory queue and would try to break down the trajectory into smaller parts if necessary. The result is then saved to the warehouse and also returned as a list of new formulas' information.
        
        :param msg: The trajectory data to process in the message schema (JSON) defined in Trajectory Queue.
        :return: A dictionary containing new formulas and the evolution path.
        """
        avgQs = [step.avgQ for step in msg.trajectory.steps]
        
        if len(avgQs) == 0:
            return {"new_formulas": [], "evo_path": []}  # No steps to process

        ns = self.traj_num_summits
        granu = self.traj_granularity
        pieces: list[tuple[int,int]] = []
        
        if len(avgQs) > ns:
            top_ns = sorted((np.argpartition(avgQs, ns)[-ns:] + 1).tolist())
            
            if top_ns[-1] != len(avgQs):
                top_ns.append(len(avgQs))
            s = 0
            
            for t in top_ns:
                l = t - s
                pn = (l + granu - 1) // granu
                
                for i in range(pn):
                    pieces.append((s + i * granu, min(s + (i + 1) * granu, t)))
                
                s = t
        else:
            summit = int(np.argmax(avgQs))
            pieces = [(0, summit), (summit, len(avgQs))]
        
        base_fdef = self.retrieve_definition(msg.trajectory.base_formula_id)
        fg = self.definition_to_graph(base_fdef)
        prev_formula_id: str = msg.trajectory.base_formula_id
        new_formulas: list[dict] = []
        steps_buffer: list[TrajectoryInfoStep] = []
        evo_path: list[str] = [msg.trajectory.base_formula_id]
        cur_size = msg.base_size

        for i, piece in enumerate(pieces):
            # Process each piece of the trajectory
            s, t = piece
            # Each piece is [s, t) and represents a segment of the trajectory,
            # where `s` is the start index and `t` is the end index.
            
            # In turn add or delete gates in the formula graph based on the trajectory steps
            for step in msg.trajectory.steps[s:t]:
                if step.token_type == 'ADD':
                    self.add_gate_to_graph(fg, step.token_literals)
                    cur_size += 1
                elif step.token_type == 'DEL':
                    self.del_gate_from_graph(fg, step.token_literals)
                    cur_size -= 1
                elif step.token_type == 'EOS':
                    pass
                else:
                    raise ValueError(f"Unknown token type: {step.token_type}")
            
            # Check if the formula graph is a duplicate
            wl_hash = weisfeiler_lehman_graph_hash(
                fg, 
                node_attr="label",
                iterations=self.hash_iterations, 
            )
            
            if (fid := self.isomorphic_to(fg, wl_hash)) is not None:
                steps_buffer.extend(msg.trajectory.steps[s:t])
                evo_path.append(fid)

                continue  # Skip storing if it's a duplicate
            
            # If not a duplicate, store the trajectory and formula
            response = self.warehouse.post("/trajectory", json={
                "base_formula_id": prev_formula_id,
                "steps": [
                    {
                        "token_type": 0 if step.token_type == 'ADD' else 1,
                        "token_literals": step.token_literals,
                        "reward": step.reward,
                    }
                    for step in steps_buffer + msg.trajectory.steps[s:t]
                ]
            })
            steps_buffer.clear()  # Clear the buffer after processing
            
            if response.status_code != 201:
                raise ValueError(f"Failed to process trajectory piece {s}:{t}: {response.text}")
            
            traj_id = response.json().get("id")
            
            formula_data = {
                "base_formula_id": prev_formula_id,
                "trajectory_id": traj_id,
                "avgQ": msg.trajectory.steps[t-1].avgQ,
                "wl_hash": wl_hash,
                "num_vars": msg.num_vars,
                "width": msg.width,
                "size": cur_size,
                "node_id": "" # Unused in this context, but required by the API
            }
            response = self.warehouse.post("/formula/info", json=formula_data)

            if response.status_code != 201:
                raise ValueError(f"Failed to store formula for trajectory piece {s}:{t}: {response.text}")
            
            # Store the formula ID in the isomorphism hash table
            formula_data["id"] = response.json().get("id")
            response = self.warehouse.post("/formula/likely_isomorphic", json={
                "wl_hash": wl_hash,
                "formula_id": formula_data["id"],
            })
            
            if response.status_code != 201:
                raise ValueError(f"Failed to check for isomorphic formulas after storing: {response.text}")
            
            # Prepare for the next iteration
            evo_path.append(formula_data["id"])
            prev_formula_id = formula_data["id"]
            del formula_data["node_id"]
            new_formulas.append(formula_data)

        # | Attribute | Type    | Description                                   |
        # | :------: | :------: | --------------------------------------------- |
        # | new_formulas | `list[dict]`  | A list of dictionaries representing the new formulas' information, each containing the formula ID, base formula ID, trajectory ID, average-case deterministic query complexity, number of variables, width, size and wl-hash value.  |
        # | evo_path | `list[str]` | A list of formula IDs representing the evolution path of the formulas in the trajectory. This is used to track the evolution of formulas over time. |
        return {"new_formulas": new_formulas, "evo_path": evo_path}