import numpy as np
from ..models import TrajectoryQueueMessage
from ..models.api import TrajectoryProcessingContext
from ..env.formula_graph import FormulaGraph
from . import WarehouseAgent

class TrajectoryProcessor:
    def __init__(self, warehouse: WarehouseAgent, **config):
        self.warehouse = warehouse # TODO
        self.config = config
        # Default number of iterations for WL hash
        self.hash_iterations: int = config.get("iterations", 5)  
        # Default granularity for Q-value discretization
        self.traj_granularity: int = config.get("granularity", 20)  
        # Default number of summits to consider
        # `traj_num_summits` should be no more than `traj_granularity`
        self.traj_num_summits: int = config.get("num_summits", 5)

    def retrieve_definition(self, formula_id: str | None) -> list[int]:
        """
        Retrieves the formula definition from the warehouse.
        
        Parameters:
            formula_id (str): The identifier of the formula to retrieve.
        
        Returns:
            list: The formula definition.
        """
        if formula_id is None:
            return []
        
        # Handle non-existent formulas (like initial empty formulas)
        try:
            return self.warehouse.get_formula_definition(formula_id)
        except Exception as e:
            # Check for specific HTTP errors - avoid catching all exceptions
            error_str = str(e).lower()
            if "404" in error_str or "not found" in error_str:
                return []  # Empty formula definition for missing formulas
            # Log and re-raise unexpected errors
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Unexpected error retrieving formula {formula_id}: {e}")
            raise  # Re-raise other errors


    def isomorphic_to(self, formula_graph: FormulaGraph, wl_hash: str | None = None) -> str | None:
        """
        Returns the ID of an existing formula in the warehouse that is isomorphic to the given formula graph. If no isomorphic formula is found, it returns `None`. 
        This method uses the Weisfeiler-Lehman hash to determine if the formula is a duplicate. If the `wl_hash` is provided, it will be used to check for isomorphism; otherwise, the hash will be computed from the `formula_graph`.

        :parameters:
            formula_graph (FormulaGraph): The graph representation of the formula to check for isomorphism.
            wl_hash (str | None): The precomputed Weisfeiler-Lehman hash of the formula graph. If `None`, it will be computed.
        :returns:
            str | None: The ID of the isomorphic formula if found, otherwise `None`.
        """
        # Implementation of the duplicate check using Weisfeiler-Lehman hash
        if wl_hash is None:
            wl_hash = formula_graph.wl_hash(iterations=self.hash_iterations)
        
        isomorphic_ids = self.warehouse.get_likely_isomorphic(wl_hash)
        
        # Check each candidate for isomorphism
        for fid in isomorphic_ids:
            try:
                fdef = self.retrieve_definition(fid)
                fg = FormulaGraph(fdef)
                
                if formula_graph.is_isomorphic_to(fg):
                    return fid
                    
            except Exception:
                continue
        
        return None
    
    def reconstruct_base_formula(self, prefix_traj: list[tuple[int, int, float]]) -> FormulaGraph:
        """Reconstruct base formula from prefix trajectory instead of warehouse retrieval.
        
        Args:
            prefix_traj: Complete trajectory steps as tuples (token_type, token_literals, cur_avgQ)
            
        Returns:
            FormulaGraph: The reconstructed base formula graph
        """
        # Start with empty formula
        fg = FormulaGraph([])
        
        # Apply each step in the prefix trajectory to build the base formula
        for step in prefix_traj:
            token_type = step[0]
            token_literals = step[1]
            
            if token_type == 0:  # ADD
                fg.add_gate(token_literals)
            elif token_type == 1:  # DEL
                fg.remove_gate(token_literals)
            elif token_type == 2:  # EOS
                pass  # End of sequence, no action needed
            else:
                raise ValueError(f"Unknown token type in prefix trajectory: {token_type}")
        
        return fg
    
    def check_base_formula_exists(self, base_formula: FormulaGraph) -> tuple[bool, str | None]:
        """Check if the base formula already exists in the database.
        
        Args:
            base_formula: The base formula graph to check
            
        Returns:
            tuple: (exists, formula_id) where exists is bool and formula_id is the existing ID or None
        """
        wl_hash = base_formula.wl_hash(iterations=self.hash_iterations)
        existing_id = self.isomorphic_to(base_formula, wl_hash)
        return (existing_id is not None, existing_id)
    
    def process_trajectory(self, context: TrajectoryProcessingContext) -> dict:
        """V2 trajectory processing with embedded formula reconstruction.
        
        This method eliminates the linked list structure by using prefix_traj for 
        base formula reconstruction and processing only suffix_traj for new steps.
        
        Args:
            context: TrajectoryProcessingContext containing prefix_traj and suffix_traj
            
        Returns:
            dict: Processing results including new formulas and metadata
        """
        # Reconstruct base formula from prefix trajectory (no warehouse dependency)
        base_formula = self.reconstruct_base_formula(context.prefix_traj)
        
        # Check if base formula already exists in database
        base_exists, base_formula_id = self.check_base_formula_exists(base_formula)
        
        # Extract avgQ values from suffix trajectory for processing
        suffix_steps = context.suffix_traj
        if not suffix_steps:
            return {
                "new_formulas": [],
                "evo_path": [base_formula_id] if base_formula_id else [],
                "base_formula_exists": base_exists,
                "processed_formulas": 0,
                "new_nodes_created": 0
            }
        
        avgQs = [step[2] for step in suffix_steps]  # step[2] is cur_avgQ
        ns = self.traj_num_summits
        granu = self.traj_granularity
        pieces: list[tuple[int, int]] = []
        
        # Apply trajectory segmentation logic (same as V1 but only for suffix)
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
            summit = int(np.argmax(avgQs)) if avgQs else 0
            pieces = [(0, summit), (summit, len(avgQs))]
        
        # Start with reconstructed base formula - use proper deep copy
        if hasattr(base_formula, 'copy'):
            fg = base_formula.copy()
        elif hasattr(base_formula, 'gates'):
            # Deep copy the gates to avoid modifying the original
            fg = FormulaGraph(list(base_formula.gates))
        else:
            raise ValueError("FormulaGraph object missing expected attributes")
        evo_path: list[str] = [base_formula_id] if base_formula_id else []
        
        # V2: Save complete trajectory (prefix + suffix) once at the beginning
        # Both prefix_traj and suffix_traj are already in tuple format
        complete_trajectory_steps = list(context.prefix_traj) + list(suffix_steps)
        
        # Post the complete trajectory to get traj_id
        complete_traj_id = self.warehouse.post_trajectory(steps=complete_trajectory_steps)
        prefix_length = len(context.prefix_traj)
        
        new_formulas = []
        new_nodes_created = 0
        
        # Process each piece of the suffix trajectory
        for i, piece in enumerate(pieces):
            s, t = piece
            
            # OPTIMIZATION: Save formula state before applying trajectory slice
            nodes_before = fg.num_nodes
            edges_before = fg.num_edges
            gates_before = set(fg.gates)  # Create a copy of the gates set
            
            # Apply suffix trajectory steps to the formula graph
            slice_steps = suffix_steps[s:t]
            for step in slice_steps:
                token_type = step[0]
                token_literals = step[1]
                
                if token_type == 0:  # ADD
                    fg.add_gate(token_literals)
                elif token_type == 1:  # DEL
                    fg.remove_gate(token_literals)
                elif token_type == 2:  # EOS
                    pass
                else:
                    raise ValueError(f"Unknown token type: {token_type}")
            
            # OPTIMIZATION: Check if formula structure unchanged (no-effect slice)
            nodes_after = fg.num_nodes
            edges_after = fg.num_edges
            gates_after = set(fg.gates)
            
            if (nodes_before == nodes_after and 
                edges_before == edges_after and 
                gates_before == gates_after):
                # No effect - skip expensive isomorphism check
                continue
            
            # Check if the formula graph is a duplicate
            wl_hash = fg.wl_hash(iterations=self.hash_iterations)
            
            if (fid := self.isomorphic_to(fg, wl_hash)) is not None:
                evo_path.append(fid)
                continue  # Skip storing if it's a duplicate
            
            # If not a duplicate, store the formula with correct traj_slice
            # V2: Use the complete trajectory ID and calculate correct slice position
            traj_slice = prefix_length + t - 1  # Position in complete trajectory
            
            # Get avgQ from the last step in this piece
            final_avgQ = suffix_steps[t-1][2] if t > 0 else 0.0  # step[2] is cur_avgQ
            
            formula_data = {
                "avgQ": final_avgQ,
                "wl_hash": wl_hash,
                "num_vars": context.processing_metadata.get("num_vars", 4),  # Default or from context
                "width": context.processing_metadata.get("width", 3),      # Default or from context
                "size": fg.size, 
            }
            
            # Post a new evolution graph node to the warehouse (V2 integrated approach)
            formula_data["node_id"] = self.warehouse.post_evolution_graph_node(
                avgQ=formula_data["avgQ"],
                num_vars=formula_data["num_vars"],
                width=formula_data["width"],
                size=formula_data["size"],
                wl_hash=wl_hash,
                traj_id=complete_traj_id,  # Use the complete trajectory ID
                traj_slice=traj_slice,     # Correct position in complete trajectory
            )
            
            # Post the likely isomorphic formula to the warehouse
            self.warehouse.post_likely_isomorphic(
                wl_hash=wl_hash,
                formula_id=formula_data["node_id"],
            )
            
            # Track new formulas and prepare for next iteration
            new_formulas.append(formula_data)
            new_nodes_created += 1
            evo_path.append(formula_data["node_id"])
        
        # Save the evolution path to the warehouse
        evo_path = [fid for fid in evo_path if fid]
        if len(evo_path) > 1:  # Only save if there's an actual path
            self.warehouse.post_evolution_graph_path(path=evo_path)
        
        return {
            "new_formulas": new_formulas,
            "evo_path": evo_path,
            "base_formula_exists": base_exists,
            "processed_formulas": len(pieces),
            "new_nodes_created": new_nodes_created
        }