"""
ArmRanker module for filtering and selecting the best arms (formulas).
"""

import math
import random
from functools import reduce
from collections import namedtuple
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from . import WarehouseAgent
from ..utils import to_lambda


class Arm(BaseModel):
    """Represents an arm (formula) with its properties."""
    avgQ: float
    visited_counter: int
    in_degree: int
    out_degree: int


class ArmRanker:
    """
    Local class responsible for filtering and selecting the best arms (formulas) 
    based on the trajectories and the evolution graph.
    """
    
    def __init__(self, warehouse, **config):
        """
        Initialize the ArmRanker with a warehouse agent and configuration.
        
        Args:
            warehouse: WarehouseAgent or AsyncWarehouseAgent instance for database operations
            **config: Configuration parameters including:
                - eps: Small value to avoid division by zero (default: 0.1)
                - wq: Weight for the average Q value (default: 1.0) 
                - wvc: Weight for the visited counter (default: 2.0)
        """
        self.warehouse = warehouse
        
        # Set default configuration
        default_config = {
            "eps": 0.1,  # Small value to avoid division by zero
            "wq": 1.0,   # Weight for the average Q value
            "wvc": 2.0,  # Weight for the visited counter
        }
        default_config.update(config)
        self.config = namedtuple("Config", default_config.keys())(**default_config)
    
    def score(self, arm: Arm, num_vars: int, total_visited: int) -> float:
        """
        Scores a single arm (formula) based on its properties.

        Args:
            arm: An Arm object representing a formula with its properties
            num_vars: The number of variables in the formula, used for normalization
            total_visited: The total number of visits across all arms, used for normalization
            
        Returns:
            A float score representing the performance and potential of the arm
        """
        lmbd = to_lambda(arm.avgQ, n=num_vars, eps=self.config.eps)

        # Compute the score using a weighted formula
        score = (
            self.config.wq * (arm.avgQ + (lmbd or 0))
            + self.config.wvc * math.sqrt(math.log(total_visited) / (arm.visited_counter + 1))
        )
        return score

    def topk_arms(
        self, 
        num_vars: int, 
        width: int, 
        k: int, 
        size_constraint: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Filter and return the top-K arms based on the provided criteria.
        
        Args:
            num_vars: The number of variables in the formula
            width: The width of the formula  
            k: The number of top arms to return
            size_constraint: The maximum size of the formula (optional)
            
        Returns:
            List of dictionaries containing formula_id and definition for top-K arms
        """
        params = {
            "num_vars": num_vars,
            "width": width,
            "size_constraint": size_constraint,
        }
        
        # Download nodes and hypernodes
        nodes = self.warehouse.download_nodes(**params)
        hypernodes = self.warehouse.download_hypernodes(**params)
        
        # Process the downloaded data
        nmap = {node.pop("node_id"): node for node in nodes}
        hmap = {hn["hnid"]: hn["nodes"] for hn in hypernodes}
        hset = set(reduce(lambda x, y: x + y, hmap.values(), []))
        
        # Create arm mapping for hypernodes
        # Note: visited_counter doesn't exist in V2, so we use in_degree + out_degree as a proxy
        armmap = {
            hnid: Arm(**{
                "avgQ": nmap[nodes[0]]['avgQ'],
                "visited_counter": sum([nmap[nid]['in_degree'] + nmap[nid]['out_degree'] + 1 for nid in nodes]),
                "in_degree": sum([nmap[nid]['in_degree'] for nid in nodes]),
                "out_degree": sum([nmap[nid]['out_degree'] for nid in nodes]),
            })
            for hnid, nodes in hmap.items()
        }
        
        # Add individual nodes that are not part of hypernodes
        # Use in_degree + out_degree + 1 as proxy for visited_counter
        armmap.update({
            nid: Arm(**{
                **node,
                "visited_counter": node['in_degree'] + node['out_degree'] + 1
            }) for nid, node in nmap.items() if nid not in hset
        })
        
        # Calculate total visited count and rank arms
        total_visited = sum(arm.visited_counter for arm in armmap.values())
        ranking = sorted([
            (
                self.score(arm, num_vars, total_visited), 
                aid,
            )
            for aid, arm in armmap.items()
        ], reverse=True)
        
        # Select top-K arms with random selection from hypernodes
        rng = random.Random()
        selected_fids = [
            rng.choice(hmap[aid]) if aid in hmap else aid 
            for _, aid in ranking[:k]
        ]
        
        # Get formula definitions
        definitions = [
            self.warehouse.get_formula_definition(fid)
            for fid in selected_fids
        ]

        # Build result
        selected_arms = []
        for fid, definition in zip(selected_fids, definitions):
            selected_arms.append({
                "node_id": fid,
                "definition": definition,
            })

        return selected_arms


# Convenience functions that can be used independently
def score_arm(arm: Arm, num_vars: int, total_visited: int, config: Optional[Dict] = None) -> float:
    """
    Standalone function to score a single arm.
    
    Args:
        arm: An Arm object representing a formula with its properties
        num_vars: The number of variables in the formula
        total_visited: The total number of visits across all arms
        config: Optional configuration dict with eps, wq, wvc values
        
    Returns:
        A float score representing the performance and potential of the arm
    """
    default_config = {"eps": 0.1, "wq": 1.0, "wvc": 2.0}
    if config:
        default_config.update(config)
    
    lmbd = to_lambda(arm.avgQ, n=num_vars, eps=default_config["eps"])
    
    score = (
        default_config["wq"] * (arm.avgQ + (lmbd or 0))
        + default_config["wvc"] * math.sqrt(math.log(total_visited) / (arm.visited_counter + 1))
    )
    return score