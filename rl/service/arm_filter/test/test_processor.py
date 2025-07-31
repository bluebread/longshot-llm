import pytest
import httpx
import functools
from networkx import weisfeiler_lehman_graph_hash

from processor import TrajectoryProcessor

host = "localhost"
port = 8000
warehouse_url = f"http://{host}:{port}"
hash_iters = 3

@pytest.fixture
def processor():
    with httpx.Client(base_url=warehouse_url) as warehouse:
        processor = TrajectoryProcessor(warehouse, iterations=hash_iters)
        yield processor

class TestTrajectoryProcessor:
    """Test suite for the trajectory processing functionality."""
    def encode_literals(self, pos: list[int], neg: list[int]) -> str:
        """
        Encodes a list of integers to a base64 string.
        """
        pos = [(1 << i) for i in pos]
        neg = [(1 << i) for i in neg]
        p = functools.reduce(lambda x, y: x | y, pos, 0)
        n = functools.reduce(lambda x, y: x | y, neg, 0)
        return p | (n << 32)
    
    def save_formula_example(self, warehouse: httpx.Client, definition: list, wl_hash: str) -> str:
        """
        Saves a formula to the warehouse and returns its ID.
        """
        response = warehouse.post("/trajectory", json={
            "base_formula_id": None,
            "steps": [
                {
                    "token_type": 0, 
                    "token_literals": d,
                    "reward": 0.0
                } 
                for d in definition
            ]
        })
        
        assert response.status_code == 201, f"Failed to save formula: {response.text}"

        traj_id = response.json()["id"]
        response = warehouse.post("/formula/info", json={
            "base_formula_id": None,
            "trajectory_id": traj_id,
            "avgQ": 0.0,
            "wl_hash": wl_hash,
            "num_vars": len(definition),
            "width": 1,
            "size": len(definition),
            "node_id": ""
        })
        
        assert response.status_code == 201, f"Failed to create formula: {response.text}"
        
        return response.json()["id"], traj_id

    def del_formula_example(self, warehouse: httpx.Client, fid: str, tid: str):
        """
        Deletes a formula from the warehouse.
        """
        response = warehouse.delete(f"/formula/info", params={"id": fid})
        assert response.status_code == 200, f"Failed to delete formula: {response.text}"
        
        response = warehouse.delete(f"/trajectory", params={"id": tid})
        assert response.status_code == 200, f"Failed to delete trajectory: {response.text}"

    def test_isomorphic_to_success(self, processor: TrajectoryProcessor):
        """
        Test the isomorphic_to endpoint with a valid graph.
        """
        # Define two example isomorphic formulas
        fdef1 = [
            ((0,2),()), ((1,),()), ((2,),(1,)), ((0,1),()), ((),(1,2)), ((2,),())
        ]
        fdef2 = [
            ((),(1,2)), ((0,),()), ((),(0,1)), ((0,),(2,)), ((1,),(0,)), ((),(1,))
        ]
        fdef1 = [self.encode_literals(p, n) for p, n in fdef1]
        fdef2 = [self.encode_literals(p, n) for p, n in fdef2]
        fg1 = TrajectoryProcessor.definition_to_graph(fdef1)
        fg2 = TrajectoryProcessor.definition_to_graph(fdef2)
        wl1 = weisfeiler_lehman_graph_hash(fg1, iterations=hash_iters, node_attr="label")
        wl2 = weisfeiler_lehman_graph_hash(fg2, iterations=hash_iters, node_attr="label")
        assert wl1 == wl2, "WL hashes should match for isomorphic graphs"
        
        fid2, tid2 = self.save_formula_example(processor.warehouse, fdef2, wl2)
        response = processor.warehouse.post("/formula/likely_isomorphic", json={
            "wl_hash": wl2,
            "formula_id": fid2,
        })
        
        assert response.status_code == 201, f"Failed to check isomorphism: {response.text}"
        assert processor.isomorphic_to(fg1) == fid2
        
        self.del_formula_example(processor.warehouse, fid2, tid2)
        response = processor.warehouse.delete("/formula/likely_isomorphic", params={"wl_hash": wl2})
        assert response.status_code == 200, f"Failed to delete isomorphic formula: {response.text}"
    
    def test_isomorphic_to_not_found(self, processor: TrajectoryProcessor):
        """
        Test that isomorphic_to returns None for non-isomorphic graphs.
        """
        # Define two non-isomorphic formulas
        fdef1 = [
            ((0,2),()), ((1,),()), ((2,),(1,)), ((0,1),()), ((),(1,2)), ((2,),())
        ]
        fdef2 = [
            ((),(1,2)), ((0,),()), ((),(0,1)), ((0,),(2,)), ((1,),(0,))
        ]
        fdef1 = [self.encode_literals(p, n) for p, n in fdef1]
        fdef2 = [self.encode_literals(p, n) for p, n in fdef2]
        fg1 = TrajectoryProcessor.definition_to_graph(fdef1)
        fg2 = TrajectoryProcessor.definition_to_graph(fdef2)
        wl1 = weisfeiler_lehman_graph_hash(fg1, iterations=hash_iters, node_attr="label")
        wl2 = weisfeiler_lehman_graph_hash(fg2, iterations=hash_iters, node_attr="label")
        assert wl1 != wl2, "WL hashes should not match for non-isomorphic graphs"
        
        fid2, tid2 = self.save_formula_example(processor.warehouse, fdef2, wl2)
        response = processor.warehouse.post("/formula/likely_isomorphic", json={
            "wl_hash": wl2,
            "formula_id": fid2,
        })
        
        assert response.status_code == 201, f"Failed to check isomorphism: {response.text}"
        assert processor.isomorphic_to(fg1) is None
        
        self.del_formula_example(processor.warehouse, fid2, tid2)
        response = processor.warehouse.delete("/formula/likely_isomorphic", params={"wl_hash": wl2})
        assert response.status_code == 200, f"Failed to delete isomorphic formula: {response.text}"