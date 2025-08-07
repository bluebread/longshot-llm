import pytest
import httpx
import functools
import pprint
from datetime import datetime
from networkx import weisfeiler_lehman_graph_hash

from processor import TrajectoryProcessor
from longshot.models import TrajectoryQueueMessage
from longshot.agent import WarehouseAgent

host = "localhost"
port = 8000
warehouse_url = f"http://{host}:{port}"
hash_iters = 3

@pytest.fixture
def processor():
    with WarehouseAgent(host, port) as warehouse:
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
        
        fid2, tid2 = self.save_formula_example(processor.warehouse._client, fdef2, wl2)
        response = processor.warehouse._client.post("/formula/likely_isomorphic", json={
            "wl_hash": wl2,
            "formula_id": fid2,
        })
        
        assert response.status_code == 201, f"Failed to check isomorphism: {response.text}"
        assert processor.isomorphic_to(fg1) == fid2
        
        self.del_formula_example(processor.warehouse._client, fid2, tid2)
        response = processor.warehouse._client.delete("/formula/likely_isomorphic", params={"wl_hash": wl2})
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
        
        fid2, tid2 = self.save_formula_example(processor.warehouse._client, fdef2, wl2)
        response = processor.warehouse._client.post("/formula/likely_isomorphic", json={
            "wl_hash": wl2,
            "formula_id": fid2,
        })
        
        assert response.status_code == 201, f"Failed to check isomorphism: {response.text}"
        assert processor.isomorphic_to(fg1) is None
        
        self.del_formula_example(processor.warehouse._client, fid2, tid2)
        response = processor.warehouse._client.delete("/formula/likely_isomorphic", params={"wl_hash": wl2})
        assert response.status_code == 200, f"Failed to delete isomorphic formula: {response.text}"
        
    def test_process_trajectory(self, processor: TrajectoryProcessor):
        """
        Test the process_trajectory method with a valid trajectory.
        """
        processor.traj_granularity = 2
        processor.traj_num_summits = 2
        
        ls = [((0,2),()), ((1,),()), ((2,),(1,)), ((0,1),()), ((),(1,2)), ((2,),())]
        ls = [self.encode_literals(p, n) for p, n in ls]
        qs = [0, 1, 2, 0, 0, 3]
        rs = [0] * len(ls)
        assert len(ls) == len(qs) and len(qs) == len(rs), "Lists must be of the same length"
        
        base_info = {
            "num_vars": 4,
            "width": 2,
            "base_size": 0,
            "timestamp": datetime.now(),
        }
        
        msg1 = TrajectoryQueueMessage(
            **base_info,
            trajectory={
                "steps": [
                    {
                        "order": i,
                        "token_type": "ADD",
                        "token_literals": l,
                        "avgQ": q,
                        "reward": r,
                    }
                    for i, l, q, r in zip(range(len(ls[:3])), ls[:3], qs[:3], rs[:3])
                ]
            }
        )
        
        msg2 = TrajectoryQueueMessage(
            **base_info,
            trajectory={
                "steps": [
                    {
                        "order": i,
                        "token_type": "ADD",
                        "token_literals": l,
                        "avgQ": q,
                        "reward": r,
                    }
                    for i, l, q, r in zip(range(len(ls)), ls, qs, rs)
                ]
            }
        )
        expect_size = [2, 3, 5, 6]
        expect_avgQ = [1.0, 2.0, 0.0, 3.0]
        
        try:
            processor.process_trajectory(msg1)
            processor.process_trajectory(msg2)
            
            # TODO: obtain data from warehouse and check if correctly processed
            n = base_info["num_vars"]
            w = base_info["width"]
            nodes = processor.warehouse.download_nodes(n, w)

            assert len(nodes) == 4, "Should have downloaded four nodes"
            
            for i, node in enumerate(nodes):
                finfo = processor.warehouse.get_formula_info(node["formula_id"])
                node["trajectory_id"] = finfo["trajectory_id"]
                node["wl_hash"] = finfo["wl_hash"]
                assert node["size"] == expect_size[i], f"Node {i} size mismatch"
                assert node["avgQ"] == expect_avgQ[i], f"Node {i} avgQ mismatch"
                assert node["num_vars"] == n, f"Node {i} num_vars mismatch"
                assert node["width"] == w, f"Node {i} width mismatch"
                
        finally:
            # Clean up the warehouse by deleting the nodes created during the test
            # This ensures that the warehouse remains clean for subsequent tests.
            for node in nodes:
                processor.warehouse.delete_formula_info(node["formula_id"])
                processor.warehouse.delete_trajectory(node["trajectory_id"])
                processor.warehouse.delete_likely_isomorphic(node["wl_hash"])
                processor.warehouse.delete_evolution_graph_node(node["formula_id"])