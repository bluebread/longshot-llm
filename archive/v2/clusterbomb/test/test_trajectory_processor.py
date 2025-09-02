import pytest
import httpx
import functools
import pprint
from datetime import datetime

from longshot.service import TrajectoryProcessor, WarehouseClient
from longshot.models import TrajectoryQueueMessage
from longshot.service.api_models import TrajectoryProcessingContext
from longshot.formula import FormulaGraph

host = "localhost"
port = 8000
warehouse_url = f"http://{host}:{port}"
hash_iters = 3

@pytest.fixture
def processor():
    with WarehouseClient(host, port) as warehouse:
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
        Saves a formula to the warehouse and returns its ID (V2 compatible).
        """
        response = warehouse.post("/trajectory", json={
            "steps": [
                (0, d, 0.0)  # (token_type, token_literals, cur_avgQ)
                for d in definition
            ],
            "max_num_vars": 8,  # Test default value
            "max_width": 4,     # Test default value
            "max_size": 100     # Test default value
        })
        
        assert response.status_code == 201, f"Failed to save trajectory: {response.text}"

        traj_id = response.json()["traj_id"]
        
        # In V2, create evolution graph node directly (integrated approach)
        response = warehouse.post("/evolution_graph/node", json={
            "node_id": f"test_node_{traj_id[:8]}",
            "avgQ": 0.0,
            "num_vars": len(definition),
            "width": 1,
            "size": len(definition),
            "wl_hash": wl_hash,
            "traj_id": traj_id,
            "traj_slice": len(definition) - 1
        })
        
        assert response.status_code == 201, f"Failed to create node: {response.text}"
        
        return response.json()["node_id"], traj_id

    def del_formula_example(self, warehouse: httpx.Client, fid: str, tid: str):
        """
        Deletes a formula from the warehouse (V2 compatible).
        """
        response = warehouse.delete(f"/evolution_graph/node", params={"node_id": fid})
        assert response.status_code == 200, f"Failed to delete node: {response.text}"
        
        response = warehouse.delete(f"/trajectory", params={"traj_id": tid})
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
        fg1 = FormulaGraph(fdef1)
        fg2 = FormulaGraph(fdef2)
        wl1 = fg1.wl_hash(iterations=hash_iters)
        wl2 = fg2.wl_hash(iterations=hash_iters)
        assert wl1 == wl2, "WL hashes should match for isomorphic graphs"
        
        fid2, tid2 = self.save_formula_example(processor.warehouse._client, fdef2, wl2)
        response = processor.warehouse._client.post("/formula/likely_isomorphic", json={
            "wl_hash": wl2,
            "node_id": fid2,
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
        fg1 = FormulaGraph(fdef1)
        fg2 = FormulaGraph(fdef2)
        wl1 = fg1.wl_hash(iterations=hash_iters)
        wl2 = fg2.wl_hash(iterations=hash_iters)
        assert wl1 != wl2, "WL hashes should not match for non-isomorphic graphs"
        
        fid2, tid2 = self.save_formula_example(processor.warehouse._client, fdef2, wl2)
        response = processor.warehouse._client.post("/formula/likely_isomorphic", json={
            "wl_hash": wl2,
            "node_id": fid2,
        })
        
        assert response.status_code == 201, f"Failed to check isomorphism: {response.text}"
        assert processor.isomorphic_to(fg1) is None
        
        self.del_formula_example(processor.warehouse._client, fid2, tid2)
        response = processor.warehouse._client.delete("/formula/likely_isomorphic", params={"wl_hash": wl2})
        assert response.status_code == 200, f"Failed to delete isomorphic formula: {response.text}"
        
    
    def test_process_trajectory_v2(self, processor: TrajectoryProcessor):
        """Test the new V2 process_trajectory_v2 method."""
        processor.traj_granularity = 2
        processor.traj_num_summits = 2
        
        # Create prefix trajectory (base formula construction)
        prefix_steps = [
            (0, self.encode_literals([0, 2], []), 0.0),  # ADD
            (0, self.encode_literals([1], []), 0.5)       # ADD
        ]
        
        # Create suffix trajectory (new steps)
        suffix_steps = [
            (0, self.encode_literals([2], [1]), 1.0),      # ADD
            (0, self.encode_literals([0, 1], []), 2.0),    # ADD
            (0, self.encode_literals([], [1, 2]), 3.0)     # ADD
        ]
        
        # Create V2 processing context
        context = TrajectoryProcessingContext(
            prefix_traj=prefix_steps,
            suffix_traj=suffix_steps,
            base_formula_hash=None,
            processing_metadata={
                "max_num_vars": 4,
                "max_width": 3,
                "max_size": 100
            }
        )
        
        try:
            # Test V2 trajectory processing
            result = processor.process_trajectory(context)
            
            # Verify result structure
            assert "new_formulas" in result
            assert "evo_path" in result
            assert "base_formula_exists" in result
            assert "processed_formulas" in result
            assert "new_nodes_created" in result
            
            # Verify results are reasonable
            assert isinstance(result["new_formulas"], list)
            assert isinstance(result["evo_path"], list)
            assert isinstance(result["base_formula_exists"], bool)
            assert isinstance(result["processed_formulas"], int)
            assert isinstance(result["new_nodes_created"], list)  # Now a list of node IDs
            
            print(f"✅ V2 trajectory processing completed successfully:")
            print(f"   - Base formula exists: {result['base_formula_exists']}")
            print(f"   - Processed formulas: {result['processed_formulas']}")
            print(f"   - New nodes created: {len(result['new_nodes_created'])} nodes")
            print(f"   - Evolution path length: {len(result['evo_path'])}")
            
        except Exception as e:
            print(f"⚠️  V2 trajectory processing failed: {e}")
            raise e
            
        finally:
            # Clean up any nodes that might have been created
            try:
                num_vars = context.processing_metadata["num_vars"]
                width = context.processing_metadata["width"]
                nodes = processor.warehouse.download_nodes(num_vars, width)
                for node in nodes:
                    try:
                        processor.warehouse.delete_evolution_graph_node(node["node_id"])
                        if "traj_id" in node:
                            processor.warehouse.delete_trajectory(node["traj_id"])
                    except Exception:
                        pass  # Ignore cleanup errors
            except Exception:
                pass  # Ignore if download fails
    
    def test_reconstruct_base_formula(self, processor: TrajectoryProcessor):
        """Test base formula reconstruction from prefix trajectory."""
        
        # Create prefix trajectory that builds a specific formula
        prefix_steps = [
            (0, self.encode_literals([0, 1], []), 1.0),  # ADD
            (0, self.encode_literals([2], [0]), 1.5),    # ADD
            (1, self.encode_literals([0, 1], []), 1.2)   # DEL - remove first gate
        ]
        
        try:
            # Test base formula reconstruction
            base_formula = processor.reconstruct_base_formula(prefix_steps)
            
            # Verify the formula graph was created
            assert hasattr(base_formula, 'size')
            
            print(f"✅ Base formula reconstruction completed successfully")
            print(f"   - Formula has gates: {hasattr(base_formula, 'gates')}")
            
        except Exception as e:
            print(f"⚠️  Base formula reconstruction failed: {e}")
            raise e
    
    def test_check_base_formula_exists(self, processor: TrajectoryProcessor):
        """Test checking if base formula already exists in database."""
        
        # Create a simple formula
        formula_def = [self.encode_literals([0, 1], []), self.encode_literals([2], [])]
        formula_graph = FormulaGraph(formula_def)
        
        try:
            # Test checking if formula exists (should be False initially)
            exists, formula_id = processor.check_base_formula_exists(formula_graph)
            
            # Initially should not exist
            assert isinstance(exists, bool)
            assert formula_id is None or isinstance(formula_id, str)
            
            print(f"✅ Base formula existence check completed")
            print(f"   - Formula exists: {exists}")
            print(f"   - Formula ID: {formula_id}")
            
        except Exception as e:
            print(f"⚠️  Base formula existence check failed: {e}")
            raise e