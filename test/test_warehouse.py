import pytest
import httpx
from longshot.utils import encode_float64_to_base64

host = "localhost"
port = 8000
warehouse_url = f"http://{host}:{port}"

@pytest.fixture
def client():
    with httpx.Client(base_url=warehouse_url) as client:
        yield client

# Formula info endpoints removed in V2 - formulas are now integrated into Evolution Graph nodes

class TestLikelyIsomorphic:
    """Test likely isomorphic endpoints."""
    
    def test_crd_likely_isomorphic(self, client: httpx.Client):
        """Test /formula/likely_isomorphic endpoints."""
        # Test post
        isomorphic_data = {
            "wl_hash": "abcd1234...",
            "node_id": "f123"
        }
        response = client.post("/formula/likely_isomorphic", json=isomorphic_data)
        assert response.status_code == 201
        data = response.json()
        assert data["message"] == "Likely isomorphic formula added successfully"
        
        # Test get
        response = client.get("/formula/likely_isomorphic", params={"wl_hash": "abcd1234..."})
        assert response.status_code == 200
        data = response.json()
        assert "isomorphic_ids" in data
        assert isinstance(data["isomorphic_ids"], list)
        assert "f123" in data["isomorphic_ids"]
        
        # Test delete
        response = client.delete("/formula/likely_isomorphic", params={"wl_hash": "abcd1234..."})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Likely isomorphic formula deleted successfully"
    
    
class TestTrajectory:
    """Test trajectory endpoints."""
    
    def test_crud_trajectory(self, client: httpx.Client):
        """Test /trajectory endpoints with V2 schema."""
        # Test post
        trajectory_data = {
            "steps": [
                {
                    "token_type": 0,
                    "token_literals": 5,
                    "cur_avgQ": 7/3
                }
            ]
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        data = response.json()
        assert "traj_id" in data
        assert len(data["traj_id"]) > 0
        trajectory_id = data["traj_id"]
        
        # Test update
        update_data = {
            "traj_id": trajectory_id,
            "steps": [
                {
                    "order": 0,
                    "token_type": 1,
                    "token_literals": 3,
                    "cur_avgQ": 9/7
                }
            ]
        }
        response = client.put("/trajectory", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Trajectory updated successfully"
        
        # Test get
        response = client.get("/trajectory", params={"traj_id": trajectory_id})
        assert response.status_code == 200
        data = response.json()
        assert data["traj_id"] == trajectory_id
        assert len(data["steps"]) == 1
        step = data["steps"][0]
        assert step["token_type"] == 1
        assert step["token_literals"] == 3
        assert step["cur_avgQ"] == 9/7

        # Test delete
        response = client.delete("/trajectory", params={"traj_id": trajectory_id})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Trajectory deleted successfully"

    def test_code_422(self, client: httpx.Client):
        """Test 422 error for invalid trajectory."""
        # Test post with invalid data
        invalid_data = {
            "steps": [
                {
                    "token_type": "not_an_int",  # Invalid type
                    "token_literals": 5,
                    "cur_avgQ": 7/3
                }
            ]
        }
        response = client.post("/trajectory", json=invalid_data)
        assert response.status_code == 422
        data = response.json()
        assert data["detail"][0]["msg"] == "Input should be a valid integer, unable to parse string as an integer"

    def test_code_404(self, client: httpx.Client):
        """Test 404 error for non-existent trajectory."""
        response = client.get("/trajectory", params={"traj_id": "non_existent_id"})
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Trajectory not found"


class TestEvolutionGraphNode:
    """Test evolution graph node endpoints."""
    
    def test_crud_evolution_graph_node(self, client: httpx.Client):
        """Test /evolution_graph/node endpoints with V2 integrated schema."""
        # Clean up any existing test node
        try:
            client.delete("/evolution_graph/node", params={"node_id": "f123"})
        except:
            pass  # Ignore if it doesn't exist
            
        # First create a trajectory for the node
        trajectory_data = {
            "steps": [
                {"token_type": 0, "token_literals": 5, "cur_avgQ": 2.5}
            ]
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        try:
            # Test post with integrated formula data
            node_data = {
                "node_id": "f123",
                "avgQ": 2.5,
                "num_vars": 3,
                "width": 4,
                "size": 5,
                "wl_hash": "test_hash_123",
                "traj_id": traj_id,
                "traj_slice": 0
            }
            response = client.post("/evolution_graph/node", json=node_data)
            assert response.status_code == 201
            data = response.json()
            assert "node_id" in data
            assert node_data["node_id"] == data["node_id"]
            assert len(data["node_id"]) > 0
            node_id = data["node_id"]
            
            # Test update
            update_data = {
                "node_id": "f123",
                "avgQ": 3.0,
                "size": 6
            }
            response = client.put("/evolution_graph/node", json=update_data)
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Node updated successfully"
            
            # Test get
            response = client.get("/evolution_graph/node", params={"node_id": node_id})
            assert response.status_code == 200
            data = response.json()
            assert data["node_id"] == "f123"
            assert data["avgQ"] == 3.0  # Updated value
            assert data["num_vars"] == 3
            assert data["width"] == 4
            assert data["size"] == 6  # Updated value
            assert data["wl_hash"] == "test_hash_123"
            assert data["traj_id"] == traj_id
            assert data["traj_slice"] == 0
            
            # Test delete
            response = client.delete("/evolution_graph/node", params={"node_id": node_id})
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Node deleted successfully"
            
        finally:
            # Clean up trajectory
            client.delete("/trajectory", params={"traj_id": traj_id})
    
    def test_code_422(self, client: httpx.Client):
        """Test 422 error for invalid evolution graph node."""
        # Test post with invalid data
        invalid_data = {
            "node_id": "f123",
            "avgQ": "not_a_float",  # Invalid type
            "num_vars": 3,
            "width": 4,
            "size": 5,
            "wl_hash": "test_hash",
            "traj_id": "test_traj",
            "traj_slice": 0
        }
        response = client.post("/evolution_graph/node", json=invalid_data)
        assert response.status_code == 422
        data = response.json()
        assert data["detail"][0]["msg"] == "Input should be a valid number, unable to parse string as a number"
        
        # Test update with invalid data
        invalid_update_data = {
            "node_id": "f123",
            "avgQ": "not_a_float"  # Invalid type
        }
        response = client.put("/evolution_graph/node", json=invalid_update_data)
        assert response.status_code == 422
        data = response.json()
        assert data["detail"][0]["msg"] == "Input should be a valid number, unable to parse string as a number"
    
    def test_code_404(self, client: httpx.Client):
        """Test 404 error for non-existent evolution graph node."""
        response = client.get("/evolution_graph/node", params={"node_id": "non_existent_id"})
        print(response.text)
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Node not found"

class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self, client: httpx.Client):
        """Test GET /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        

class TestHighLevelAPI:
    """Test high-level API endpoints."""
    
    def test_get_formula_definition(self, client: httpx.Client):
        """Test GET /formula/definition endpoint with V2 schema."""
        formula_ids = []
        trajectory_ids = []

        try:
            # Create trajectories with V2 schema (cur_avgQ instead of reward)
            trajectory_data = {
                "steps": [
                    {"token_type": 0, "token_literals": 5, "cur_avgQ": 1.0},
                    {"token_type": 1, "token_literals": 10, "cur_avgQ": 1.5}
                ]
            }
            response = client.post("/trajectory", json=trajectory_data)
            assert response.status_code == 201
            traj_id = response.json()["traj_id"]
            trajectory_ids.append(traj_id)

            # Create evolution graph node with integrated formula data
            node_data = {
                "node_id": "test_formula_def",
                "avgQ": 1.5,
                "num_vars": 2,
                "width": 2,
                "size": 2,
                "wl_hash": "test_hash_def",
                "traj_id": traj_id,
                "traj_slice": 1  # Use slice 1 (second step)
            }
            response = client.post("/evolution_graph/node", json=node_data)
            assert response.status_code == 201
            formula_ids.append(node_data["node_id"])

            # Get the formula definition (should reconstruct from trajectory slice)
            response = client.get("/formula/definition", params={"node_id": "test_formula_def"})
            assert response.status_code == 200
            definition = response.json()

            # Verify the definition matches the trajectory slice
            # Step 0: ADD 5 → definition = {5}
            # Step 1: DELETE 10 → definition = {5} (10 wasn't in set)
            assert definition["id"] == "test_formula_def"
            assert set(definition["definition"]) == {5}  # Result after ADD 5, DELETE 10

        finally:
            # Clean up all created test data
            for fid in formula_ids:
                client.delete("/evolution_graph/node", params={"node_id": fid})
            for tid in trajectory_ids:
                client.delete("/trajectory", params={"traj_id": tid})
    
    def test_create_new_path(self, client: httpx.Client):
        """Test POST /path endpoint."""
        path_data = {
            "path": ["f123", "f456", "f789"]
        }
        
        # Create a trajectory for the nodes
        trajectory_data = {
            "steps": [
                {"token_type": 0, "token_literals": 5, "cur_avgQ": 1.0}
            ]
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        try:
            for i, fid in enumerate(path_data["path"]):
                # Create a formula node for each formula ID with V2 required fields
                response = client.post("/evolution_graph/node", json={
                    "node_id": fid,
                    "avgQ": 1.0 + i * 0.5,
                    "num_vars": 2,
                    "width": 2,
                    "size": 2,
                    "wl_hash": f"hash_{fid}",
                    "traj_id": traj_id,
                    "traj_slice": 0
                })
                response.raise_for_status()
        
            # Create a new path with the created formula nodes
            response = client.post("/evolution_graph/path", json=path_data)
            assert response.status_code == 201
            data = response.json()
            assert data["message"] == "Path created successfully"
        finally:
            # Clean up created formula nodes
            for fid in path_data["path"]:
                response = client.delete("/evolution_graph/node", params={"node_id": fid})
            # Clean up trajectory
            client.delete("/trajectory", params={"traj_id": traj_id})
    
    def test_download_nodes(self, client: httpx.Client):
        """Test GET /evolution_graph/download_nodes endpoint."""
        # Create trajectory for the nodes
        trajectory_data = {
            "steps": [
                {"token_type": 0, "token_literals": 5, "cur_avgQ": 1.0}
            ]
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        # Create some nodes for testing with V2 required fields
        nodes_data = [
            {"node_id": "f123", "avgQ": 2.5, "num_vars": 3, "width": 4, "size": 5, "wl_hash": "hash123", "traj_id": traj_id, "traj_slice": 0},
            {"node_id": "f456", "avgQ": 3.0, "num_vars": 3, "width": 4, "size": 6, "wl_hash": "hash456", "traj_id": traj_id, "traj_slice": 0}
        ]
        
        for node in nodes_data:
            response = client.post("/evolution_graph/node", json=node)
            assert response.status_code == 201
        
        try:
            # Download nodes
            response = client.get("/evolution_graph/download_nodes", params={"num_vars": 3, "width": 4, "size_constraint": 5})
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data["nodes"], list)
            assert len(data["nodes"]) == 1
            
            # Verify the structure of the first node
            first_node = data["nodes"][0]
            assert "node_id" in first_node
            assert first_node["node_id"] == "f123"
            assert "avgQ" in first_node
            assert first_node["avgQ"] == 2.5
            assert "num_vars" in first_node
            assert first_node["num_vars"] == 3
            assert "width" in first_node
            assert first_node["width"] == 4
            assert "size" in first_node
            assert first_node["size"] == 5
            
        finally:
            # Clean up created nodes
            for node in nodes_data:
                response = client.delete("/evolution_graph/node", params={"node_id": node["node_id"]})
            # Clean up trajectory
            client.delete("/trajectory", params={"traj_id": traj_id})
    
    def test_download_hypernodes(self, client: httpx.Client):
        # Create trajectory for the nodes
        trajectory_data = {
            "steps": [
                {"token_type": 0, "token_literals": 5, "cur_avgQ": 1.0}
            ]
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        nodes = [
            ("f1", 0.0), # 0
            ("f2", 1.0), # 1
            ("f3", 1.0), # 2
            ("f4", 1.0), # 3
            ("f5", 2.5), # 4
            ("f6", 2.5), # 5
        ]
        path1 = [nodes[0], nodes[1], nodes[2], nodes[4]]
        path2 = [nodes[0], nodes[1], nodes[3], nodes[5]]
    
        try:
            for path in [path1, path2]:
                for fid, avgQ in path:
                    # Create a formula node for each formula ID with V2 required fields
                    response = client.post("/evolution_graph/node", json={
                        "node_id": fid,
                        "avgQ": avgQ,
                        "num_vars": 2,
                        "width": 2,
                        "size": 2,
                        "wl_hash": f"hash_{fid}",
                        "traj_id": traj_id,
                        "traj_slice": 0
                    })
                    response.raise_for_status()
                # Create a new path with the created formula nodes
                response = client.post("/evolution_graph/path", json={
                    "path": [fid for fid, _ in path]
                })
                response.raise_for_status()
    
            # Download hypernodes
            response = client.get("/evolution_graph/download_hypernodes", params={
                "num_vars": 2, 
                "width": 2, 
                "size_constraint": 2
            })
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data["hypernodes"], list)
            assert len(data["hypernodes"]) == 1
            assert set(data["hypernodes"][0]["nodes"]) == set(["f2", "f3", "f4"])
            
        finally:
            # Clean up created nodes and paths
            for fid, _ in nodes:
                response = client.delete("/evolution_graph/node", params={"node_id": fid})
            # Clean up trajectory
            client.delete("/trajectory", params={"traj_id": traj_id})


if __name__ == "__main__":
    pytest.main(["-v", __file__])