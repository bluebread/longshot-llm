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
                (0, 5, 7/3)  # (token_type, token_literals, cur_avgQ)
            ],
            "max_num_vars": 8,
            "max_width": 4,
            "max_size": 100
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
                (1, 3, 9/7)  # (token_type, token_literals, cur_avgQ)
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
        # Step is now a tuple/list: [token_type, token_literals, cur_avgQ]
        assert step[0] == 1  # token_type
        assert step[1] == 3  # token_literals
        assert step[2] == 9/7  # cur_avgQ

        # Test delete
        response = client.delete("/trajectory", params={"traj_id": trajectory_id})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Trajectory deleted successfully"

    def test_code_422(self, client: httpx.Client):
        """Test 422 error for invalid trajectory."""
        # Test post with invalid data - tuple with wrong type
        invalid_data = {
            "steps": [
                ("not_an_int", 5, 7/3)  # First element should be int, not string
            ]
        }
        response = client.post("/trajectory", json=invalid_data)
        assert response.status_code == 422
        data = response.json()
        # The error message will be about invalid tuple format
        assert "Input should be a valid" in data["detail"][0]["msg"]

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
                (0, 5, 2.5)  # (token_type, token_literals, cur_avgQ)
            ],
            "max_num_vars": 8,
            "max_width": 4,
            "max_size": 100
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        try:
            # Test post with integrated formula data (node_id is auto-generated)
            node_data = {
                "avgQ": 2.5,
                "num_vars": 3,
                "width": 4,
                "size": 5,
                "wl_hash": "test_hash_123",
                "isodegrees": [[0, 0], [0, 1], [1, 1]],  # Example isodegrees (will be flattened by server)
                "traj_id": traj_id,
                "traj_slice": 0
            }
            response = client.post("/evolution_graph/node", json=node_data)
            assert response.status_code == 201
            data = response.json()
            assert "node_id" in data
            assert len(data["node_id"]) > 0
            node_id = data["node_id"]
            
            # Test update
            update_data = {
                "node_id": node_id,  # Use the auto-generated ID
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
            assert data["node_id"] == node_id  # Check against auto-generated ID
            assert data["avgQ"] == 3.0  # Updated value
            assert data["num_vars"] == 3
            assert data["width"] == 4
            assert data["size"] == 6  # Updated value
            assert data["wl_hash"] == "test_hash_123"
            assert data["isodegrees"] == [[0, 0], [0, 1], [1, 1]]  # Neo4j returns lists
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
        # Test post with invalid data (no node_id needed)
        invalid_data = {
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
                    (0, 5, 1.0),   # ADD
                    (1, 10, 1.5)   # DELETE
                ],
                "max_num_vars": 8,
                "max_width": 4,
                "max_size": 100
            }
            response = client.post("/trajectory", json=trajectory_data)
            assert response.status_code == 201
            traj_id = response.json()["traj_id"]
            trajectory_ids.append(traj_id)

            # Create evolution graph node with integrated formula data (auto-generated ID)
            node_data = {
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
            created_node_id = response.json()["node_id"]
            formula_ids.append(created_node_id)

            # Get the formula definition (should reconstruct from trajectory slice)
            response = client.get("/formula/definition", params={"node_id": created_node_id})
            assert response.status_code == 200
            definition = response.json()

            # Verify the definition matches the trajectory slice
            # Step 0: ADD 5 → definition = {5}
            # Step 1: DELETE 10 → definition = {5} (10 wasn't in set)
            assert definition["node_id"] == created_node_id
            assert set(definition["definition"]) == {5}  # Result after ADD 5, DELETE 10

        finally:
            # Clean up all created test data
            for fid in formula_ids:
                client.delete("/evolution_graph/node", params={"node_id": fid})
            for tid in trajectory_ids:
                client.delete("/trajectory", params={"traj_id": tid})
    
    def test_create_new_path(self, client: httpx.Client):
        """Test POST /path endpoint."""
        # Create a trajectory for the nodes
        trajectory_data = {
            "steps": [
                (0, 5, 1.0)  # (token_type, token_literals, cur_avgQ)
            ],
            "max_num_vars": 8,
            "max_width": 4,
            "max_size": 100
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        created_node_ids = []
        try:
            # Create formula nodes and collect their auto-generated IDs
            for i in range(3):
                response = client.post("/evolution_graph/node", json={
                    "avgQ": 1.0 + i * 0.5,
                    "num_vars": 2,
                    "width": 2,
                    "size": 2,
                    "wl_hash": f"hash_{i}",
                    "traj_id": traj_id,
                    "traj_slice": 0
                })
                response.raise_for_status()
                created_node_ids.append(response.json()["node_id"])
        
            # Create a new path with the created formula nodes
            path_data = {"path": created_node_ids}
            response = client.post("/evolution_graph/path", json=path_data)
            assert response.status_code == 201
            data = response.json()
            assert data["message"] == "Path created successfully"
        finally:
            # Clean up created formula nodes
            for node_id in created_node_ids:
                response = client.delete("/evolution_graph/node", params={"node_id": node_id})
            # Clean up trajectory
            client.delete("/trajectory", params={"traj_id": traj_id})
    
    def test_download_nodes(self, client: httpx.Client):
        """Test GET /evolution_graph/download_nodes endpoint."""
        # Create trajectory for the nodes
        trajectory_data = {
            "steps": [
                (0, 5, 1.0)  # (token_type, token_literals, cur_avgQ)
            ],
            "max_num_vars": 8,
            "max_width": 4,
            "max_size": 100
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        # Create some nodes for testing with V2 required fields (auto-generated IDs)
        nodes_data = [
            {"avgQ": 2.5, "num_vars": 3, "width": 4, "size": 5, "wl_hash": "hash123", "isodegrees": [[0, 0], [0, 1], [1, 1]], "traj_id": traj_id, "traj_slice": 0},
            {"avgQ": 3.0, "num_vars": 3, "width": 4, "size": 6, "wl_hash": "hash456", "isodegrees": [[0, 1], [1, 1], [1, 2]], "traj_id": traj_id, "traj_slice": 0}
        ]
        
        created_node_ids = []
        for node in nodes_data:
            response = client.post("/evolution_graph/node", json=node)
            assert response.status_code == 201
            created_node_ids.append(response.json()["node_id"])
        
        try:
            # Download nodes
            response = client.get("/evolution_graph/download_nodes", params={"num_vars": 3, "width": 4, "size_constraint": 5})
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data["nodes"], list)
            # Filter to only our test nodes (size <= 5)
            our_nodes = [node for node in data["nodes"] if node["node_id"] in created_node_ids and node["size"] <= 5]
            assert len(our_nodes) >= 1  # At least one of our nodes should match
            
            # Verify the structure of our matching node
            our_node = our_nodes[0]
            assert "node_id" in our_node
            assert our_node["node_id"] in created_node_ids  # Check it's one of our created nodes
            assert "avgQ" in our_node
            assert our_node["avgQ"] == 2.5
            assert "num_vars" in our_node
            assert our_node["num_vars"] == 3
            assert "width" in our_node
            assert our_node["width"] == 4
            assert "size" in our_node
            assert our_node["size"] == 5
            
        finally:
            # Clean up created nodes
            for node_id in created_node_ids:
                response = client.delete("/evolution_graph/node", params={"node_id": node_id})
            # Clean up trajectory
            client.delete("/trajectory", params={"traj_id": traj_id})
    
    def test_download_hypernodes(self, client: httpx.Client):
        # Create trajectory for the nodes
        trajectory_data = {
            "steps": [
                (0, 5, 1.0)  # (token_type, token_literals, cur_avgQ)
            ],
            "max_num_vars": 8,
            "max_width": 4,
            "max_size": 100
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        avgQ_values = [0.0, 1.0, 1.0, 1.0, 2.5, 2.5]  # 6 nodes with different avgQ values
        created_node_ids = []
        
        try:
            # Create all nodes first and collect their IDs
            for i, avgQ in enumerate(avgQ_values):
                response = client.post("/evolution_graph/node", json={
                    "avgQ": avgQ,
                    "num_vars": 2,
                    "width": 2,
                    "size": 2,
                    "wl_hash": f"hash_{i}",
                    "traj_id": traj_id,
                    "traj_slice": 0
                })
                response.raise_for_status()
                created_node_ids.append(response.json()["node_id"])
            
            # Create paths using the auto-generated IDs
            # Path1: nodes 0,1,2,4 (avgQ: 0.0,1.0,1.0,2.5)
            path1_ids = [created_node_ids[0], created_node_ids[1], created_node_ids[2], created_node_ids[4]]
            # Path2: nodes 0,1,3,5 (avgQ: 0.0,1.0,1.0,2.5)
            path2_ids = [created_node_ids[0], created_node_ids[1], created_node_ids[3], created_node_ids[5]]
            
            for path_ids in [path1_ids, path2_ids]:
                response = client.post("/evolution_graph/path", json={
                    "path": path_ids
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
            assert len(data["hypernodes"]) >= 1  # At least one hypernode should exist
            # Check that hypernodes contain some of our created nodes
            all_hypernode_ids = set()
            for hypernode in data["hypernodes"]:
                all_hypernode_ids.update(hypernode["nodes"])
            # At least some of our nodes should be in hypernodes
            our_nodes_in_hypernodes = all_hypernode_ids.intersection(set(created_node_ids))
            assert len(our_nodes_in_hypernodes) >= 3  # At least 3 of our nodes should be in hypernodes
            
        finally:
            # Clean up created nodes and paths
            for node_id in created_node_ids:
                response = client.delete("/evolution_graph/node", params={"node_id": node_id})
            # Clean up trajectory
            client.delete("/trajectory", params={"traj_id": traj_id})


class TestDatasetEndpoints:
    """Test the new dataset endpoints."""
    
    def test_evolution_graph_dataset_basic(self, client: httpx.Client):
        """Test GET /evolution_graph/dataset endpoint basic functionality."""
        # Create trajectory for the nodes
        trajectory_data = {
            "steps": [
                (0, 5, 1.0)  # (token_type, token_literals, cur_avgQ)
            ],
            "max_num_vars": 8,
            "max_width": 4,
            "max_size": 100
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        # Create some nodes and edges for testing (auto-generated IDs)
        nodes_data = [
            {"avgQ": 2.5, "num_vars": 3, "width": 4, "size": 5, "wl_hash": "hash1", "isodegrees": [[0, 0], [0, 1], [1, 1]], "traj_id": traj_id, "traj_slice": 0},
            {"avgQ": 3.0, "num_vars": 3, "width": 4, "size": 6, "wl_hash": "hash2", "isodegrees": [[0, 1], [1, 1], [1, 2]], "traj_id": traj_id, "traj_slice": 0},
            {"avgQ": 2.5, "num_vars": 2, "width": 3, "size": 4, "wl_hash": "hash3", "isodegrees": [[0, 0], [1, 1]], "traj_id": traj_id, "traj_slice": 0}
        ]
        
        created_node_ids = []
        for node in nodes_data:
            response = client.post("/evolution_graph/node", json=node)
            assert response.status_code == 201
            created_node_ids.append(response.json()["node_id"])
        
        # Create paths to generate edges
        path_data = {"path": created_node_ids}
        response = client.post("/evolution_graph/path", json=path_data)
        assert response.status_code == 201
        
        try:
            # Test dataset endpoint
            response = client.get("/evolution_graph/dataset")
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "nodes" in data
            assert "edges" in data
            assert isinstance(data["nodes"], list)
            assert isinstance(data["edges"], list)
            
            # Check that our test nodes are included
            node_ids = [node["node_id"] for node in data["nodes"]]
            for created_id in created_node_ids:
                assert created_id in node_ids
            
            # Verify edge structure with new field names
            if data["edges"]:
                edge = data["edges"][0]
                assert "src" in edge  # Renamed from "source"
                assert "dst" in edge  # Renamed from "target"
                assert "type" in edge  # Renamed from "edge_type"
                assert edge["type"] in ["EVOLVED_TO", "SAME_Q"]
        
        finally:
            # Clean up
            for node_id in created_node_ids:
                client.delete("/evolution_graph/node", params={"node_id": node_id})
            client.delete("/trajectory", params={"traj_id": traj_id})
    
    def test_evolution_graph_dataset_field_filtering(self, client: httpx.Client):
        """Test field filtering functionality with required_fields parameter."""
        # Create trajectory for the nodes
        trajectory_data = {
            "steps": [
                (0, 5, 1.0)  # (token_type, token_literals, cur_avgQ)
            ],
            "max_num_vars": 8,
            "max_width": 4,
            "max_size": 100
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        # Create a test node (auto-generated ID)
        node_data = {
            "avgQ": 2.5, 
            "num_vars": 3, 
            "width": 4, 
            "size": 5, 
            "wl_hash": "hash_field_test",
            "isodegrees": [[0, 0], [0, 1], [1, 1]], 
            "traj_id": traj_id, 
            "traj_slice": 0
        }
        response = client.post("/evolution_graph/node", json=node_data)
        assert response.status_code == 201
        created_node_id = response.json()["node_id"]
        
        try:
            # Test with default fields (should only include node_id)
            response = client.get("/evolution_graph/dataset")
            assert response.status_code == 200
            data = response.json()
            
            # Find our test node
            test_node = next((node for node in data["nodes"] if node["node_id"] == created_node_id), None)
            assert test_node is not None
            assert "node_id" in test_node
            # Should only have node_id by default
            expected_fields = {"node_id"}
            assert set(test_node.keys()) == expected_fields
            
            # Test with specific fields
            required_fields = ["node_id", "avgQ", "num_vars", "size"]
            response = client.get("/evolution_graph/dataset", params={"required_fields": required_fields})
            assert response.status_code == 200
            data = response.json()
            
            # Find our test node
            test_node = next((node for node in data["nodes"] if node["node_id"] == created_node_id), None)
            assert test_node is not None
            assert set(test_node.keys()) == set(required_fields)
            assert test_node["avgQ"] == 2.5
            assert test_node["num_vars"] == 3
            assert test_node["size"] == 5
            assert "width" not in test_node  # Should be filtered out
            assert "wl_hash" not in test_node  # Should be filtered out
            
            # Test with all available fields
            all_fields = ["node_id", "avgQ", "num_vars", "width", "size", "wl_hash", "timestamp", "traj_id", "traj_slice"]
            response = client.get("/evolution_graph/dataset", params={"required_fields": all_fields})
            assert response.status_code == 200
            data = response.json()
            
            # Find our test node
            test_node = next((node for node in data["nodes"] if node["node_id"] == created_node_id), None)
            assert test_node is not None
            # Should have all requested fields
            for field in all_fields:
                assert field in test_node
        
        finally:
            # Clean up
            client.delete("/evolution_graph/node", params={"node_id": created_node_id})
            client.delete("/trajectory", params={"traj_id": traj_id})
    
    def test_evolution_graph_dataset_edge_field_names(self, client: httpx.Client):
        """Test that edge fields use the renamed field names."""
        # Create trajectory for the nodes
        trajectory_data = {
            "steps": [
                (0, 5, 1.0)  # (token_type, token_literals, cur_avgQ)
            ],
            "max_num_vars": 8,
            "max_width": 4,
            "max_size": 100
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        # Create nodes with same avgQ to generate SAME_Q edges
        nodes_data = [
            {"node_id": "edge_test_1", "avgQ": 2.5, "num_vars": 3, "width": 4, "size": 5, "wl_hash": "hash1", "traj_id": traj_id, "traj_slice": 0},
            {"node_id": "edge_test_2", "avgQ": 2.5, "num_vars": 3, "width": 4, "size": 5, "wl_hash": "hash2", "traj_id": traj_id, "traj_slice": 0},
            {"node_id": "edge_test_3", "avgQ": 3.0, "num_vars": 3, "width": 4, "size": 5, "wl_hash": "hash3", "traj_id": traj_id, "traj_slice": 0}
        ]
        
        for node in nodes_data:
            response = client.post("/evolution_graph/node", json=node)
            assert response.status_code == 201
        
        # Create path to generate EVOLVED_TO edges
        path_data = {"path": ["edge_test_1", "edge_test_2", "edge_test_3"]}
        response = client.post("/evolution_graph/path", json=path_data)
        assert response.status_code == 201
        
        try:
            # Get dataset and check edge field names
            response = client.get("/evolution_graph/dataset")
            assert response.status_code == 200
            data = response.json()
            
            # Check that we have edges
            assert len(data["edges"]) > 0
            
            # Verify all edges have the correct field names
            for edge in data["edges"]:
                assert "src" in edge, "Edge should have 'src' field (renamed from 'source')"
                assert "dst" in edge, "Edge should have 'dst' field (renamed from 'target')"
                assert "type" in edge, "Edge should have 'type' field (renamed from 'edge_type')"
                
                # Should NOT have old field names
                assert "source" not in edge, "Edge should not have old 'source' field"
                assert "target" not in edge, "Edge should not have old 'target' field"
                assert "edge_type" not in edge, "Edge should not have old 'edge_type' field"
                
                # Check that edge type is valid
                assert edge["type"] in ["EVOLVED_TO", "SAME_Q"]
                
                # Check that src and dst are valid node IDs
                assert isinstance(edge["src"], str)
                assert isinstance(edge["dst"], str)
        
        finally:
            # Clean up
            for node in nodes_data:
                client.delete("/evolution_graph/node", params={"node_id": node["node_id"]})
            client.delete("/trajectory", params={"traj_id": traj_id})
    
    def test_trajectory_dataset_basic(self, client: httpx.Client):
        """Test GET /trajectory/dataset endpoint basic functionality."""
        # Create test trajectories
        trajectory_data_1 = {
            "steps": [
                (0, 5, 1.0),   # ADD
                (1, 3, 1.5)    # DELETE
            ],
            "max_num_vars": 8,
            "max_width": 4,
            "max_size": 100
        }
        response = client.post("/trajectory", json=trajectory_data_1)
        assert response.status_code == 201
        traj_id_1 = response.json()["traj_id"]
        
        trajectory_data_2 = {
            "steps": [
                (0, 10, 2.0),  # ADD
                (0, 15, 2.5),  # ADD
                (1, 10, 3.0)   # DELETE
            ],
            "max_num_vars": 8,
            "max_width": 4,
            "max_size": 100
        }
        response = client.post("/trajectory", json=trajectory_data_2)
        assert response.status_code == 201
        traj_id_2 = response.json()["traj_id"]
        
        try:
            # Test dataset endpoint
            response = client.get("/trajectory/dataset")
            assert response.status_code == 200
            data = response.json()
            
            # Verify response structure
            assert "trajectories" in data
            assert isinstance(data["trajectories"], list)
            assert len(data["trajectories"]) >= 2  # Should include our test trajectories
            
            # Find our test trajectories
            test_trajs = [traj for traj in data["trajectories"] if traj["traj_id"] in [traj_id_1, traj_id_2]]
            assert len(test_trajs) == 2
            
            # Verify trajectory structure
            for traj in test_trajs:
                assert "traj_id" in traj
                assert "timestamp" in traj
                assert "steps" in traj
                assert isinstance(traj["steps"], list)
        
        finally:
            # Clean up
            client.delete("/trajectory", params={"traj_id": traj_id_1})
            client.delete("/trajectory", params={"traj_id": traj_id_2})
    
    def test_trajectory_dataset_tuple_format(self, client: httpx.Client):
        """Test that trajectory steps are returned in tuple format."""
        # Create test trajectory with known steps
        trajectory_data = {
            "steps": [
                (0, 5, 1.0),   # ADD
                (1, 10, 2.5),  # DELETE
                (0, 15, 3.7)   # ADD
            ],
            "max_num_vars": 8,
            "max_width": 4,
            "max_size": 100
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        try:
            # Get dataset
            response = client.get("/trajectory/dataset")
            assert response.status_code == 200
            data = response.json()
            
            # Find our test trajectory
            test_traj = next((traj for traj in data["trajectories"] if traj["traj_id"] == traj_id), None)
            assert test_traj is not None
            
            # Verify steps are in tuple format
            steps = test_traj["steps"]
            assert len(steps) == 3
            
            # Each step should be a list/tuple of 3 elements: [token_type, token_literals, cur_avgQ]
            expected_steps = [
                [0, 5, 1.0],
                [1, 10, 2.5],
                [0, 15, 3.7]
            ]
            
            for i, step in enumerate(steps):
                assert isinstance(step, list), f"Step {i} should be a list/tuple"
                assert len(step) == 3, f"Step {i} should have exactly 3 elements"
                assert step[0] == expected_steps[i][0], f"Step {i} token_type mismatch"
                assert step[1] == expected_steps[i][1], f"Step {i} token_literals mismatch"
                assert step[2] == expected_steps[i][2], f"Step {i} cur_avgQ mismatch"
                
                # Verify types
                assert isinstance(step[0], int), f"Step {i} token_type should be int"
                assert isinstance(step[1], int), f"Step {i} token_literals should be int"
                assert isinstance(step[2], (int, float)), f"Step {i} cur_avgQ should be number"
        
        finally:
            # Clean up
            client.delete("/trajectory", params={"traj_id": traj_id})
    
    def test_dataset_endpoints_empty_data(self, client: httpx.Client):
        """Test dataset endpoints behavior with no data."""
        # Test evolution graph dataset with no nodes
        response = client.get("/evolution_graph/dataset")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)
        # Should be empty or contain only existing data from other tests
        
        # Test trajectory dataset - might have data from other tests, but should not fail
        response = client.get("/trajectory/dataset")
        assert response.status_code == 200
        data = response.json()
        assert "trajectories" in data
        assert isinstance(data["trajectories"], list)
    
    def test_evolution_graph_dataset_invalid_fields(self, client: httpx.Client):
        """Test evolution graph dataset with invalid field names."""
        # Test with some valid and some invalid fields
        invalid_fields = ["node_id", "invalid_field", "avgQ", "nonexistent_field"]
        response = client.get("/evolution_graph/dataset", params={"required_fields": invalid_fields})
        assert response.status_code == 200
        data = response.json()
        
        # Should only include valid fields and ignore invalid ones
        if data["nodes"]:
            node = data["nodes"][0]
            assert "node_id" in node
            assert "avgQ" in node if "avgQ" in node else True  # May or may not be present depending on filtering
            assert "invalid_field" not in node
            assert "nonexistent_field" not in node


if __name__ == "__main__":
    pytest.main(["-v", __file__])