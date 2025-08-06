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

class TestFormulaInfo:
    """Test formula info endpoints."""
    def test_crud_formula_info(self, client: httpx.Client):
        """Test /formula/info endpoints."""
        # Test post
        formula_data = {
            "base_formula_id": "f010",
            "trajectory_id": "t786",
            "avgQ": 3.7,
            "wl_hash": "xyz1234...",
            "num_vars": 5,
            "width": 3,
            "size": 10,
            "node_id": "n321",
        }
        response = client.post("/formula/info", json=formula_data)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert len(data["id"]) > 0
        formula_id = data["id"]
        
        # Test update
        update_data = {
            "id": formula_id,
            "avgQ": 3.0,
            "size": 6
        }
        response = client.put("/formula/info", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Formula updated successfully"
        
        # Test get
        response = client.get("/formula/info", params={"id": formula_id})
        assert response.status_code == 200
        data = response.json()
        assert data["base_formula_id"] == "f010"
        assert data["trajectory_id"] == "t786"
        assert data["avgQ"] == 3.0
        assert data["wl_hash"] == "xyz1234..."
        assert data["num_vars"] == 5
        assert data["width"] == 3
        assert data["size"] == 6
        assert data["node_id"] == "n321"
        
        # Test delete
        response = client.delete("/formula/info", params={"id": formula_id})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Formula deleted successfully"

    def test_code_422(self, client: httpx.Client):
        """Test 422 error for invalid formula info."""
        # Test post with invalid data
        invalid_data = {
            "base_formula_id": "f010",
            "trajectory_id": "t786",
            "avgQ": "not_a_float",  # Invalid type
            "wl_hash": "xyz1234...",
            "num_vars": 5,
            "width": 3,
            "size": 10,
            "node_id": "n321",
        }
        response = client.post("/formula/info", json=invalid_data)
        assert response.status_code == 422
        data = response.json()
        assert data["detail"][0]["msg"] == "Input should be a valid number, unable to parse string as a number"
        
        # Test update with invalid data
        invalid_update_data = {
            "id": "invalid_id",
            "avgQ": "not_a_float"  # Invalid type
        }
        response = client.put("/formula/info", json=invalid_update_data)
        assert response.status_code == 422
        data = response.json()
        assert data["detail"][0]["msg"] == "Input should be a valid number, unable to parse string as a number"


class TestLikelyIsomorphic:
    """Test likely isomorphic endpoints."""
    
    def test_crd_likely_isomorphic(self, client: httpx.Client):
        """Test /formula/likely_isomorphic endpoints."""
        # Test post
        isomorphic_data = {
            "wl_hash": "abcd1234...",
            "formula_id": "f123"
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
        """Test /trajectory endpoints."""
        # Test post
        trajectory_data = {
            "base_formula_id": "f123",
            "steps": [
                {
                    "token_type": 0,
                    "token_literals": 5,
                    "reward": 7/3
                }
            ]
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert len(data["id"]) > 0
        trajectory_id = data["id"]
        
        # Test update
        update_data = {
            "id": trajectory_id,
            "steps": [
                {
                    "order": 0,
                    "token_type": 1,
                    "token_literals": 3,
                    "reward": 9/7
                }
            ]
        }
        response = client.put("/trajectory", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Trajectory updated successfully"
        
        # Test get
        response = client.get("/trajectory", params={"id": trajectory_id})
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == trajectory_id
        assert len(data["steps"]) == 1
        step = data["steps"][0]
        assert step["token_type"] == 1
        assert step["token_literals"] == 3
        assert step["reward"] == 9/7

        # Test delete
        response = client.delete("/trajectory", params={"id": trajectory_id})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Trajectory deleted successfully"

    def test_code_422(self, client: httpx.Client):
        """Test 422 error for invalid trajectory."""
        # Test post with invalid data
        invalid_data = {
            "base_formula_id": "f123",
            "steps": [
                {
                    "token_type": "not_an_int",  # Invalid type
                    "token_literals": 5,
                    "reward": 7/3
                }
            ]
        }
        response = client.post("/trajectory", json=invalid_data)
        assert response.status_code == 422
        data = response.json()
        assert data["detail"][0]["msg"] == "Input should be a valid integer, unable to parse string as an integer"

    def test_code_404(self, client: httpx.Client):
        """Test 404 error for non-existent trajectory."""
        response = client.get("/trajectory", params={"id": "non_existent_id"})
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Trajectory not found"

class TestEvolutionGraphNode:
    """Test evolution graph node endpoints."""
    
    def test_crud_evolution_graph_node(self, client: httpx.Client):
        """Test /evolution_graph/node endpoints."""
        # Test post
        node_data = {
            "formula_id": "f123",
            "avgQ": 2.5,
            "num_vars": 3,
            "width": 4,
            "size": 5
        }
        response = client.post("/evolution_graph/node", json=node_data)
        assert response.status_code == 201
        data = response.json()
        assert "formula_id" in data
        assert node_data["formula_id"] == data["formula_id"]
        assert len(data["formula_id"]) > 0
        formula_id = data["formula_id"]
        
        # Test update
        update_data = {
            "formula_id": "f123",
            "inc_visited_counter": 10,
        }
        response = client.put("/evolution_graph/node", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Node updated successfully"
        
        # Test get
        response = client.get("/evolution_graph/node", params={"id": formula_id})
        assert response.status_code == 200
        data = response.json()
        assert data["formula_id"] == "f123"
        assert data["avgQ"] == 2.5
        assert data["num_vars"] == 3
        assert data["width"] == 4
        assert data["size"] == 5
        assert data["visited_counter"] == 10  # Updated value
        
        # Test delete
        response = client.delete("/evolution_graph/node", params={"formula_id": formula_id})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Node deleted successfully"
    
    def test_code_422(self, client: httpx.Client):
        """Test 422 error for invalid evolution graph node."""
        # Test post with invalid data
        invalid_data = {
            "formula_id": "f123",
            "avgQ": "not_a_float",  # Invalid type
            "num_vars": 3,
            "width": 4,
            "size": 5
        }
        response = client.post("/evolution_graph/node", json=invalid_data)
        assert response.status_code == 422
        data = response.json()
        assert data["detail"][0]["msg"] == "Input should be a valid number, unable to parse string as a number"
        
        # Test update with invalid data
        invalid_update_data = {
            "formula_id": "f123",
            "inc_visited_counter": "not_an_int"  # Invalid type
        }
        response = client.put("/evolution_graph/node", json=invalid_update_data)
        assert response.status_code == 422
        data = response.json()
        assert data["detail"][0]["msg"] == "Input should be a valid integer, unable to parse string as an integer"
    
    def test_code_404(self, client: httpx.Client):
        """Test 404 error for non-existent evolution graph node."""
        response = client.get("/evolution_graph/node", params={"id": "non_existent_id"})
        print(response.text)
        assert response.status_code == 404
        data = response.json()
        assert data["detail"] == "Node not found"


# class TestEvolutionGraphEdge:
#     """Test evolution graph edge endpoints."""
    
#     def test_get_evolution_graph_edge(self, client: httpx.Client):
#         """Test GET /evolution_graph/edge endpoint."""
#         response = client.get("/evolution_graph/edge", params={"edge_id": "e456"})
#         assert response.status_code == 200
#         data = response.json()
#         assert data["base_formula_id"] == "f123"
#         assert data["new_formula_id"] == "f124"
    
#     def test_create_evolution_graph_edge(self, client: httpx.Client):
#         """Test POST /evolution_graph/edge endpoint."""
#         edge_data = {
#             "base_formula_id": "f123",
#             "new_formula_id": "f456"
#         }
#         response = client.post("/evolution_graph/edge", json=edge_data)
#         assert response.status_code == 201
#         data = response.json()
#         assert "edge_id" in data
#         assert len(data["edge_id"]) > 0
    
#     def test_delete_evolution_graph_edge(self, client: httpx.Client):
#         """Test DELETE /evolution_graph/edge endpoint."""
#         response = client.delete("/evolution_graph/edge", params={"edge_id": "e456"})
#         assert response.status_code == 200
#         data = response.json()
#         assert data["message"] == "Edge deleted successfully"


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
        """Test GET /formula/definition endpoint."""
        formula_ids = []
        trajectory_ids = []

        try:
            # Insert base formula (f0)
            f0_data = {
                # "base_formula_id": "base_formula",
                # "trajectory_id": None,
                "avgQ": 1.0, "wl_hash": "hash0", "num_vars": 2, "width": 2, "size": 2, "node_id": "n0"
            }
            response = client.post("/formula/info", json=f0_data)
            assert response.status_code == 201
            f0_id = response.json()["id"]
            formula_ids.append(f0_id)

            # Insert first trajectory (t1) from f0
            t1_data = {
                "base_formula_id": f0_id,
                "steps": [
                    {"token_type": 0, "token_literals": 1, "reward": 0.5},
                    {"token_type": 0, "token_literals": 3, "reward": 0.5}
                ]
            }
            response = client.post("/trajectory", json=t1_data)
            assert response.status_code == 201
            t1_id = response.json()["id"]
            trajectory_ids.append(t1_id)

            # Insert first derived formula (f1)
            f1_data = {
                "base_formula_id": f0_id, "trajectory_id": t1_id,
                "avgQ": 1.5, "wl_hash": "hash1", "num_vars": 2, "width": 2, "size": 3, "node_id": "n1"
            }
            response = client.post("/formula/info", json=f1_data)
            assert response.status_code == 201
            f1_id = response.json()["id"]
            formula_ids.append(f1_id)

            # Insert second trajectory (t2) from f1
            t2_data = {
                "base_formula_id": f1_id,
                "steps": [
                    {"token_type": 0, "token_literals": 2, "reward": 0.6}, {"token_type": 1, "token_literals": 1, "reward": 0.6}
                ]
            }
            response = client.post("/trajectory", json=t2_data)
            assert response.status_code == 201
            t2_id = response.json()["id"]
            trajectory_ids.append(t2_id)

            # Insert second derived formula (f2)
            f2_data = {
                "base_formula_id": f1_id, "trajectory_id": t2_id,
                "avgQ": 2.1, "wl_hash": "hash2", "num_vars": 2, "width": 2, "size": 4, "node_id": "n2"
            }
            response = client.post("/formula/info", json=f2_data)
            assert response.status_code == 201
            f2_id = response.json()["id"]
            formula_ids.append(f2_id)

            # Get the full definition of the last formula (f2)
            response = client.get("/formula/definition", params={"id": f2_id})
            assert response.status_code == 200
            definition = response.json()

            # Verify the definition
            assert definition["id"] == f2_id
            assert set(definition["definition"]) == {2, 3} 

        finally:
            # Clean up all created test data
            for fid in formula_ids:
                client.delete("/formula/info", params={"id": fid})
            for tid in trajectory_ids:
                client.delete("/trajectory", params={"id": tid})
    
    def test_create_new_path(self, client: httpx.Client):
        """Test POST /path endpoint."""
        path_data = {
            "path": ["f123", "f456", "f789"]
        }
        
        for fid in path_data["path"]:
            # Create a formula node for each formula ID
            response = client.post("/evolution_graph/node", json={
                "formula_id": fid,
                "avgQ": 1.0,
                "num_vars": 2,
                "width": 2,
                "size": 2
            })
            response.raise_for_status()
        
        try:
            # Create a new path with the created formula nodes
            response = client.post("/evolution_graph/path", json=path_data)
            assert response.status_code == 201
            data = response.json()
            assert data["message"] == "Path created successfully"
        finally:
            # Clean up created formula nodes
            for fid in path_data["path"]:
                response = client.delete("/evolution_graph/node", params={"formula_id": fid})
                response.raise_for_status()
    
    def test_download_nodes(self, client: httpx.Client):
        """Test GET /evolution_graph/download_nodes endpoint."""
        # Create some nodes for testing
        nodes_data = [
            {"formula_id": "f123", "avgQ": 2.5, "num_vars": 3, "width": 4, "size": 5},
            {"formula_id": "f456", "avgQ": 3.0, "num_vars": 3, "width": 4, "size": 6}
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
            assert "formula_id" in first_node
            assert first_node["formula_id"] == "f123"
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
                response = client.delete("/evolution_graph/node", params={"formula_id": node["formula_id"]})
                response.raise_for_status()
    
    def test_download_hypernodes(self, client: httpx.Client):
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
    
        for path in [path1, path2]:
            for fid, avgQ in path:
                # Create a formula node for each formula ID
                response = client.post("/evolution_graph/node", json={
                    "formula_id": fid,
                    "avgQ": avgQ,
                    "num_vars": 2,
                    "width": 2,
                    "size": 2
                })
                response.raise_for_status()
            # Create a new path with the created formula nodes
            response = client.post("/evolution_graph/path", json={
                "path": [fid for fid, _ in path]
            })
            response.raise_for_status()
    
        # try:
        #     # Download hypernodes
        #     response = client.get("/evolution_graph/download_hypernodes", params={
        #         "num_vars": 2, 
        #         "width": 2, 
        #         "size_constraint": 2
        #     })
        #     assert response.status_code == 200
        #     data = response.json()
        #     assert isinstance(data["hypernodes"], list)
        #     assert len(data["hypernodes"]) == 1
        #     assert data["hypernodes"][0] == (1.0, ["f2", "f3", "f4"])
        # finally:
        #     # Clean up created nodes and paths
        #     # for fid, _ in nodes:
        #     #     response = client.delete("/evolution_graph/node", params={"formula_id": fid})
        #     #     response.raise_for_status()
        #     pass # # Uncomment when the endpoint is implemented
    
    # def test_get_evolution_subgraph(self, client: httpx.Client):
    #     """Test GET /evolution_graph/subgraph endpoint."""
    #     response = client.get("/evolution_graph/subgraph", params={"num_vars": 3, "width": 2})
    #     assert response.status_code == 200
    #     data = response.json()
    #     assert "nodes" in data
    #     assert "edges" in data
    #     assert isinstance(data["nodes"], list)
    #     assert isinstance(data["edges"], list)
    
    # def test_add_formula(self, client: httpx.Client):
    #     """Test POST /formula/add endpoint."""
    #     formula_data = {
    #         "base_formula_id": "f122",
    #         "trajectory_id": "t456",
    #         "avgQ": 2.8,
    #         "wl_hash": "xyz789...",
    #         "num_vars": 4,
    #         "width": 3,
    #         "size": 7
    #     }
    #     response = client.post("/formula/add", json=formula_data)
    #     assert response.status_code == 201
    #     data = response.json()
    #     assert "id" in data
    #     assert len(data["id"]) > 0
    
    # def test_add_subgraph(self, client: httpx.Client):
    #     """Test POST /evolution_graph/subgraph endpoint."""
    #     subgraph_data = {
    #         "nodes": [{"formula_id": "f123", "avgQ": 2.5}],
    #         "edges": [{"base_formula_id": "f123", "new_formula_id": "f124"}]
    #     }
    #     response = client.post("/evolution_graph/subgraph", json=subgraph_data)
    #     assert response.status_code == 201
    #     data = response.json()
    #     assert data["message"] == "Subgraph added successfully"
    
    # def test_contract_edge(self, client: httpx.Client):
    #     """Test POST /evolution_graph/contract_edge endpoint."""
    #     contract_data = {
    #         "edge_id": "e456"
    #     }
    #     response = client.post("/evolution_graph/contract_edge", json=contract_data)
    #     assert response.status_code == 200
    #     data = response.json()
    #     assert data["message"] == "Edge contracted successfully"


if __name__ == "__main__":
    pytest.main(["-v", __file__])