import pytest
import httpx
from lsutils import encode_float64_to_base64

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
    
    def test_get_likely_isomorphic(self, client: httpx.Client):
        """Test GET /formula/likely_isomorphic endpoint."""
        response = client.get("/formula/likely_isomorphic", params={"wl_hash": "abcd1234"})
        assert response.status_code == 200
        data = response.json()
        assert "isomorphic_ids" in data
        assert isinstance(data["isomorphic_ids"], list)
        assert "f123" in data["isomorphic_ids"]
        assert "f124" in data["isomorphic_ids"]
    
    def test_add_likely_isomorphic(self, client: httpx.Client):
        """Test POST /formula/likely_isomorphic endpoint."""
        isomorphic_data = {
            "wl_hash": "abcd1234...",
            "formula_id": "f125"
        }
        response = client.post("/formula/likely_isomorphic", json=isomorphic_data)
        assert response.status_code == 201
        data = response.json()
        assert data["message"] == "Likely isomorphic formula added successfully"


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
    
    def test_get_evolution_graph_node(self, client: httpx.Client):
        """Test GET /evolution_graph/node endpoint."""
        response = client.get("/evolution_graph/node", params={"id": "n789"})
        assert response.status_code == 200
        data = response.json()
        assert data["formula_id"] == "f123"
        assert data["avgQ"] == 2.5
        assert data["visited_counter"] == 10
        assert data["inactive"] is False
        assert data["in_degree"] == 2
        assert data["out_degree"] == 3
    
    def test_create_evolution_graph_node(self, client: httpx.Client):
        """Test POST /evolution_graph/node endpoint."""
        node_data = {
            "formula_id": "f456",
            "avgQ": 3.0
        }
        response = client.post("/evolution_graph/node", json=node_data)
        assert response.status_code == 201
        data = response.json()
        assert "node_id" in data
        assert len(data["node_id"]) > 0
    
    def test_update_evolution_graph_node(self, client: httpx.Client):
        """Test PUT /evolution_graph/node endpoint."""
        update_data = {
            "node_id": "n789",
            "inc_visited_counter": 5,
            "inactive": True
        }
        response = client.put("/evolution_graph/node", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Node updated successfully"
    
    def test_delete_evolution_graph_node(self, client: httpx.Client):
        """Test DELETE /evolution_graph/node endpoint."""
        response = client.delete("/evolution_graph/node", params={"node_id": "n789"})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Node deleted successfully"


class TestEvolutionGraphEdge:
    """Test evolution graph edge endpoints."""
    
    def test_get_evolution_graph_edge(self, client: httpx.Client):
        """Test GET /evolution_graph/edge endpoint."""
        response = client.get("/evolution_graph/edge", params={"edge_id": "e456"})
        assert response.status_code == 200
        data = response.json()
        assert data["base_formula_id"] == "f123"
        assert data["new_formula_id"] == "f124"
    
    def test_create_evolution_graph_edge(self, client: httpx.Client):
        """Test POST /evolution_graph/edge endpoint."""
        edge_data = {
            "base_formula_id": "f123",
            "new_formula_id": "f456"
        }
        response = client.post("/evolution_graph/edge", json=edge_data)
        assert response.status_code == 201
        data = response.json()
        assert "edge_id" in data
        assert len(data["edge_id"]) > 0
    
    def test_delete_evolution_graph_edge(self, client: httpx.Client):
        """Test DELETE /evolution_graph/edge endpoint."""
        response = client.delete("/evolution_graph/edge", params={"edge_id": "e456"})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Edge deleted successfully"


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
        response = client.get("/formula/definition", params={"id": "f123"})
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "f123"
        assert "definition" in data
        assert isinstance(data["definition"], list)
        assert len(data["definition"]) == 2
        assert data["definition"][0] == ["x1", "x2", "x3"]
        assert data["definition"][1] == ["x4", "x5"]
    
    def test_get_evolution_subgraph(self, client: httpx.Client):
        """Test GET /evolution_graph/subgraph endpoint."""
        response = client.get("/evolution_graph/subgraph", params={"num_vars": 3, "width": 2})
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)
    
    def test_add_formula(self, client: httpx.Client):
        """Test POST /formula/add endpoint."""
        formula_data = {
            "base_formula_id": "f122",
            "trajectory_id": "t456",
            "avgQ": 2.8,
            "wl_hash": "xyz789...",
            "num_vars": 4,
            "width": 3,
            "size": 7
        }
        response = client.post("/formula/add", json=formula_data)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert len(data["id"]) > 0
    
    def test_add_subgraph(self, client: httpx.Client):
        """Test POST /evolution_graph/subgraph endpoint."""
        subgraph_data = {
            "nodes": [{"formula_id": "f123", "avgQ": 2.5}],
            "edges": [{"base_formula_id": "f123", "new_formula_id": "f124"}]
        }
        response = client.post("/evolution_graph/subgraph", json=subgraph_data)
        assert response.status_code == 201
        data = response.json()
        assert data["message"] == "Subgraph added successfully"
    
    def test_contract_edge(self, client: httpx.Client):
        """Test POST /evolution_graph/contract_edge endpoint."""
        contract_data = {
            "edge_id": "e456"
        }
        response = client.post("/evolution_graph/contract_edge", json=contract_data)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Edge contracted successfully"


if __name__ == "__main__":
    pytest.main(["-v", __file__])