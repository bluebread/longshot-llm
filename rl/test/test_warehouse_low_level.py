import pytest
import httpx

host = "localhost"
port = 8000
warehouse_url = f"http://{host}:{port}"

@pytest.fixture
def client():
    with httpx.Client(base_url=warehouse_url) as client:
        yield client


class TestFormulaInfo:
    """Test formula info endpoints."""
    
    def test_get_formula_info(self, client):
        """Test GET /formula/info endpoint."""
        response = client.get("/formula/info", params={"id": "f123"})
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "f123"
        assert data["base_formula_id"] == "f122"
        assert data["trajectory_id"] == "t456"
        assert data["avgQ"] == 2.5
        assert data["wl_hash"] == "abcd1234..."
        assert data["num_vars"] == 3
        assert data["width"] == 2
        assert data["size"] == 5
        assert data["timestamp"] == "2025-07-21T12:00:00Z"
        assert data["node_id"] == "n789"
        assert data["full_trajectory_id"] == "t999"
    
    def test_create_formula_info(self, client):
        """Test POST /formula/info endpoint."""
        formula_data = {
            "base_formula_id": "f122",
            "trajectory_id": "t456",
            "avgQ": 2.5,
            "wl-hash": "abcd1234...",
            "num_vars": 3,
            "width": 2,
            "size": 5,
            "timestamp": "2025-07-21T12:00:00Z",
            "node_id": "n789",
            "full_trajectory_id": "t999"
        }
        response = client.post("/formula/info", json=formula_data)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert len(data["id"]) > 0
    
    def test_update_formula_info(self, client):
        """Test PUT /formula/info endpoint."""
        update_data = {
            "id": "f123",
            "avgQ": 3.0,
            "size": 6
        }
        response = client.put("/formula/info", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Formula updated successfully"
    
    def test_delete_formula_info(self, client):
        """Test DELETE /formula/info endpoint."""
        response = client.delete("/formula/info", params={"id": "f123"})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Formula deleted successfully"


class TestLikelyIsomorphic:
    """Test likely isomorphic endpoints."""
    
    def test_get_likely_isomorphic(self, client):
        """Test GET /formula/likely_isomorphic endpoint."""
        response = client.get("/formula/likely_isomorphic", params={"wl-hash": "abcd1234..."})
        assert response.status_code == 200
        data = response.json()
        assert "isomorphic_ids" in data
        assert isinstance(data["isomorphic_ids"], list)
        assert "f123" in data["isomorphic_ids"]
        assert "f124" in data["isomorphic_ids"]
    
    def test_add_likely_isomorphic(self, client):
        """Test POST /formula/likely_isomorphic endpoint."""
        isomorphic_data = {
            "wl-hash": "abcd1234...",
            "formula_id": "f125"
        }
        response = client.post("/formula/likely_isomorphic", json=isomorphic_data)
        assert response.status_code == 201
        data = response.json()
        assert data["message"] == "Likely isomorphic formula added successfully"


class TestTrajectory:
    """Test trajectory endpoints."""
    
    def test_get_trajectory(self, client):
        """Test GET /trajectory endpoint."""
        response = client.get("/trajectory", params={"id": "t456"})
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "t456"
        assert "steps" in data
        assert isinstance(data["steps"], list)
        assert len(data["steps"]) == 1
        step = data["steps"][0]
        assert step["order"] == 0
        assert step["token_type"] == 0
        assert step["token_literals"] == 5
        assert step["reward"] == 0.1
    
    def test_create_trajectory(self, client):
        """Test POST /trajectory endpoint."""
        trajectory_data = {
            "steps": [
                {
                    "order": 0,
                    "token_type": 1,
                    "token_literals": 3,
                    "reward": 0.2
                },
                {
                    "order": 1,
                    "token_type": 0,
                    "token_literals": 7,
                    "reward": 0.15
                }
            ]
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert len(data["id"]) > 0
    
    def test_update_trajectory(self, client):
        """Test PUT /trajectory endpoint."""
        update_data = {
            "id": "t456",
            "steps": [
                {
                    "order": 0,
                    "token_type": 1,
                    "token_literals": 4,
                    "reward": 0.3
                }
            ]
        }
        response = client.put("/trajectory", json=update_data)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Trajectory updated successfully"
    
    def test_delete_trajectory(self, client):
        """Test DELETE /trajectory endpoint."""
        response = client.delete("/trajectory", params={"id": "t456"})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Trajectory deleted successfully"


class TestEvolutionGraphNode:
    """Test evolution graph node endpoints."""
    
    def test_get_evolution_graph_node(self, client):
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
    
    def test_create_evolution_graph_node(self, client):
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
    
    def test_update_evolution_graph_node(self, client):
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
    
    def test_delete_evolution_graph_node(self, client):
        """Test DELETE /evolution_graph/node endpoint."""
        response = client.delete("/evolution_graph/node", params={"node_id": "n789"})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Node deleted successfully"


class TestEvolutionGraphEdge:
    """Test evolution graph edge endpoints."""
    
    def test_get_evolution_graph_edge(self, client):
        """Test GET /evolution_graph/edge endpoint."""
        response = client.get("/evolution_graph/edge", params={"edge_id": "e456"})
        assert response.status_code == 200
        data = response.json()
        assert data["base_formula_id"] == "f123"
        assert data["new_formula_id"] == "f124"
    
    def test_create_evolution_graph_edge(self, client):
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
    
    def test_delete_evolution_graph_edge(self, client):
        """Test DELETE /evolution_graph/edge endpoint."""
        response = client.delete("/evolution_graph/edge", params={"edge_id": "e456"})
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Edge deleted successfully"


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self, client):
        """Test GET /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        

if __name__ == "__main__":
    pytest.main(["-v", __file__])