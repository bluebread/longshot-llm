import pytest
import httpx

host = "localhost"
port = 8000
warehouse_url = f"http://{host}:{port}"

@pytest.fixture
def client():
    with httpx.Client(base_url=warehouse_url) as client:
        yield client

class TestHighLevelAPI:
    """Test high-level API endpoints."""
    
    def test_get_formula_definition(self, client):
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
    
    def test_get_evolution_subgraph(self, client):
        """Test GET /evolution_graph/subgraph endpoint."""
        response = client.get("/evolution_graph/subgraph", params={"num_vars": 3, "width": 2})
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert isinstance(data["nodes"], list)
        assert isinstance(data["edges"], list)
    
    def test_add_formula(self, client):
        """Test POST /formula/add endpoint."""
        formula_data = {
            "base_formula_id": "f122",
            "trajectory_id": "t456",
            "avgQ": 2.8,
            "wl-hash": "xyz789...",
            "num_vars": 4,
            "width": 3,
            "size": 7
        }
        response = client.post("/formula/add", json=formula_data)
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert len(data["id"]) > 0
    
    def test_add_subgraph(self, client):
        """Test POST /evolution_graph/subgraph endpoint."""
        subgraph_data = {
            "nodes": [{"formula_id": "f123", "avgQ": 2.5}],
            "edges": [{"base_formula_id": "f123", "new_formula_id": "f124"}]
        }
        response = client.post("/evolution_graph/subgraph", json=subgraph_data)
        assert response.status_code == 201
        data = response.json()
        assert data["message"] == "Subgraph added successfully"
    
    def test_contract_edge(self, client):
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