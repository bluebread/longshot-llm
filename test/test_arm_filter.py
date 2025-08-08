import pathlib
import pytest
import httpx
import functools
from longshot.agent import WarehouseAgent

host = "localhost"
port = 8050
armfilter_url = f"http://{host}:{port}"

@pytest.fixture(scope="module")
def client():
    path_data = {
        "avgQ": [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 1.0],
        "vc": [10, 20, 30, 40, 50, 60, 70],
        "definition": [
            [
                {
                    "token_type": 0,
                    "token_literals": i,
                    "reward": 1.0,
                } for i in range(4)
            ],
        ] * 7
    }
    fs = []
    ts = []
    
    with WarehouseAgent(host="localhost", port=8000) as warehouse:
        iterator = zip(
            path_data["avgQ"], 
            path_data["vc"],
            path_data["definition"],
        )
        
        for avgQ, vc, fdef in iterator:
            # Create a formula node for each formula ID
            traj_id = warehouse.post_trajectory(
                steps=fdef
            )
            fid = warehouse.post_formula_info(
                trajectory_id=traj_id,
                avgQ=avgQ,
                num_vars=4,
                width=2,
                size=vc,
                wl_hash=f"formula_{traj_id}",
                node_id="",
            )
            warehouse.post_evolution_graph_node(
                formula_id=fid,
                avgQ=avgQ,
                num_vars=4,
                width=2,
                size=vc,
            )
            warehouse.put_evolution_graph_node(
                formula_id=fid,
                inc_visited_counter=vc
            )
            
            fs.append(fid)
            ts.append(traj_id)

        warehouse.post_evolution_graph_path(
            path=fs,
        )
            
    with httpx.Client(base_url=armfilter_url) as client:
        yield client
        
    with WarehouseAgent(host="localhost", port=8000) as warehouse:
        # Clean up the test data
        for fid in fs:
            warehouse.delete_formula_info(fid)
            warehouse.delete_evolution_graph_node(fid)
        for tid in ts:
            warehouse.delete_trajectory(tid)

class TestArmFilterAPI:
    """Test suite for the Arm Filter API endpoints."""
    def test_topk_arms_success(self, client: httpx.Client):
        """
        Test the /topk_arms endpoint with required parameters.
        """
        params = {
            "num_vars": 4,
            "width": 2,
            "k": 2,
            "size_constraint": 100
        }
        response = client.get("/topk_arms", params=params)
        assert response.status_code == 200
        data = response.json()
        assert "top_k_arms" in data
        assert len(data["top_k_arms"]) == 2
        first_arm = data["top_k_arms"][0]
        assert "formula_id" in first_arm
        assert "definition" in first_arm
        
        
    def test_topk_arms_missing_parameter(self, client: httpx.Client):
        """
        Test that a 422 error is returned if a required parameter is missing.
        """
        params = {
            "num_vars": 4,
            "k": 2
        }
        response = client.get("/topk_arms", params=params)
        assert response.status_code == 422


    def test_topk_arms_invalid_parameter_type(self, client: httpx.Client):
        """
        Test that a 422 error is returned for invalid parameter types.
        """
        params = {
            "num_vars": "ten",  # Invalid type
            "width": 5,
            "k": 2
        }
        response = client.get("/topk_arms", params=params)
        assert response.status_code == 422

if __name__ == "__main__":
    pytest.main([pathlib.Path(__file__).name])