import pytest
import httpx
from datetime import datetime, timedelta

host = "localhost"
port = 8000
warehouse_url = f"http://{host}:{port}"

@pytest.fixture
def client():
    with httpx.Client(base_url=warehouse_url) as client:
        yield client

    
class TestTrajectory:
    """Test trajectory endpoints."""
    
    def test_crud_trajectory(self, client: httpx.Client):
        """Test /trajectory endpoints with V2 schema."""
        # Test post
        trajectory_data = {
            "steps": [
                (0, 5, 7/3)  # (token_type, token_literals, cur_avgQ)
            ],
            "num_vars": 8,
            "width": 4
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


class TestHealthCheck:
    """Test health check endpoint."""
    
    def test_health_check(self, client: httpx.Client):
        """Test GET /health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestDatasetEndpoints:
    """Test the trajectory dataset endpoints."""
    
    def test_trajectory_dataset_basic(self, client: httpx.Client):
        """Test GET /trajectory/dataset endpoint basic functionality."""
        # Create test trajectories
        trajectory_data_1 = {
            "steps": [
                (0, 5, 1.0),   # ADD
                (1, 3, 1.5)    # DELETE
            ],
            "num_vars": 8,
            "width": 4
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
            "num_vars": 8,
            "width": 4
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
            "num_vars": 8,
            "width": 4
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
    
    def test_trajectory_dataset_filtering(self, client: httpx.Client):
        """Test GET /trajectory/dataset with filtering parameters."""
        # Create trajectories with different parameters
        traj_ids = []
        
        # Trajectory 1: num_vars=3, width=2
        trajectory_data_1 = {
            "steps": [(0, 5, 1.0)],
            "num_vars": 3,
            "width": 2
        }
        response = client.post("/trajectory", json=trajectory_data_1)
        assert response.status_code == 201
        traj_ids.append(response.json()["traj_id"])
        
        # Trajectory 2: num_vars=3, width=4
        trajectory_data_2 = {
            "steps": [(0, 10, 2.0)],
            "num_vars": 3,
            "width": 4
        }
        response = client.post("/trajectory", json=trajectory_data_2)
        assert response.status_code == 201
        traj_ids.append(response.json()["traj_id"])
        
        # Trajectory 3: num_vars=5, width=2
        trajectory_data_3 = {
            "steps": [(0, 15, 3.0)],
            "num_vars": 5,
            "width": 2
        }
        response = client.post("/trajectory", json=trajectory_data_3)
        assert response.status_code == 201
        traj_ids.append(response.json()["traj_id"])
        
        try:
            # Test filtering by num_vars
            response = client.get("/trajectory/dataset", params={"num_vars": 3})
            assert response.status_code == 200
            data = response.json()
            filtered_trajs = [traj for traj in data["trajectories"] if traj["traj_id"] in traj_ids]
            assert len(filtered_trajs) == 2  # Should only get trajectories 1 and 2
            assert all(traj["num_vars"] == 3 for traj in filtered_trajs)
            
            # Test filtering by width
            response = client.get("/trajectory/dataset", params={"width": 2})
            assert response.status_code == 200
            data = response.json()
            filtered_trajs = [traj for traj in data["trajectories"] if traj["traj_id"] in traj_ids]
            assert len(filtered_trajs) == 2  # Should only get trajectories 1 and 3
            assert all(traj["width"] == 2 for traj in filtered_trajs)
            
            # Test filtering by both num_vars and width
            response = client.get("/trajectory/dataset", params={"num_vars": 3, "width": 2})
            assert response.status_code == 200
            data = response.json()
            filtered_trajs = [traj for traj in data["trajectories"] if traj["traj_id"] in traj_ids]
            assert len(filtered_trajs) == 1  # Should only get trajectory 1
            assert filtered_trajs[0]["num_vars"] == 3
            assert filtered_trajs[0]["width"] == 2
            
        finally:
            # Clean up
            for traj_id in traj_ids:
                client.delete("/trajectory", params={"traj_id": traj_id})
    
    def test_trajectory_dataset_timestamp_filtering(self, client: httpx.Client):
        """Test GET /trajectory/dataset with timestamp filtering parameters."""
        # Create a trajectory to test with
        trajectory_data = {
            "steps": [(0, 5, 1.0)],
            "num_vars": 3,
            "width": 2
        }
        response = client.post("/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        try:
            # Get the created trajectory to know its timestamp
            response = client.get("/trajectory", params={"traj_id": traj_id})
            assert response.status_code == 200
            traj_timestamp_str = response.json()["timestamp"]
            traj_timestamp = datetime.fromisoformat(traj_timestamp_str.replace('Z', '+00:00'))
            
            # Test filtering with 'since' parameter (should include the trajectory)
            since_time = traj_timestamp - timedelta(minutes=1)
            response = client.get("/trajectory/dataset", params={"since": since_time.isoformat()})
            assert response.status_code == 200
            data = response.json()
            filtered_trajs = [traj for traj in data["trajectories"] if traj["traj_id"] == traj_id]
            assert len(filtered_trajs) == 1  # Should include our trajectory
            
            # Test filtering with 'since' parameter (should exclude the trajectory)
            since_time = traj_timestamp + timedelta(minutes=1)
            response = client.get("/trajectory/dataset", params={"since": since_time.isoformat()})
            assert response.status_code == 200
            data = response.json()
            filtered_trajs = [traj for traj in data["trajectories"] if traj["traj_id"] == traj_id]
            assert len(filtered_trajs) == 0  # Should exclude our trajectory
            
            # Test filtering with 'until' parameter (should include the trajectory)
            until_time = traj_timestamp + timedelta(minutes=1)
            response = client.get("/trajectory/dataset", params={"until": until_time.isoformat()})
            assert response.status_code == 200
            data = response.json()
            filtered_trajs = [traj for traj in data["trajectories"] if traj["traj_id"] == traj_id]
            assert len(filtered_trajs) == 1  # Should include our trajectory
            
            # Test filtering with 'until' parameter (should exclude the trajectory)
            until_time = traj_timestamp - timedelta(minutes=1)
            response = client.get("/trajectory/dataset", params={"until": until_time.isoformat()})
            assert response.status_code == 200
            data = response.json()
            filtered_trajs = [traj for traj in data["trajectories"] if traj["traj_id"] == traj_id]
            assert len(filtered_trajs) == 0  # Should exclude our trajectory
            
            # Test filtering with both 'since' and 'until' (should include the trajectory)
            since_time = traj_timestamp - timedelta(minutes=1)
            until_time = traj_timestamp + timedelta(minutes=1)
            response = client.get("/trajectory/dataset", params={
                "since": since_time.isoformat(),
                "until": until_time.isoformat()
            })
            assert response.status_code == 200
            data = response.json()
            filtered_trajs = [traj for traj in data["trajectories"] if traj["traj_id"] == traj_id]
            assert len(filtered_trajs) == 1  # Should include our trajectory
            
            # Test with 'since' > 'until' (should return 400 error)
            since_time = traj_timestamp + timedelta(minutes=1)
            until_time = traj_timestamp - timedelta(minutes=1)
            response = client.get("/trajectory/dataset", params={
                "since": since_time.isoformat(),
                "until": until_time.isoformat()
            })
            assert response.status_code == 400
            error_detail = response.json()["detail"]
            assert "Invalid date range" in error_detail
            assert "cannot be after" in error_detail
            
        finally:
            # Clean up
            client.delete("/trajectory", params={"traj_id": traj_id})
    
    def test_dataset_endpoints_empty_data(self, client: httpx.Client):
        """Test dataset endpoints behavior with no data."""
        # Test trajectory dataset - might have data from other tests, but should not fail
        response = client.get("/trajectory/dataset")
        assert response.status_code == 200
        data = response.json()
        assert "trajectories" in data
        assert isinstance(data["trajectories"], list)


if __name__ == "__main__":
    pytest.main(["-v", __file__])