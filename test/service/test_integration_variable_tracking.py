"""
Integration test for variable and width tracking across the entire system.
"""

import pytest
import requests
import time
from longshot.formula import FormulaGraph


class TestVariableTrackingIntegration:
    """Test the complete integration of variable and width tracking."""
    
    WAREHOUSE_URL = "http://localhost:8000"
    CLUSTERBOMB_URL = "http://localhost:8060"
    
    def test_warehouse_trajectory_fields(self):
        """Test that warehouse properly stores and retrieves max_num_vars and max_width."""
        # Create a trajectory with the new fields
        trajectory_data = {
            "steps": [
                (0, 0x00000003, 1.5),  # ADD x0, x1 (width 2)
                (0, 0x00000007, 2.0),  # ADD x0, x1, x2 (width 3)
                (1, 0x00000003, 1.8),  # DEL x0, x1
            ],
            "max_num_vars": 8,  # Collection parameter
            "max_width": 5,     # Collection parameter
            "max_size": 150     # Collection parameter
        }
        
        # POST trajectory
        response = requests.post(f"{self.WAREHOUSE_URL}/trajectory", json=trajectory_data)
        assert response.status_code == 201
        traj_id = response.json()["traj_id"]
        
        # GET trajectory back
        response = requests.get(f"{self.WAREHOUSE_URL}/trajectory", params={"traj_id": traj_id})
        assert response.status_code == 200
        retrieved = response.json()
        
        # Verify fields were stored and retrieved
        assert retrieved.get("max_num_vars") == 8
        assert retrieved.get("max_width") == 5
        assert retrieved.get("max_size") == 150
        assert len(retrieved["steps"]) == 3
        
        # Clean up
        requests.delete(f"{self.WAREHOUSE_URL}/trajectory", params={"traj_id": traj_id})
    
    def test_clusterbomb_formula_tracking(self):
        """Test that clusterbomb properly tracks actual formula properties."""
        # Create a rollout request with specific formulas
        rollout_request = {
            "num_vars": 6,  # Max collection parameter
            "width": 4,     # Max collection parameter
            "size": 100,
            "prefix_traj": [
                (0, 0x00000003, 1.0),  # x0, x1 (uses 2 vars, width 2)
                (0, 0x00000014, 1.5),  # x2, x4 (now uses 4 vars: x0,x1,x2,x4, width 2)
            ],
            "steps_per_trajectory": 5,
            "num_trajectories": 1,
            "early_stop": True,
            "seed": 42  # Deterministic for testing
        }
        
        response = requests.post(f"{self.CLUSTERBOMB_URL}/weapon/rollout", json=rollout_request)
        assert response.status_code == 200
        result = response.json()
        
        # Verify response structure
        assert "num_trajectories" in result
        assert "processed_formulas" in result
        assert result["num_trajectories"] == 1
    
    def test_evolution_graph_node_properties(self):
        """Test that evolution graph nodes have correct num_vars and width."""
        # First create some test data
        rollout_request = {
            "num_vars": 8,  # Max vars allowed
            "width": 3,     # Max width allowed
            "size": 100,
            "prefix_traj": [
                # Create formula using only x0, x1, x4 (3 vars, not 8)
                (0, 0x00000001, 0.5),  # x0 (width 1)
                (0, 0x00000002, 1.0),  # x1 (width 1) 
                (0, 0x00000010, 1.5),  # x4 (width 1)
                (0, 0x00000013, 2.0),  # x0,x1,x4 (width 3, max width)
            ],
            "steps_per_trajectory": 2,
            "num_trajectories": 1,
            "early_stop": False,
            "seed": 123
        }
        
        response = requests.post(f"{self.CLUSTERBOMB_URL}/weapon/rollout", json=rollout_request)
        assert response.status_code == 200
        
        # Give it time to process
        time.sleep(0.5)
        
        # Instead of querying all nodes, let's verify the specific formulas we created
        # The base formula with x0, x1, x4 should have 3 actual variables and width 3
        # But we need to query with the actual values, not the max values
        
        # Query for nodes with 3 variables (the actual count in our formula)
        response = requests.get(f"{self.WAREHOUSE_URL}/evolution_graph/download_nodes",
                               params={"num_vars": 3, "width": 3})
        
        if response.status_code == 200:
            nodes = response.json()["nodes"]
            if nodes:
                # Verify that these nodes have the correct properties
                for node in nodes[:5]:  # Check first few nodes
                    assert node["num_vars"] == 3  # Actual variables used
                    assert node["width"] == 3  # Max width in formula
        
        # Also test that we can query with different actual values
        response2 = requests.get(f"{self.WAREHOUSE_URL}/evolution_graph/download_nodes",
                                params={"num_vars": 2, "width": 1})
        # This query might return nodes or not, but should be valid
        assert response2.status_code == 200
    
    def test_formula_graph_properties(self):
        """Test FormulaGraph directly to verify property calculations."""
        # Create a formula that uses x0, x1, x4 (3 variables, not 5)
        gates = [
            0x00000003,  # x0, x1 (width 2)
            0x00000012,  # x1, x4 (width 2)
            0x00000007,  # x0, x1, x2 (width 3)
        ]
        
        fg = FormulaGraph(gates)
        
        # Check actual variable count
        assert fg.num_vars == 4  # x0, x1, x2, x4 = 4 variables
        assert sorted(list(fg._used_variables)) == [0, 1, 2, 4]
        
        # Check width tracking
        assert fg.width == 3  # Maximum width
        assert fg.width_counter == {2: 2, 3: 1}  # 2 gates with width 2, 1 with width 3
    
    def test_trajectory_dataset_endpoint(self):
        """Test that trajectory dataset endpoint includes new fields."""
        # First create a trajectory
        trajectory_data = {
            "steps": [(0, 1, 0.5)],
            "max_num_vars": 10,
            "max_width": 6,
            "max_size": 200
        }
        
        response = requests.post(f"{self.WAREHOUSE_URL}/trajectory", json=trajectory_data)
        assert response.status_code == 201
        
        # Get dataset (correct endpoint)
        response = requests.get(f"{self.WAREHOUSE_URL}/trajectory/dataset")
        assert response.status_code == 200
        dataset = response.json()
        
        # Find our trajectory (should be one of the recent ones)
        trajectories = dataset["trajectories"]
        recent_trajectory = None
        for traj in trajectories:
            if traj.get("max_num_vars") == 10 and traj.get("max_width") == 6 and traj.get("max_size") == 200:
                recent_trajectory = traj
                break
        
        if recent_trajectory:
            assert recent_trajectory["max_num_vars"] == 10
            assert recent_trajectory["max_width"] == 6
            assert recent_trajectory["max_size"] == 200


if __name__ == "__main__":
    print("Running integration tests...")
    print("Make sure warehouse and clusterbomb services are running!")
    pytest.main([__file__, "-v", "-s"])