import pathlib
import pytest
from longshot.agent import WarehouseAgent, ArmRanker

@pytest.fixture(scope="module")
def ranker():
    """Fixture that sets up test data and returns an ArmRanker instance."""
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
    
    # Set up test data
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
    
    # Create ArmRanker with WarehouseAgent
    with WarehouseAgent(host="localhost", port=8000) as sync_warehouse:
        ranker = ArmRanker(sync_warehouse)
        yield ranker
        
    # Clean up the test data
    with WarehouseAgent(host="localhost", port=8000) as warehouse:
        for fid in fs:
            warehouse.delete_formula_info(fid)
            warehouse.delete_evolution_graph_node(fid)
        for tid in ts:
            warehouse.delete_trajectory(tid)

class TestArmRanker:
    """Test suite for the ArmRanker class."""
    
    def test_topk_arms_success(self, ranker: ArmRanker):
        """
        Test the topk_arms method with valid parameters.
        """
        result = ranker.topk_arms(
            num_vars=4,
            width=2,
            k=2,
            size_constraint=100
        )
        
        assert isinstance(result, list)
        assert len(result) == 2
        
        first_arm = result[0]
        assert "formula_id" in first_arm
        assert "definition" in first_arm
        assert isinstance(first_arm["definition"], list)
        
    def test_topk_arms_no_size_constraint(self, ranker: ArmRanker):
        """
        Test the topk_arms method without size constraint.
        """
        result = ranker.topk_arms(
            num_vars=4,
            width=2,
            k=3
        )
        
        assert isinstance(result, list)
        assert len(result) == 3
        
        for arm in result:
            assert "formula_id" in arm
            assert "definition" in arm
    
    def test_score_method(self, ranker: ArmRanker):
        """
        Test the score method with sample arm data.
        """
        from longshot.agent.ranker import Arm
        
        arm = Arm(
            avgQ=2.5,
            visited_counter=10,
            in_degree=2,
            out_degree=3
        )
        
        score = ranker.score(arm, num_vars=4, total_visited=100)
        assert isinstance(score, float)
        assert score > 0
        
    def test_score_standalone_function(self):
        """
        Test the standalone score_arm function.
        """
        from longshot.agent.ranker import Arm, score_arm
        
        arm = Arm(
            avgQ=2.5,
            visited_counter=10,
            in_degree=2,
            out_degree=3
        )
        
        score = score_arm(arm, num_vars=4, total_visited=100)
        assert isinstance(score, float)
        assert score > 0
        
        # Test with custom config
        custom_config = {"eps": 0.2, "wq": 2.0, "wvc": 1.0}
        score_custom = score_arm(arm, num_vars=4, total_visited=100, config=custom_config)
        assert isinstance(score_custom, float)
        assert score_custom != score  # Should be different with different config

if __name__ == "__main__":
    pytest.main([pathlib.Path(__file__).name])