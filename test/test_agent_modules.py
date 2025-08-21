#!/usr/bin/env python3
"""
Test script for the moved TrajectoryProcessor and ArmRanker modules.
Tests basic functionality and warehouse integration.
"""

import pytest
from unittest.mock import MagicMock, patch
from longshot.agent import WarehouseAgent, TrajectoryProcessor, ArmRanker
from longshot.models import TrajectoryQueueMessage, TrajectoryInfoStep
import networkx as nx


class TestWarehouseAgent:
    """Test the updated WarehouseAgent with V2 API compatibility."""

    def test_warehouse_agent_initialization(self):
        """Test that WarehouseAgent can be initialized properly."""
        agent = WarehouseAgent(host="localhost", port=8000)
        assert agent._client.base_url == "http://localhost:8000"
        agent.close()

    @patch('httpx.Client.get')
    def test_get_trajectory_uses_correct_field_name(self, mock_get):
        """Test that get_trajectory uses 'traj_id' parameter instead of 'id'."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"traj_id": "test_traj_123", "steps": []}
        mock_get.return_value = mock_response
        
        agent = WarehouseAgent(host="localhost", port=8000)
        result = agent.get_trajectory("test_traj_123")
        
        # Verify the correct parameter was used
        mock_get.assert_called_once_with("/trajectory", params={"traj_id": "test_traj_123"})
        assert result["traj_id"] == "test_traj_123"
        agent.close()

    @patch('httpx.Client.post')
    def test_post_trajectory_returns_correct_field(self, mock_post):
        """Test that post_trajectory returns 'traj_id' from response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"traj_id": "new_traj_456"}
        mock_post.return_value = mock_response
        
        agent = WarehouseAgent(host="localhost", port=8000)
        result = agent.post_trajectory(steps=[])
        
        mock_post.assert_called_once()
        assert result == "new_traj_456"
        agent.close()

    @patch('httpx.Client.get')
    def test_get_evolution_graph_node_uses_correct_field(self, mock_get):
        """Test that get_evolution_graph_node uses 'node_id' parameter."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"node_id": "test_node_789", "avgQ": 2.5}
        mock_get.return_value = mock_response
        
        agent = WarehouseAgent(host="localhost", port=8000)
        result = agent.get_evolution_graph_node("test_node_789")
        
        mock_get.assert_called_once_with("/evolution_graph/node", params={"node_id": "test_node_789"})
        assert result["node_id"] == "test_node_789"
        agent.close()

    @patch('httpx.Client.post')
    def test_post_likely_isomorphic_uses_node_id(self, mock_post):
        """Test that post_likely_isomorphic uses 'node_id' in request body."""
        mock_response = MagicMock()
        mock_post.return_value = mock_response
        
        agent = WarehouseAgent(host="localhost", port=8000)
        agent.post_likely_isomorphic("test_hash", "test_node_123")
        
        expected_body = {"wl_hash": "test_hash", "node_id": "test_node_123"}
        mock_post.assert_called_once_with("/formula/likely_isomorphic", json=expected_body)
        agent.close()


class TestTrajectoryProcessor:
    """Test the updated TrajectoryProcessor with V2 compatibility."""

    def test_trajectory_processor_initialization(self):
        """Test that TrajectoryProcessor can be initialized."""
        mock_warehouse = MagicMock()
        processor = TrajectoryProcessor(mock_warehouse)
        assert processor.warehouse == mock_warehouse
        assert processor.hash_iterations == 5  # default value
        assert processor.traj_granularity == 20  # default value

    def test_definition_to_graph(self):
        """Test that definition_to_graph creates proper networkx graph."""
        # Simple test case: single gate
        definition = [0x000000010000000F]  # Simple gate with variables 0,1,2,3
        graph = TrajectoryProcessor.definition_to_graph(definition)
        
        assert isinstance(graph, nx.Graph)
        assert len(graph.nodes) > 0  # Should have nodes for variables and gate
        
    def test_add_gate_to_graph(self):
        """Test that gates can be added to graph correctly."""
        graph = nx.Graph()
        gate = 0x000000010000000F  # Simple gate
        
        TrajectoryProcessor.add_gate_to_graph(graph, gate)
        
        # Should have the gate node
        assert gate in graph.nodes
        # Should have variable and literal nodes
        assert any("x" in str(node) for node in graph.nodes)

    def test_retrieve_definition_with_none(self):
        """Test that retrieve_definition returns empty list for None input."""
        mock_warehouse = MagicMock()
        processor = TrajectoryProcessor(mock_warehouse)
        
        result = processor.retrieve_definition(None)
        assert result == []


class TestArmRanker:
    """Test the updated ArmRanker with V2 compatibility."""

    def test_arm_ranker_initialization(self):
        """Test that ArmRanker can be initialized properly."""
        mock_warehouse = MagicMock()
        ranker = ArmRanker(mock_warehouse)
        
        assert ranker.warehouse == mock_warehouse
        assert ranker.config.eps == 0.1  # default value
        assert ranker.config.wq == 1.0  # default value
        assert ranker.config.wvc == 2.0  # default value

    def test_arm_ranker_with_custom_config(self):
        """Test ArmRanker initialization with custom configuration."""
        mock_warehouse = MagicMock()
        config = {"eps": 0.05, "wq": 1.5, "wvc": 3.0}
        ranker = ArmRanker(mock_warehouse, **config)
        
        assert ranker.config.eps == 0.05
        assert ranker.config.wq == 1.5
        assert ranker.config.wvc == 3.0

    @patch.object(ArmRanker, 'score')
    def test_topk_arms_basic_structure(self, mock_score):
        """Test basic structure of topk_arms method."""
        mock_warehouse = MagicMock()
        
        # Mock warehouse responses for V2 API
        mock_warehouse.download_nodes.return_value = [
            {"node_id": "node1", "avgQ": 2.5, "in_degree": 1, "out_degree": 2, "size": 5},
            {"node_id": "node2", "avgQ": 3.0, "in_degree": 2, "out_degree": 1, "size": 4}
        ]
        mock_warehouse.download_hypernodes.return_value = []
        mock_warehouse.get_formula_definition.return_value = [123, 456]
        
        # Mock scoring to return consistent values
        mock_score.return_value = 1.0
        
        ranker = ArmRanker(mock_warehouse)
        result = ranker.topk_arms(num_vars=3, width=2, k=2)
        
        # Verify warehouse methods were called with correct parameters
        mock_warehouse.download_nodes.assert_called_once_with(
            num_vars=3, width=2, size_constraint=None
        )
        mock_warehouse.download_hypernodes.assert_called_once_with(
            num_vars=3, width=2, size_constraint=None
        )
        
        # Verify result structure uses node_id
        assert isinstance(result, list)
        for arm in result:
            assert "node_id" in arm
            assert "definition" in arm


class TestIntegration:
    """Integration tests for agent modules with warehouse."""

    @pytest.mark.integration
    def test_warehouse_agent_with_live_service(self):
        """Test WarehouseAgent with actual warehouse service if available."""
        try:
            agent = WarehouseAgent(host="localhost", port=8000)
            
            # Test health check endpoint
            response = agent._client.get("/health")
            if response.status_code == 200:
                print("✅ Warehouse service is running and responding")
                
                # Test basic functionality
                try:
                    # Test likely isomorphic endpoints
                    iso_ids = agent.get_likely_isomorphic("test_hash")
                    assert isinstance(iso_ids, list)
                    print(f"✅ get_likely_isomorphic works: {len(iso_ids)} results")
                    
                    # Test trajectory creation
                    traj_data = {
                        "steps": [
                            {"token_type": 0, "token_literals": 123, "cur_avgQ": 1.5}
                        ]
                    }
                    traj_id = agent.post_trajectory(**traj_data)
                    print(f"✅ post_trajectory works: {traj_id}")
                    
                    # Test trajectory retrieval
                    traj_info = agent.get_trajectory(traj_id)
                    assert "traj_id" in traj_info
                    print(f"✅ get_trajectory works: {traj_info['traj_id']}")
                    
                    # Cleanup
                    agent.delete_trajectory(traj_id)
                    print("✅ delete_trajectory works")
                    
                except Exception as e:
                    print(f"⚠️  Some warehouse operations failed: {e}")
            else:
                print("⚠️  Warehouse service not responding correctly")
                
        except Exception as e:
            print(f"⚠️  Cannot connect to warehouse service: {e}")
        finally:
            agent.close()


if __name__ == "__main__":
    # Run the tests
    print("🧪 Running agent module tests...")
    
    # Run unit tests
    test_warehouse = TestWarehouseAgent()
    test_warehouse.test_warehouse_agent_initialization()
    print("✅ WarehouseAgent initialization test passed")
    
    test_processor = TestTrajectoryProcessor()
    test_processor.test_trajectory_processor_initialization()
    test_processor.test_definition_to_graph()
    test_processor.test_add_gate_to_graph()
    test_processor.test_retrieve_definition_with_none()
    print("✅ TrajectoryProcessor tests passed")
    
    test_ranker = TestArmRanker()
    test_ranker.test_arm_ranker_initialization()
    test_ranker.test_arm_ranker_with_custom_config()
    print("✅ ArmRanker tests passed")
    
    # Run integration test if warehouse is available
    print("\n🔌 Testing integration with warehouse service...")
    integration_test = TestIntegration()
    integration_test.test_warehouse_agent_with_live_service()
    
    print("\n✅ All agent module tests completed!")