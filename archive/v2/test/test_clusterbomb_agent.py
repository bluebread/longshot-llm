"""
Test suite for the ClusterbombAgent client.

This test verifies the ClusterbombAgent functionality including:
- Health check operations
- Weapon rollout requests
- Async operations
- Error handling
"""

import pytest
import pytest_asyncio
import httpx
from unittest.mock import Mock, patch, AsyncMock
from longshot.service import ClusterbombAgent, AsyncClusterbombAgent
from longshot.models import WeaponRolloutRequest, WeaponRolloutResponse


class TestClusterbombAgent:
    """Test suite for ClusterbombAgent client."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock HTTP client for testing."""
        with patch('httpx.Client') as mock:
            yield mock
    
    @pytest.fixture
    def agent(self, mock_client):
        """Create a ClusterbombAgent instance with mocked client."""
        agent = ClusterbombAgent('localhost', 8060)
        return agent
    
    def test_agent_initialization(self):
        """Test ClusterbombAgent initialization."""
        agent = ClusterbombAgent('localhost', 8060)
        assert agent._client is not None
        assert agent._client.base_url == 'http://localhost:8060'
        agent.close()
    
    def test_health_check(self, agent, mock_client):
        """Test health check method."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {"status": "healthy", "service": "clusterbomb"}
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.get.return_value = mock_response
        agent._client.get = Mock(return_value=mock_response)
        
        # Call health check
        result = agent.health_check()
        
        # Verify
        assert result["status"] == "healthy"
        assert result["service"] == "clusterbomb"
        agent._client.get.assert_called_once_with("/health")
    
    def test_weapon_rollout(self, agent, mock_client):
        """Test weapon rollout method."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "total_steps": 50,
            "num_trajectories": 5,
            "new_nodes_created": ["node1", "node2"],
            "evopaths": [["node1", "node2"]]
        }
        mock_response.raise_for_status = Mock()
        
        mock_client.return_value.post.return_value = mock_response
        agent._client.post = Mock(return_value=mock_response)
        
        # Create prefix trajectory for V2 schema
        prefix_traj = [
            (0, 3, 0.5),  # (token_type, token_literals, cur_avgQ)
            (0, 4, 1.0)
        ]
        
        # Call weapon rollout
        result = agent.weapon_rollout(
            num_vars=3,
            width=2,
            size=5,
            steps_per_trajectory=10,
            num_trajectories=5,
            prefix_traj=prefix_traj,
            seed=42
        )
        
        # Verify response
        assert isinstance(result, WeaponRolloutResponse)
        assert result.total_steps == 50
        assert result.num_trajectories == 5
        assert result.new_nodes_created == ["node1", "node2"]
        assert result.evopaths == [["node1", "node2"]]
        
        # Verify request was made correctly
        agent._client.post.assert_called_once()
        call_args = agent._client.post.call_args
        assert call_args[0][0] == "/weapon/rollout"
        
        # Check request body
        request_body = call_args[1]["json"]
        assert request_body["num_vars"] == 3
        assert request_body["width"] == 2
        assert request_body["size"] == 5
        assert request_body["steps_per_trajectory"] == 10
        assert request_body["num_trajectories"] == 5
        assert "prefix_traj" in request_body
        assert len(request_body["prefix_traj"]) == 2
        assert request_body["seed"] == 42
    
    def test_weapon_rollout_without_optional_params(self, agent, mock_client):
        """Test weapon rollout without optional parameters."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "total_steps": 30,
            "num_trajectories": 3,
            "new_nodes_created": [],
            "evopaths": []
        }
        mock_response.raise_for_status = Mock()
        
        agent._client.post = Mock(return_value=mock_response)
        
        # Create minimal prefix trajectory
        prefix_traj = [
            (0, 1, 0.0)  # (token_type, token_literals, cur_avgQ)
        ]
        
        # Call without optional params
        result = agent.weapon_rollout(
            num_vars=2,
            width=2,
            size=3,
            steps_per_trajectory=10,
            num_trajectories=3,
            prefix_traj=prefix_traj
        )
        
        # Verify response
        assert result.total_steps == 30
        assert result.num_trajectories == 3
        
        # Check that optional fields are not in request
        call_args = agent._client.post.call_args
        request_body = call_args[1]["json"]
        assert "seed" not in request_body or request_body["seed"] is None
        assert "prefix_traj" in request_body
        assert len(request_body["prefix_traj"]) == 1
    
    def test_context_manager(self):
        """Test agent as context manager."""
        with ClusterbombAgent('localhost', 8060) as agent:
            assert agent._client is not None
        # Client should be closed after exiting context
    
    def test_error_handling(self, agent):
        """Test error handling for HTTP errors."""
        # Setup mock to raise HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=Mock(), response=Mock(status_code=500)
        )
        agent._client.get = Mock(return_value=mock_response)
        
        # Should raise exception
        with pytest.raises(httpx.HTTPStatusError):
            agent.health_check()


class TestAsyncClusterbombAgent:
    """Test suite for AsyncClusterbombAgent."""
    
    @pytest_asyncio.fixture
    async def mock_async_client(self):
        """Mock async HTTP client for testing."""
        with patch('httpx.AsyncClient') as mock:
            yield mock
    
    @pytest_asyncio.fixture
    async def async_agent(self, mock_async_client):
        """Create an AsyncClusterbombAgent instance with mocked client."""
        agent = AsyncClusterbombAgent('localhost', 8060)
        return agent
    
    @pytest.mark.asyncio
    async def test_async_agent_initialization(self):
        """Test AsyncClusterbombAgent initialization."""
        agent = AsyncClusterbombAgent('localhost', 8060)
        assert agent._client is not None
        assert agent._client.base_url == 'http://localhost:8060'
        await agent.close()
    
    @pytest.mark.asyncio
    async def test_async_health_check(self, async_agent):
        """Test async health check method."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {"status": "healthy", "service": "clusterbomb"}
        mock_response.raise_for_status = Mock()
        
        async_agent._client.get = AsyncMock(return_value=mock_response)
        
        # Call health check
        result = await async_agent.health_check()
        
        # Verify
        assert result["status"] == "healthy"
        assert result["service"] == "clusterbomb"
        async_agent._client.get.assert_called_once_with("/health")
    
    @pytest.mark.asyncio
    async def test_async_weapon_rollout(self, async_agent):
        """Test async weapon rollout method."""
        # Setup mock response
        mock_response = Mock()
        mock_response.json.return_value = {
            "total_steps": 100,
            "num_trajectories": 10,
            "new_nodes_created": ["async_node1"],
            "evopaths": []
        }
        mock_response.raise_for_status = Mock()
        
        async_agent._client.post = AsyncMock(return_value=mock_response)
        
        # Create prefix trajectory for V2 schema
        prefix_traj = [
            (0, 5, 1.5),  # ADD
            (1, 6, 2.0)   # DELETE
        ]
        
        # Call weapon rollout
        result = await async_agent.weapon_rollout(
            num_vars=4,
            width=3,
            size=8,
            steps_per_trajectory=10,
            num_trajectories=10,
            prefix_traj=prefix_traj,
            seed=123
        )
        
        # Verify response
        assert isinstance(result, WeaponRolloutResponse)
        assert result.total_steps == 100
        assert result.num_trajectories == 10
        
        # Verify request
        async_agent._client.post.assert_called_once()
        call_args = async_agent._client.post.call_args
        assert call_args[0][0] == "/weapon/rollout"
    
    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async agent as context manager."""
        async with AsyncClusterbombAgent('localhost', 8060) as agent:
            assert agent._client is not None
        # Client should be closed after exiting context
    
    @pytest.mark.asyncio
    async def test_async_error_handling(self, async_agent):
        """Test async error handling."""
        # Setup mock to raise HTTP error
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Error", request=Mock(), response=Mock(status_code=422)
        )
        async_agent._client.get = AsyncMock(return_value=mock_response)
        
        # Should raise exception
        with pytest.raises(httpx.HTTPStatusError):
            await async_agent.health_check()


class TestClusterbombAgentIntegration:
    """Integration tests with actual service (if running)."""
    
    @pytest.mark.integration
    def test_real_service_health_check(self):
        """Test against real service if available."""
        try:
            agent = ClusterbombAgent('localhost', 8060)
            result = agent.health_check()
            assert result["status"] == "healthy"
            assert result["service"] == "clusterbomb"
            agent.close()
        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Clusterbomb service not available")
    
    @pytest.mark.integration
    def test_real_service_weapon_rollout(self):
        """Test weapon rollout against real service if available."""
        try:
            agent = ClusterbombAgent('localhost', 8060)
            
            # Create test prefix trajectory
            prefix_traj = [
                (0, 3, 0.5),  # (token_type, token_literals, cur_avgQ)
                (0, 4, 1.0)
            ]
            
            # Try a small rollout
            result = agent.weapon_rollout(
                num_vars=2,
                width=2,
                size=3,
                steps_per_trajectory=5,
                num_trajectories=1,
                prefix_traj=prefix_traj,
                seed=42
            )
            
            assert result.total_steps == 5
            assert result.num_trajectories == 1
            
            agent.close()
        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Clusterbomb service not available")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_real_service(self):
        """Test async agent against real service if available."""
        try:
            async with AsyncClusterbombAgent('localhost', 8060) as agent:
                result = await agent.health_check()
                assert result["status"] == "healthy"
                assert result["service"] == "clusterbomb"
        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Clusterbomb service not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])