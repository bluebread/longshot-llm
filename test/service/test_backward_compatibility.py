"""
Test backward compatibility of WarehouseClient and AsyncWarehouseClient.
Ensures that existing code using these clients continues to work.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from longshot.service.warehouse import WarehouseClient, AsyncWarehouseClient


class TestBackwardCompatibility:
    """Test that existing code without models_output_folder still works."""
    
    def test_sync_client_trajectory_methods_backward_compat(self):
        """Test that trajectory methods work without models_output_folder (sync)."""
        with patch("longshot.service.warehouse.httpx.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # This is how existing code would use the client
            # No models_output_folder parameter - should still work
            client = WarehouseClient("localhost", 8000)
            
            # Test trajectory methods still work
            mock_response = MagicMock()
            mock_response.json.return_value = {"traj_id": "123", "steps": []}
            mock_client.get.return_value = mock_response
            
            # These should all work as before
            result = client.get_trajectory("test_id")
            assert result["traj_id"] == "123"
            
            mock_response.json.return_value = {"traj_id": "456"}
            mock_client.post.return_value = mock_response
            traj_id = client.post_trajectory(steps=[[0, 1, 0.5]])
            assert traj_id == "456"
            
            mock_client.put.return_value = mock_response
            client.put_trajectory(traj_id="456", steps=[[1, 2, 0.7]])
            
            mock_client.delete.return_value = mock_response
            client.delete_trajectory("456")
            
            mock_response.json.return_value = {"trajectories": []}
            mock_client.get.return_value = mock_response
            dataset = client.get_trajectory_dataset()
            assert "trajectories" in dataset
            
            client.close()
    
    @pytest.mark.asyncio
    async def test_async_client_trajectory_methods_backward_compat(self):
        """Test that trajectory methods work without models_output_folder (async)."""
        with patch("longshot.service.warehouse.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # This is how existing code would use the async client
            # No models_output_folder parameter - should still work
            client = AsyncWarehouseClient("localhost", 8000)
            
            # Test trajectory methods still work
            mock_response = MagicMock()
            mock_response.json.return_value = {"traj_id": "123", "steps": []}
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            
            # These should all work as before
            result = await client.get_trajectory("test_id")
            assert result["traj_id"] == "123"
            
            mock_response.json.return_value = {"traj_id": "456"}
            mock_client.post.return_value = mock_response
            traj_id = await client.post_trajectory(steps=[[0, 1, 0.5]])
            assert traj_id == "456"
            
            mock_client.put.return_value = mock_response
            await client.put_trajectory(traj_id="456", steps=[[1, 2, 0.7]])
            
            mock_client.delete.return_value = mock_response
            await client.delete_trajectory("456")
            
            mock_response.json.return_value = {"trajectories": []}
            mock_client.get.return_value = mock_response
            dataset = await client.get_trajectory_dataset()
            assert "trajectories" in dataset
            
            await client.aclose()
    
    def test_sync_client_with_positional_args(self):
        """Test that existing code using positional arguments still works."""
        with patch("longshot.service.warehouse.httpx.Client"):
            # Existing code might use positional arguments
            client1 = WarehouseClient("localhost", 8000)
            assert client1._models_output_folder.name == "warehouse_models"
            client1.close()
            
            # New code can specify models folder
            client2 = WarehouseClient("localhost", 8000, "/custom/path")
            assert str(client2._models_output_folder) == "/custom/path"
            client2.close()
            
            # Or use keyword argument
            client3 = WarehouseClient("localhost", 8000, models_output_folder="/another/path")
            assert str(client3._models_output_folder) == "/another/path"
            client3.close()
    
    @pytest.mark.asyncio
    async def test_async_client_with_positional_args(self):
        """Test that existing async code using positional arguments still works."""
        with patch("longshot.service.warehouse.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Existing code might use positional arguments
            client1 = AsyncWarehouseClient("localhost", 8000)
            assert client1._models_output_folder.name == "warehouse_models"
            await client1.aclose()
            
            # New code can specify models folder
            client2 = AsyncWarehouseClient("localhost", 8000, "/custom/path")
            assert str(client2._models_output_folder) == "/custom/path"
            await client2.aclose()
            
            # Or use keyword argument
            client3 = AsyncWarehouseClient("localhost", 8000, models_output_folder="/another/path")
            assert str(client3._models_output_folder) == "/another/path"
            await client3.aclose()
    
    def test_context_manager_backward_compat(self):
        """Test that context managers work without models_output_folder."""
        with patch("longshot.service.warehouse.httpx.Client"):
            # Sync context manager
            with WarehouseClient("localhost", 8000) as client:
                assert client._models_output_folder.name == "warehouse_models"
    
    @pytest.mark.asyncio
    async def test_async_context_manager_backward_compat(self):
        """Test that async context managers work without models_output_folder."""
        with patch("longshot.service.warehouse.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Async context manager
            async with AsyncWarehouseClient("localhost", 8000) as client:
                assert client._models_output_folder.name == "warehouse_models"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])