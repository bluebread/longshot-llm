"""
Tests for Parameter Server functionality in WarehouseClient and AsyncWarehouseClient.
"""

import pytest
import httpx
from pathlib import Path
import tempfile
import zipfile
import json
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import asyncio

from longshot.service.warehouse import WarehouseClient, AsyncWarehouseClient


@pytest.fixture
def temp_models_folder():
    """Create a temporary folder for models output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_httpx_client():
    """Mock httpx.Client for testing."""
    with patch("longshot.service.warehouse.httpx.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        yield mock_client


@pytest.fixture
def mock_httpx_async_client():
    """Mock httpx.AsyncClient for testing."""
    with patch("longshot.service.warehouse.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        yield mock_client


def create_test_zip_file(folder: Path, name: str = "test_model.zip") -> Path:
    """Create a test ZIP file in the specified folder."""
    zip_path = folder / name
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("model.pt", b"model data")
        zf.writestr("config.json", json.dumps({"version": "1.0"}))
    return zip_path


class TestWarehouseClientParameterServer:
    """Test Parameter Server methods in WarehouseClient."""
    
    def test_backward_compatibility_no_models_folder(self):
        """Test backward compatibility - client works without models_output_folder."""
        # This should work for backward compatibility
        client = WarehouseClient("localhost", 8000)
        
        # Should create default folder
        assert client._models_output_folder == Path("./warehouse_models")
        assert client._models_output_folder.exists()
        
        # Clean up
        client.close()
        if Path("./warehouse_models").exists():
            import shutil
            shutil.rmtree("./warehouse_models")
    
    def test_init_with_models_folder(self, temp_models_folder):
        """Test client initialization with models output folder."""
        client = WarehouseClient("localhost", 8000, str(temp_models_folder))
        assert client._models_output_folder == temp_models_folder
        assert temp_models_folder.exists()
        client.close()
    
    def test_init_creates_models_folder(self):
        """Test that initialization creates the models folder if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_folder = Path(temp_dir) / "models" / "subfolder"
            assert not models_folder.exists()
            
            client = WarehouseClient("localhost", 8000, str(models_folder))
            assert models_folder.exists()
            client.close()
    
    def test_get_models(self, mock_httpx_client, temp_models_folder):
        """Test getting models with filters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"model_id": "123", "filename": "model.zip"}],
            "count": 1
        }
        mock_httpx_client.get.return_value = mock_response
        
        client = WarehouseClient("localhost", 8000, str(temp_models_folder))
        result = client.get_models(num_vars=4, width=3, tags=["production"])
        
        assert result["count"] == 1
        mock_httpx_client.get.assert_called_once_with(
            "/models", 
            params={"num_vars": 4, "width": 3, "tags": ["production"]}
        )
        client.close()
    
    def test_get_latest_model(self, mock_httpx_client, temp_models_folder):
        """Test getting the latest model."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model_id": "456",
            "filename": "latest_model.zip",
            "upload_date": "2024-01-20T10:00:00"
        }
        mock_httpx_client.get.return_value = mock_response
        
        client = WarehouseClient("localhost", 8000, str(temp_models_folder))
        result = client.get_latest_model(num_vars=4, width=3)
        
        assert result["model_id"] == "456"
        mock_httpx_client.get.assert_called_once_with(
            "/models/latest",
            params={"num_vars": 4, "width": 3}
        )
        client.close()
    
    def test_download_model(self, mock_httpx_client, temp_models_folder):
        """Test downloading a model file."""
        mock_response = MagicMock()
        mock_response.content = b"zip file content"
        mock_response.headers = {"content-disposition": 'attachment; filename="model.zip"'}
        mock_httpx_client.get.return_value = mock_response
        
        client = WarehouseClient("localhost", 8000, str(temp_models_folder))
        output_path = client.download_model("model_id_123")
        
        assert output_path.name == "model.zip"
        assert output_path.parent == temp_models_folder
        assert output_path.read_bytes() == b"zip file content"
        
        mock_httpx_client.get.assert_called_once_with("/models/download/model_id_123")
        client.close()
    
    def test_download_model_custom_filename(self, mock_httpx_client, temp_models_folder):
        """Test downloading a model with custom filename."""
        mock_response = MagicMock()
        mock_response.content = b"zip content"
        mock_response.headers = {}
        mock_httpx_client.get.return_value = mock_response
        
        client = WarehouseClient("localhost", 8000, str(temp_models_folder))
        output_path = client.download_model("model_id", "custom_name.zip")
        
        assert output_path.name == "custom_name.zip"
        assert output_path.read_bytes() == b"zip content"
        client.close()
    
    def test_upload_model(self, mock_httpx_client, temp_models_folder):
        """Test uploading a model ZIP file."""
        # Create a test ZIP file
        zip_path = create_test_zip_file(temp_models_folder)
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model_id": "789",
            "filename": "test_model.zip",
            "message": "Model uploaded successfully"
        }
        mock_httpx_client.post.return_value = mock_response
        
        client = WarehouseClient("localhost", 8000, str(temp_models_folder))
        result = client.upload_model(
            str(zip_path),
            num_vars=4,
            width=3,
            tags=["test", "v1"]
        )
        
        assert result["model_id"] == "789"
        
        # Verify the call
        call_args = mock_httpx_client.post.call_args
        assert call_args[0][0] == "/models/upload"
        assert "files" in call_args[1]
        assert "data" in call_args[1]
        assert call_args[1]["data"]["tags"] == "test,v1"
        
        client.close()
    
    def test_upload_model_file_not_found(self, temp_models_folder):
        """Test upload error when file doesn't exist."""
        client = WarehouseClient("localhost", 8000, str(temp_models_folder))
        
        with pytest.raises(FileNotFoundError, match="ZIP file not found"):
            client.upload_model("nonexistent.zip", num_vars=4, width=3)
        
        client.close()
    
    def test_upload_model_invalid_zip(self, temp_models_folder):
        """Test upload error with invalid ZIP file."""
        # Create an invalid ZIP file
        invalid_file = temp_models_folder / "invalid.zip"
        invalid_file.write_text("not a zip")
        
        client = WarehouseClient("localhost", 8000, str(temp_models_folder))
        
        with pytest.raises(zipfile.BadZipFile):
            client.upload_model(str(invalid_file), num_vars=4, width=3)
        
        client.close()
    
    def test_create_model_zip(self, temp_models_folder):
        """Test creating a ZIP file from dictionary."""
        client = WarehouseClient("localhost", 8000, str(temp_models_folder))
        
        files_dict = {
            "model.pt": b"model binary data",
            "config.json": b'{"version": "2.0"}'
        }
        
        output_path = client.create_model_zip(files_dict, "created_model.zip")
        
        assert output_path.name == "created_model.zip"
        assert output_path.exists()
        
        # Verify ZIP contents
        with zipfile.ZipFile(output_path, 'r') as zf:
            assert set(zf.namelist()) == {"model.pt", "config.json"}
            assert zf.read("model.pt") == b"model binary data"
        
        client.close()
    
    def test_purge_models(self, mock_httpx_client, temp_models_folder):
        """Test purging all models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "deleted_count": 5,
            "freed_space": 10240
        }
        mock_httpx_client.delete.return_value = mock_response
        
        client = WarehouseClient("localhost", 8000, str(temp_models_folder))
        result = client.purge_models()
        
        assert result["success"] is True
        assert result["deleted_count"] == 5
        
        mock_httpx_client.delete.assert_called_once_with("/models/purge")
        client.close()


class TestAsyncWarehouseClientParameterServer:
    """Test Parameter Server methods in AsyncWarehouseClient."""
    
    @pytest.mark.asyncio
    async def test_backward_compatibility_no_models_folder(self):
        """Test backward compatibility - async client works without models_output_folder."""
        # This should work for backward compatibility
        client = AsyncWarehouseClient("localhost", 8000)
        
        # Should create default folder
        assert client._models_output_folder == Path("./warehouse_models")
        assert client._models_output_folder.exists()
        
        # Clean up
        await client.aclose()
        if Path("./warehouse_models").exists():
            import shutil
            shutil.rmtree("./warehouse_models")
    
    @pytest.mark.asyncio
    async def test_init_with_models_folder(self, temp_models_folder):
        """Test async client initialization with models output folder."""
        client = AsyncWarehouseClient("localhost", 8000, str(temp_models_folder))
        assert client._models_output_folder == temp_models_folder
        assert temp_models_folder.exists()
        await client.aclose()
    
    @pytest.mark.asyncio
    async def test_get_models(self, mock_httpx_async_client, temp_models_folder):
        """Test async getting models with filters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "models": [{"model_id": "123", "filename": "model.zip"}],
            "count": 1
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.get.return_value = mock_response
        
        client = AsyncWarehouseClient("localhost", 8000, str(temp_models_folder))
        result = await client.get_models(num_vars=4, width=3, tags=["async"])
        
        assert result["count"] == 1
        mock_httpx_async_client.get.assert_called_once_with(
            "/models",
            params={"num_vars": 4, "width": 3, "tags": ["async"]}
        )
        await client.aclose()
    
    @pytest.mark.asyncio
    async def test_get_latest_model(self, mock_httpx_async_client, temp_models_folder):
        """Test async getting the latest model."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model_id": "async_456",
            "filename": "async_latest.zip"
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.get.return_value = mock_response
        
        client = AsyncWarehouseClient("localhost", 8000, str(temp_models_folder))
        result = await client.get_latest_model(num_vars=5, width=4)
        
        assert result["model_id"] == "async_456"
        await client.aclose()
    
    @pytest.mark.asyncio
    async def test_download_model(self, mock_httpx_async_client, temp_models_folder):
        """Test async downloading a model file."""
        mock_response = MagicMock()
        mock_response.content = b"async zip content"
        mock_response.headers = {"content-disposition": 'attachment; filename="async_model.zip"'}
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.get.return_value = mock_response
        
        # Mock aiofiles
        with patch("aiofiles.open", create=True) as mock_aiofiles:
            mock_file = AsyncMock()
            mock_aiofiles.return_value.__aenter__.return_value = mock_file
            
            client = AsyncWarehouseClient("localhost", 8000, str(temp_models_folder))
            output_path = await client.download_model("async_model_id")
            
            assert output_path.name == "async_model.zip"
            mock_file.write.assert_called_once_with(b"async zip content")
            await client.aclose()
    
    @pytest.mark.asyncio
    async def test_upload_model(self, mock_httpx_async_client, temp_models_folder):
        """Test async uploading a model ZIP file."""
        # Create a test ZIP file
        zip_path = create_test_zip_file(temp_models_folder, "async_model.zip")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "model_id": "async_789",
            "filename": "async_model.zip"
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.post.return_value = mock_response
        
        client = AsyncWarehouseClient("localhost", 8000, str(temp_models_folder))
        result = await client.upload_model(
            str(zip_path),
            num_vars=6,
            width=5,
            tags=["async", "test"]
        )
        
        assert result["model_id"] == "async_789"
        await client.aclose()
    
    @pytest.mark.asyncio
    async def test_create_model_zip(self, temp_models_folder):
        """Test async creating a ZIP file from dictionary."""
        # Mock aiofiles
        with patch("aiofiles.open", create=True) as mock_aiofiles:
            mock_file = AsyncMock()
            mock_aiofiles.return_value.__aenter__.return_value = mock_file
            
            client = AsyncWarehouseClient("localhost", 8000, str(temp_models_folder))
            
            files_dict = {
                "async_model.pt": b"async model data",
                "async_config.json": b'{"async": true}'
            }
            
            output_path = await client.create_model_zip(files_dict, "async_created.zip")
            
            assert output_path.name == "async_created.zip"
            # Verify write was called with ZIP content
            assert mock_file.write.called
            
            await client.aclose()
    
    @pytest.mark.asyncio
    async def test_purge_models(self, mock_httpx_async_client, temp_models_folder):
        """Test async purging all models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "success": True,
            "deleted_count": 10
        }
        mock_response.raise_for_status = MagicMock()
        mock_httpx_async_client.delete.return_value = mock_response
        
        client = AsyncWarehouseClient("localhost", 8000, str(temp_models_folder))
        result = await client.purge_models()
        
        assert result["success"] is True
        assert result["deleted_count"] == 10
        
        mock_httpx_async_client.delete.assert_called_once_with("/models/purge")
        await client.aclose()
    
    @pytest.mark.asyncio
    async def test_context_manager(self, temp_models_folder):
        """Test async context manager functionality."""
        async with AsyncWarehouseClient("localhost", 8000, str(temp_models_folder)) as client:
            assert client._models_output_folder == temp_models_folder
            # Client should be properly initialized and closed automatically


if __name__ == "__main__":
    pytest.main([__file__, "-v"])