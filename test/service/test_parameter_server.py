"""
Tests for the Parameter Server functionality in Warehouse service.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime
import io
import zipfile
from unittest.mock import MagicMock, patch
from bson import ObjectId


@pytest.fixture
def mock_gridfs():
    """Mock GridFS for testing."""
    with patch("service.warehouse.main.gridfs") as mock_gfs:
        yield mock_gfs


@pytest.fixture
def mock_mongodb():
    """Mock MongoDB for testing."""
    with patch("service.warehouse.main.mongodb") as mock_db:
        yield mock_db


@pytest.fixture
def client():
    """Create test client."""
    from service.warehouse.main import app
    return TestClient(app)


def create_test_zip_file(filename="test_model.zip"):
    """Create a test ZIP file in memory."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("model.txt", "Test model content")
        zip_file.writestr("config.json", '{"version": "1.0"}')
    zip_buffer.seek(0)
    return zip_buffer


class TestModelsEndpoint:
    """Test GET /models endpoint."""
    
    def test_get_models_success(self, client, mock_mongodb):
        """Test successful retrieval of models."""
        # Mock data
        test_file_doc = {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "filename": "model_v1.zip",
            "length": 1024,
            "metadata": {
                "num_vars": 4,
                "width": 3,
                "tags": ["production", "optimized"],
                "upload_date": datetime(2024, 1, 15, 10, 30)
            }
        }
        
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([test_file_doc]))
        mock_mongodb["fs.files"].find.return_value.sort.return_value = mock_cursor
        
        response = client.get("/models?num_vars=4&width=3")
        
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["models"]) == 1
        assert data["models"][0]["num_vars"] == 4
        assert data["models"][0]["width"] == 3
        assert data["models"][0]["tags"] == ["production", "optimized"]
    
    def test_get_models_with_tags_filter(self, client, mock_mongodb):
        """Test model retrieval with tag filtering."""
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_mongodb["fs.files"].find.return_value.sort.return_value = mock_cursor
        
        response = client.get("/models?num_vars=4&width=3&tags=production&tags=optimized")
        
        assert response.status_code == 200
        # Verify the query includes tag filter
        mock_mongodb["fs.files"].find.assert_called_once()
        call_args = mock_mongodb["fs.files"].find.call_args[0][0]
        assert "metadata.tags" in call_args
        assert call_args["metadata.tags"] == {"$all": ["production", "optimized"]}
    
    def test_get_models_missing_required_params(self, client):
        """Test error when required parameters are missing."""
        response = client.get("/models")
        assert response.status_code == 422
        
        response = client.get("/models?num_vars=4")
        assert response.status_code == 422
        
        response = client.get("/models?width=3")
        assert response.status_code == 422


class TestLatestModelEndpoint:
    """Test GET /models/latest endpoint."""
    
    def test_get_latest_model_success(self, client, mock_mongodb):
        """Test successful retrieval of latest model."""
        test_file_doc = {
            "_id": ObjectId("507f1f77bcf86cd799439011"),
            "filename": "latest_model.zip",
            "length": 2048,
            "metadata": {
                "num_vars": 4,
                "width": 3,
                "tags": ["latest"],
                "upload_date": datetime(2024, 1, 20, 14, 45)
            }
        }
        
        mock_mongodb["fs.files"].find_one.return_value = test_file_doc
        
        response = client.get("/models/latest?num_vars=4&width=3")
        
        assert response.status_code == 200
        data = response.json()
        assert data["num_vars"] == 4
        assert data["width"] == 3
        assert data["filename"] == "latest_model.zip"
        assert "latest" in data["tags"]
    
    def test_get_latest_model_not_found(self, client, mock_mongodb):
        """Test 404 when no model matches criteria."""
        mock_mongodb["fs.files"].find_one.return_value = None
        
        response = client.get("/models/latest?num_vars=5&width=4")
        
        assert response.status_code == 404
        assert "No model found" in response.json()["detail"]


class TestDownloadModelEndpoint:
    """Test GET /models/download/{model_id} endpoint."""
    
    def test_download_model_success(self, client, mock_gridfs):
        """Test successful model download."""
        model_id = "507f1f77bcf86cd799439011"
        
        # Mock GridFS file
        mock_file = MagicMock()
        mock_file.filename = "test_model.zip"
        mock_file.read.side_effect = [b"test content", b""]
        
        mock_gridfs.exists.return_value = True
        mock_gridfs.get.return_value = mock_file
        
        response = client.get(f"/models/download/{model_id}")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/zip"
        assert "attachment" in response.headers["content-disposition"]
        assert "test_model.zip" in response.headers["content-disposition"]
    
    def test_download_model_invalid_id(self, client):
        """Test error with invalid model ID format."""
        response = client.get("/models/download/invalid-id")
        
        assert response.status_code == 400
        assert "Invalid model ID format" in response.json()["detail"]
    
    def test_download_model_not_found(self, client, mock_gridfs):
        """Test 404 when model file doesn't exist."""
        model_id = "507f1f77bcf86cd799439011"
        mock_gridfs.exists.return_value = False
        
        response = client.get(f"/models/download/{model_id}")
        
        assert response.status_code == 404
        assert "Model file not found" in response.json()["detail"]


class TestUploadModelEndpoint:
    """Test POST /models/upload endpoint."""
    
    def test_upload_model_success(self, client, mock_gridfs):
        """Test successful model upload."""
        zip_file = create_test_zip_file()
        
        mock_gridfs.put.return_value = ObjectId("507f1f77bcf86cd799439011")
        
        response = client.post(
            "/models/upload",
            files={"file": ("test_model.zip", zip_file, "application/zip")},
            data={
                "num_vars": 4,
                "width": 3,
                "tags": "production,optimized"
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["filename"] == "test_model.zip"
        assert data["num_vars"] == 4
        assert data["width"] == 3
        assert "production" in data["tags"]
        assert "optimized" in data["tags"]
        assert data["message"] == "Model uploaded successfully"
    
    def test_upload_model_invalid_zip(self, client):
        """Test error when uploaded file is not a valid ZIP."""
        invalid_file = io.BytesIO(b"not a zip file")
        
        response = client.post(
            "/models/upload",
            files={"file": ("invalid.zip", invalid_file, "application/zip")},
            data={
                "num_vars": 4,
                "width": 3
            }
        )
        
        assert response.status_code == 422
        assert "not a valid ZIP archive" in response.json()["detail"]
    
    def test_upload_model_without_tags(self, client, mock_gridfs):
        """Test upload without tags."""
        zip_file = create_test_zip_file()
        mock_gridfs.put.return_value = ObjectId("507f1f77bcf86cd799439011")
        
        response = client.post(
            "/models/upload",
            files={"file": ("test_model.zip", zip_file, "application/zip")},
            data={
                "num_vars": 4,
                "width": 3
            }
        )
        
        assert response.status_code == 201
        data = response.json()
        assert data["tags"] == []
    
    def test_upload_model_invalid_params(self, client):
        """Test validation of num_vars and width parameters."""
        zip_file = create_test_zip_file()
        
        # Test num_vars out of range
        response = client.post(
            "/models/upload",
            files={"file": ("test_model.zip", zip_file, "application/zip")},
            data={
                "num_vars": 100,  # Out of range
                "width": 3
            }
        )
        assert response.status_code == 422
        
        # Test width out of range
        zip_file.seek(0)
        response = client.post(
            "/models/upload",
            files={"file": ("test_model.zip", zip_file, "application/zip")},
            data={
                "num_vars": 4,
                "width": 50  # Out of range
            }
        )
        assert response.status_code == 422


class TestPurgeModelsEndpoint:
    """Test DELETE /models/purge endpoint."""
    
    def test_purge_models_success(self, client, mock_mongodb, mock_gridfs):
        """Test successful purge of all models."""
        # Mock file documents
        test_files = [
            {"_id": ObjectId("507f1f77bcf86cd799439011"), "length": 1024},
            {"_id": ObjectId("507f1f77bcf86cd799439012"), "length": 2048}
        ]
        
        # Create mock cursors
        mock_cursor1 = MagicMock()
        mock_cursor1.__iter__ = MagicMock(return_value=iter(test_files))
        
        mock_cursor2 = MagicMock()
        mock_cursor2.__iter__ = MagicMock(return_value=iter(test_files))
        
        mock_mongodb["fs.files"].find.side_effect = [mock_cursor1, mock_cursor2]
        
        response = client.delete("/models/purge")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_count"] == 2
        assert data["freed_space"] == 3072
        assert "Successfully purged 2 models" in data["message"]
        
        # Verify delete was called for each file
        assert mock_gridfs.delete.call_count == 2
    
    def test_purge_models_empty(self, client, mock_mongodb):
        """Test purge when no models exist."""
        mock_cursor = MagicMock()
        mock_cursor.__iter__ = MagicMock(return_value=iter([]))
        mock_mongodb["fs.files"].find.return_value = mock_cursor
        
        response = client.delete("/models/purge")
        
        assert response.status_code == 200
        data = response.json()
        assert data["deleted_count"] == 0
        assert data["freed_space"] == 0
    
    def test_purge_models_error(self, client, mock_mongodb):
        """Test error handling during purge."""
        mock_mongodb["fs.files"].find.side_effect = Exception("Database error")
        
        response = client.delete("/models/purge")
        
        assert response.status_code == 500
        assert "Failed to purge models" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])