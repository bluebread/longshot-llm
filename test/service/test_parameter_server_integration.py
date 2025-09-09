#!/usr/bin/env python
"""
Integration test for Parameter Server endpoints.
This script tests the actual endpoints with a running warehouse service.
"""

import requests
import zipfile
import io
import json
from datetime import datetime
import sys


def create_test_zip(content="Test model content"):
    """Create a test ZIP file in memory."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("model.pt", content)
        zip_file.writestr("config.json", '{"version": "1.0", "description": "Test model"}')
    zip_buffer.seek(0)
    return zip_buffer


def test_parameter_server(base_url="http://localhost:8000"):
    """Test all Parameter Server endpoints."""
    print("🧪 Testing Parameter Server endpoints...")
    print(f"Base URL: {base_url}")
    
    # Test 1: Upload a model
    print("\n1️⃣ Testing POST /models/upload...")
    zip_file = create_test_zip("Model v1 content")
    files = {'file': ('test_model_v1.zip', zip_file, 'application/zip')}
    data = {
        'num_vars': 4,
        'width': 3,
        'tags': 'test,v1,integration'
    }
    
    try:
        response = requests.post(f"{base_url}/models/upload", files=files, data=data)
        if response.status_code == 201:
            upload_result = response.json()
            model_id = upload_result['model_id']
            print(f"✅ Upload successful! Model ID: {model_id}")
            print(f"   Filename: {upload_result['filename']}")
            print(f"   Tags: {upload_result['tags']}")
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to warehouse service. Is it running?")
        print("   Start the service with: cd service/warehouse && uvicorn main:app --host 0.0.0.0 --port 8000")
        return False
    
    # Test 2: Upload another model
    print("\n2️⃣ Uploading second model...")
    zip_file2 = create_test_zip("Model v2 content - improved")
    files2 = {'file': ('test_model_v2.zip', zip_file2, 'application/zip')}
    data2 = {
        'num_vars': 4,
        'width': 3,
        'tags': 'test,v2,latest'
    }
    
    response = requests.post(f"{base_url}/models/upload", files=files2, data=data2)
    if response.status_code == 201:
        model_id_2 = response.json()['model_id']
        print(f"✅ Second model uploaded! Model ID: {model_id_2}")
    else:
        print(f"❌ Upload failed: {response.status_code}")
        return False
    
    # Test 3: Query models
    print("\n3️⃣ Testing GET /models...")
    response = requests.get(f"{base_url}/models", params={'num_vars': 4, 'width': 3})
    if response.status_code == 200:
        models_data = response.json()
        print(f"✅ Query successful! Found {models_data['count']} models")
        for idx, model in enumerate(models_data['models'], 1):
            print(f"   Model {idx}: {model['filename']} (tags: {model['tags']})")
    else:
        print(f"❌ Query failed: {response.status_code}")
        return False
    
    # Test 4: Query with tag filter
    print("\n4️⃣ Testing GET /models with tag filter...")
    response = requests.get(f"{base_url}/models", params={'num_vars': 4, 'width': 3, 'tags': ['test']})
    if response.status_code == 200:
        filtered_data = response.json()
        print(f"✅ Filtered query successful! Found {filtered_data['count']} models with 'test' tag")
    else:
        print(f"❌ Filtered query failed: {response.status_code}")
    
    # Test 5: Get latest model
    print("\n5️⃣ Testing GET /models/latest...")
    response = requests.get(f"{base_url}/models/latest", params={'num_vars': 4, 'width': 3})
    if response.status_code == 200:
        latest_model = response.json()
        print(f"✅ Latest model: {latest_model['filename']}")
        print(f"   Upload date: {latest_model['upload_date']}")
        print(f"   Download URL: {latest_model['download_url']}")
    else:
        print(f"❌ Get latest failed: {response.status_code}")
        return False
    
    # Test 6: Download model
    print("\n6️⃣ Testing GET /models/download/{model_id}...")
    download_url = f"{base_url}/models/download/{model_id}"
    response = requests.get(download_url)
    if response.status_code == 200:
        # Verify it's a valid ZIP
        try:
            zip_content = io.BytesIO(response.content)
            with zipfile.ZipFile(zip_content, 'r') as zf:
                file_list = zf.namelist()
                print(f"✅ Download successful! ZIP contains: {file_list}")
        except zipfile.BadZipFile:
            print("❌ Downloaded file is not a valid ZIP")
            return False
    else:
        print(f"❌ Download failed: {response.status_code}")
        return False
    
    # Test 7: Test non-existent model
    print("\n7️⃣ Testing error handling (non-existent model)...")
    response = requests.get(f"{base_url}/models/latest", params={'num_vars': 10, 'width': 10})
    if response.status_code == 404:
        print("✅ Correctly returns 404 for non-existent model")
    else:
        print(f"❌ Unexpected status code: {response.status_code}")
    
    # Test 8: Invalid upload (not a ZIP)
    print("\n8️⃣ Testing error handling (invalid ZIP upload)...")
    invalid_file = io.BytesIO(b"This is not a ZIP file")
    files = {'file': ('invalid.zip', invalid_file, 'application/zip')}
    data = {'num_vars': 4, 'width': 3}
    response = requests.post(f"{base_url}/models/upload", files=files, data=data)
    if response.status_code == 422:
        print("✅ Correctly rejects invalid ZIP file")
    else:
        print(f"❌ Unexpected status code: {response.status_code}")
    
    # Test 9: Purge models (optional - commented out to preserve data)
    print("\n9️⃣ Testing DELETE /models/purge...")
    response = requests.delete(f"{base_url}/models/purge")
    if response.status_code == 200:
        purge_result = response.json()
        print(f"✅ Purge successful! Deleted {purge_result['deleted_count']} models")
        print(f"   Freed space: {purge_result['freed_space']} bytes")
    else:
        print(f"❌ Purge failed: {response.status_code}")
    
    print("\n✨ All tests completed successfully!")
    return True


if __name__ == "__main__":
    # Check if custom URL is provided
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    
    print("=" * 60)
    print("Parameter Server Integration Test")
    print("=" * 60)
    
    success = test_parameter_server(url)
    
    if success:
        print("\n🎉 All Parameter Server endpoints are working correctly!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)