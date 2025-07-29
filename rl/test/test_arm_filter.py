import pytest
import httpx
from lsutils import encode_float64_to_base64

host = "localhost"
port = 8050
armfilter_url = f"http://{host}:{port}"

@pytest.fixture
def client():
    with httpx.Client(base_url=armfilter_url) as client:
        yield client

def test_topk_arms_success(client):
    """
    Test the /topk_arms endpoint with required parameters.
    """
    params = {
        "num_vars": 10,
        "width": 5,
        "k": 2,
        "size": 3
    }
    response = client.get("/topk_arms", params=params)
    assert response.status_code == 200
    data = response.json()
    assert "top_k_arms" in data
    assert isinstance(data["top_k_arms"], list)
    assert len(data["top_k_arms"]) == 2
    # Check structure of the first arm
    arm1 = data["top_k_arms"][0]
    assert "formula_id" in arm1
    assert "definition" in arm1
    assert isinstance(arm1["definition"], list)
    
def test_topk_arms_missing_parameter(client):
    """
    Test that a 422 error is returned if a required parameter is missing.
    """
    params = {
        "num_vars": 10,
        "k": 2
        # 'width' is missing
    }
    response = client.get("/topk_arms", params=params)
    assert response.status_code == 422

def test_topk_arms_invalid_parameter_type(client):
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
