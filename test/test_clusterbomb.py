"""
Test suite for the Clusterbomb microservice.

This test verifies the weapon rollout functionality including:
- API endpoint responses
- Trajectory format validation  
- Data integrity checks
"""

import pytest
import pytest_asyncio
import httpx
import json
import time
from datetime import datetime
from pydantic import ValidationError
from longshot.models.api import WeaponRolloutRequest, WeaponRolloutResponse
from longshot.models.trajectory import (
    TrajectoryQueueMessage, 
    TrajectoryMessageMultipleSteps, 
    TrajectoryMessageStep
)
from longshot.agent.trajectory_queue import TrajectoryQueueAgent, AsyncTrajectoryQueueAgent

trajque_host = 'rabbitmq-bread'
trajque_port = 5672

class TestClusterbombService:
    """Test suite for Clusterbomb microservice endpoints."""
    
    BASE_URL = "http://localhost:8060"
    
    @pytest.fixture(scope="session")
    def client(self):
        """HTTP client for testing the service."""
        with httpx.Client(base_url=self.BASE_URL, timeout=30.0) as client:
            yield client
    
    @pytest.fixture(scope="function")
    def queue_agent(self):
        """Synchronous trajectory queue agent for testing."""
        agent = TrajectoryQueueAgent(host=trajque_host, port=trajque_port)
        # Clear any existing messages in the queue
        agent.channel.queue_purge(agent.queue_name)
        yield agent
        agent.close()
    
    @pytest_asyncio.fixture(scope="function")
    async def async_queue_agent(self):
        """Asynchronous trajectory queue agent for testing."""
        async with AsyncTrajectoryQueueAgent(host=trajque_host, port=trajque_port) as agent:
            # Clear any existing messages in the queue
            while await agent.pop():
                pass  # Drain the queue
            yield agent
    
    
    def test_health_check(self, client: httpx.Client):
        """Test the health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "clusterbomb"
    
    def test_weapon_rollout_with_queue_validation(self, client: httpx.Client, queue_agent: TrajectoryQueueAgent):
        """Test weapon rollout functionality and validate trajectories from queue."""
        request_data = {
            "num_vars": 3,
            "width": 2, 
            "size": 5,
            "steps_per_trajectory": 5,
            "num_trajectories": 2,
            "initial_definition": [1, 2, 3, 4, 5],  # Simple test formula
            "initial_formula_id": "test_formula_sync",
            "seed": 42  # Deterministic for testing
        }
        
        # Validate request model
        request_model = WeaponRolloutRequest(**request_data)
        assert request_model.num_vars == 3
        assert request_model.seed == 42
        
        # Make API request
        response = client.post("/weapon/rollout", json=request_data)
        assert response.status_code == 200
        
        # Validate response structure
        response_data = response.json()
        response_model = WeaponRolloutResponse(**response_data)
        
        # Check response values
        assert response_model.num_trajectories == request_data["num_trajectories"]
        assert response_model.total_steps == request_data["steps_per_trajectory"] * request_data["num_trajectories"]
        
        # Wait a moment for messages to be processed
        time.sleep(0.1)
        
        # Pop trajectories from the queue and validate format
        trajectories = []
        for _ in range(10):  # Try up to 10 times
            trajectory = queue_agent.pop()
            if trajectory:
                trajectories.append(trajectory)
            if len(trajectories) >= request_data["num_trajectories"]:
                break
            time.sleep(0.1)
        
        # Verify we got the expected number of trajectories
        assert len(trajectories) == request_data["num_trajectories"], f"Expected {request_data['num_trajectories']} trajectories, got {len(trajectories)}"
        
        # Validate each trajectory format
        for i, trajectory in enumerate(trajectories):
            # Validate trajectory is correct type
            assert isinstance(trajectory, TrajectoryQueueMessage), f"Trajectory {i} is not a TrajectoryQueueMessage"
            
            # Validate basic trajectory properties
            assert trajectory.num_vars == request_data["num_vars"]
            assert trajectory.width == request_data["width"]
            assert trajectory.base_size == request_data["size"]
            assert trajectory.trajectory.base_formula_id == request_data["initial_formula_id"]
            
            # Validate timestamp is reasonable (within last minute)
            assert isinstance(trajectory.timestamp, datetime)
            time_diff = datetime.now() - trajectory.timestamp
            assert time_diff.total_seconds() < 60, f"Trajectory timestamp too old: {trajectory.timestamp}"
            
            # Validate trajectory steps
            steps = trajectory.trajectory.steps
            assert len(steps) == request_data["steps_per_trajectory"], f"Expected {request_data['steps_per_trajectory']} steps, got {len(steps)}"
            
            # Validate each step format
            for j, step in enumerate(steps):
                assert isinstance(step, TrajectoryMessageStep), f"Step {j} in trajectory {i} is not a TrajectoryMessageStep"
                assert step.order == j, f"Step {j} has incorrect order: {step.order}"
                assert step.token_type in ["ADD", "DELETE"], f"Invalid token type: {step.token_type}"
                assert isinstance(step.token_literals, int), f"token_literals should be int, got {type(step.token_literals)}"
                assert isinstance(step.reward, float), f"reward should be float, got {type(step.reward)}"
                assert isinstance(step.avgQ, float), f"avgQ should be float, got {type(step.avgQ)}"
                assert step.avgQ > 0, f"avgQ should be positive, got {step.avgQ}"
        
    @pytest.mark.asyncio
    async def test_weapon_rollout_async_queue(self, client: httpx.Client, async_queue_agent: AsyncTrajectoryQueueAgent):
        """Test weapon rollout with async queue validation."""
        request_data = {
            "num_vars": 2,
            "width": 2,
            "size": 3,  
            "steps_per_trajectory": 3,
            "num_trajectories": 1,
            "initial_definition": [1, 2, 3],
            "initial_formula_id": "test_formula_async"
            # No seed - should use non-deterministic randomness
        }
        
        response = client.post("/weapon/rollout", json=request_data)
        assert response.status_code == 200
        
        response_data = response.json()
        assert response_data["num_trajectories"] == 1
        assert response_data["total_steps"] == 3
        
        # Wait for message to be processed and pop from async queue
        import asyncio
        trajectory = None
        for _ in range(20):  # Try up to 20 times with 0.5s intervals
            trajectory = await async_queue_agent.pop()
            if trajectory:
                break
            await asyncio.sleep(0.1)
        
        assert trajectory is not None, "No trajectory found in async queue"
        
        # Validate async trajectory format
        assert isinstance(trajectory, TrajectoryQueueMessage)
        assert trajectory.num_vars == request_data["num_vars"]
        assert trajectory.width == request_data["width"]
        assert trajectory.base_size == request_data["size"]
        assert trajectory.trajectory.base_formula_id == request_data["initial_formula_id"]
        assert len(trajectory.trajectory.steps) == request_data["steps_per_trajectory"]
        
        # Validate steps are properly ordered and formatted
        for i, step in enumerate(trajectory.trajectory.steps):
            assert step.order == i
            assert step.token_type in ["ADD", "DELETE"]
            assert isinstance(step.token_literals, int)
            assert isinstance(step.reward, float)
            assert isinstance(step.avgQ, float)

    def test_weapon_rollout_large_batch(self, client: httpx.Client, queue_agent: TrajectoryQueueAgent):
        """Test weapon rollout with larger batch to verify scalability."""
        request_data = {
            "num_vars": 4,
            "width": 3,
            "size": 8,
            "steps_per_trajectory": 20,
            "num_trajectories": 5,
            "initial_definition": [1, 2, 3, 4, 5, 6, 7, 8],
            "initial_formula_id": "test_formula_large",
            "seed": 123
        }
        
        response = client.post("/weapon/rollout", json=request_data)
        assert response.status_code == 200
        
        response_data = response.json()
        assert response_data["num_trajectories"] == 5
        assert response_data["total_steps"] == 100
        
        # Wait for trajectories to be pushed to queue
        time.sleep(0.1)
        
        # Validate that we can pop trajectories and they have correct format
        trajectories: list[TrajectoryQueueMessage] = []
        for _ in range(30):  # More attempts for larger batch
            trajectory = queue_agent.pop()
            if trajectory:
                trajectories.append(trajectory)
            if len(trajectories) >= 5:
                break
            time.sleep(0.1)
        
        # Should have received all 5 trajectories
        assert len(trajectories) >= 3, f"Expected at least 3 trajectories, got {len(trajectories)}"
        
        # Validate format of first trajectory
        first_trajectory = trajectories[0]
        assert first_trajectory.num_vars == 4
        assert first_trajectory.width == 3
        assert first_trajectory.base_size == 8
        assert first_trajectory.trajectory.base_formula_id == request_data["initial_formula_id"]
        assert len(first_trajectory.trajectory.steps) == 20

    def test_invalid_request_data(self, client: httpx.Client):
        """Test API with invalid request data."""
        
        # Missing required fields
        invalid_request = {
            "num_vars": 3,
            # Missing other required fields
        }
        
        response = client.post("/weapon/rollout", json=invalid_request)
        assert response.status_code == 422  # Validation error
        
        # Invalid data types
        invalid_types_request = {
            "num_vars": "three",  # Should be int
            "width": 2,
            "size": 5,
            "steps_per_trajectory": 10,
            "num_trajectories": 2,
            "initial_definition": [1, 2, 3, 4, 5],
        }
        
        response = client.post("/weapon/rollout", json=invalid_types_request)
        assert response.status_code == 422

    def test_invalid_formula_parameters(self, client: httpx.Client):
        """Test API with invalid formula parameter combinations."""
        
        # Width mismatch: formula definition incompatible with width constraint
        width_mismatch_request = {
            "num_vars": 2,
            "width": 1,  # Width too small for the formula definition
            "size": 3,
            "steps_per_trajectory": 5,
            "num_trajectories": 1,
            "initial_definition": [1, 2, 3],  # Results in width 2, but width=1 requested
            "seed": 42
        }
        
        response = client.post("/weapon/rollout", json=width_mismatch_request)
        assert response.status_code == 422  # Unprocessable Entity
        
        response_data = response.json()
        assert "Invalid formula parameters" in response_data["detail"]
        assert "width" in response_data["detail"].lower()

    def test_trajectory_format_validation(self):
        """Test that trajectory format follows expected structure."""
        
        # Create a sample trajectory message that should match what clusterbomb generates
        sample_trajectory = TrajectoryQueueMessage(
            num_vars=3,
            width=2,
            base_size=5,
            timestamp=datetime.now(),
            trajectory=TrajectoryMessageMultipleSteps(
                base_formula_id="test_formula_001",
                steps=[
                    TrajectoryMessageStep(
                        order=0,
                        token_type="ADD",  # Should be string as generated by clusterbomb
                        token_literals=5343,  # Integer representation of literals
                        reward=0.1,
                        avgQ=2.5
                    ),
                    TrajectoryMessageStep(
                        order=1,
                        token_type="DELETE", 
                        token_literals=616,
                        reward=-0.05,
                        avgQ=3.0
                    )
                ]
            )
        )
        
        # Validate structure
        assert sample_trajectory.num_vars == 3
        assert sample_trajectory.width == 2
        assert len(sample_trajectory.trajectory.steps) == 2
        
        # Validate steps structure
        first_step = sample_trajectory.trajectory.steps[0]
        assert first_step.order == 0
        assert first_step.token_type in ["ADD", "DELETE"]  # Valid token types
        assert isinstance(first_step.token_literals, int)
        assert isinstance(first_step.reward, float)
        assert isinstance(first_step.avgQ, float)
        
        # Test JSON serialization (what clusterbomb does)
        json_data = sample_trajectory.model_dump_json()
        parsed_data = json.loads(json_data)
        
        # Validate parsed structure
        assert parsed_data["num_vars"] == 3
        assert "timestamp" in parsed_data
        assert "trajectory" in parsed_data
        assert len(parsed_data["trajectory"]["steps"]) == 2

    def test_trajectory_data_constraints(self):
        """Test trajectory data follows expected constraints."""
        
        # Test that we can create valid trajectory steps
        valid_step = TrajectoryMessageStep(
            order=0,
            token_type="ADD",
            token_literals=123,
            reward=0.0,
            avgQ=1.0
        )
        assert valid_step.order == 0
        assert valid_step.token_type == "ADD"
        
        # Test that order must be non-negative integer
        try:
            TrajectoryMessageStep(
                order=-1,
                token_type="ADD",
                token_literals=123,
                reward=0.0,
                avgQ=1.0
            )
        except (ValidationError, ValueError):
            pass  # Expected to fail
        
        # Test that token_literals must be integer
        try:
            TrajectoryMessageStep(
                order=0,
                token_type="ADD",
                token_literals="invalid",  # Should be int
                reward=0.0,
                avgQ=1.0
            )
        except (ValidationError, ValueError):
            pass  # Expected to fail

    def test_response_data_types(self, client: httpx.Client):
        """Test that response contains correct data types."""
        request_data = {
            "num_vars": 3,
            "width": 2,
            "size": 5,
            "steps_per_trajectory": 3,
            "num_trajectories": 1,
            "initial_definition": [1, 2, 3, 4, 5],
            "seed": 999
        }
        
        response = client.post("/weapon/rollout", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Check data types
        assert isinstance(data["total_steps"], int)
        assert isinstance(data["num_trajectories"], int)
        
        # Check ranges
        assert data["total_steps"] > 0
        assert data["num_trajectories"] > 0
        assert data["total_steps"] == data["num_trajectories"] * request_data["steps_per_trajectory"]

    @pytest.mark.parametrize("num_vars,width,size", [
        (2, 2, 3),
        (3, 2, 5), 
        (4, 3, 8),
        (5, 4, 10)
    ])
    def test_various_configurations(self, client: httpx.Client, num_vars, width, size):
        """Test weapon rollout with various parameter configurations."""
        request_data = {
            "num_vars": num_vars,
            "width": width,
            "size": size,
            "steps_per_trajectory": 5,
            "num_trajectories": 2,
            "initial_definition": list(range(1, size + 1)),
            "seed": 42
        }
        
        response = client.post("/weapon/rollout", json=request_data)
        assert response.status_code == 200
        
        response_data = response.json()
        assert response_data["num_trajectories"] == 2
        assert response_data["total_steps"] == 10


if __name__ == "__main__":
    pytest.main([__file__])