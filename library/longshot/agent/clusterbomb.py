"""
ClusterbombAgent provides a client interface for the Clusterbomb microservice.

This agent handles weapon rollout operations, which generate trajectories
by simulating formula transformations in the RL environment.
"""

import httpx
from typing import Any, Optional
from ..models import WeaponRolloutRequest, WeaponRolloutResponse


class ClusterbombAgent:
    """
    The ClusterbombAgent class provides a high-level interface for interacting
    with the Clusterbomb microservice, which handles weapon rollout operations
    for generating trajectories in the longshot system.
    """

    def __init__(self, host: str, port: int, **config: Any):
        """
        Initializes the ClusterbombAgent with the given host and port.
        
        Args:
            host: The Clusterbomb service host address.
            port: The Clusterbomb service port (default: 8060).
            config: Additional configuration parameters for the agent.
        """
        base_url = f"http://{host}:{port}"
        self._client = httpx.Client(base_url=base_url, timeout=60.0)
        self._config = config

    def health_check(self) -> dict:
        """
        Checks the health status of the Clusterbomb service.
        
        Returns:
            dict: Health status information including service name and status.
            
        Raises:
            httpx.HTTPException: If the service is unreachable or unhealthy.
        """
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    def weapon_rollout(
        self,
        num_vars: int,
        width: int,
        size: int,
        steps_per_trajectory: int,
        num_trajectories: int,
        initial_definition: list[int],
        initial_node_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> WeaponRolloutResponse:
        """
        Executes a weapon rollout operation to generate trajectories.
        
        This method collects trajectories from the environment by simulating
        formula transformations. The trajectories are then processed and stored
        in the warehouse.
        
        Args:
            num_vars: Number of variables in the formula.
            width: Width constraint for the formula.
            size: Size (number of gates) in the initial formula.
            steps_per_trajectory: Number of transformation steps per trajectory.
            num_trajectories: Number of trajectories to generate.
            initial_definition: Initial formula definition as a list of gate integers.
            initial_node_id: Optional ID for the initial node in the evolution graph.
            seed: Optional random seed for deterministic trajectory generation.
            
        Returns:
            WeaponRolloutResponse: Response containing the total steps executed
                                   and number of trajectories generated.
            
        Raises:
            httpx.HTTPException: If the request fails or parameters are invalid.
        """
        request = WeaponRolloutRequest(
            num_vars=num_vars,
            width=width,
            size=size,
            steps_per_trajectory=steps_per_trajectory,
            num_trajectories=num_trajectories,
            initial_definition=initial_definition,
            initial_node_id=initial_node_id,
            seed=seed,
        )
        
        response = self._client.post("/weapon/rollout", json=request.model_dump(exclude_none=True))
        response.raise_for_status()
        
        return WeaponRolloutResponse(**response.json())

    def close(self) -> None:
        """
        Closes the HTTP client connection.
        """
        self._client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures client is closed."""
        self.close()


class AsyncClusterbombAgent:
    """
    Asynchronous version of ClusterbombAgent for use in async contexts.
    """

    def __init__(self, host: str, port: int, **config: Any):
        """
        Initializes the AsyncClusterbombAgent with the given host and port.
        
        Args:
            host: The Clusterbomb service host address.
            port: The Clusterbomb service port (default: 8060).
            config: Additional configuration parameters for the agent.
        """
        base_url = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
        self._config = config

    async def health_check(self) -> dict:
        """
        Checks the health status of the Clusterbomb service.
        
        Returns:
            dict: Health status information including service name and status.
            
        Raises:
            httpx.HTTPException: If the service is unreachable or unhealthy.
        """
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    async def weapon_rollout(
        self,
        num_vars: int,
        width: int,
        size: int,
        steps_per_trajectory: int,
        num_trajectories: int,
        initial_definition: list[int],
        initial_node_id: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> WeaponRolloutResponse:
        """
        Executes a weapon rollout operation to generate trajectories.
        
        This method collects trajectories from the environment by simulating
        formula transformations. The trajectories are then processed and stored
        in the warehouse.
        
        Args:
            num_vars: Number of variables in the formula.
            width: Width constraint for the formula.
            size: Size (number of gates) in the initial formula.
            steps_per_trajectory: Number of transformation steps per trajectory.
            num_trajectories: Number of trajectories to generate.
            initial_definition: Initial formula definition as a list of gate integers.
            initial_node_id: Optional ID for the initial node in the evolution graph.
            seed: Optional random seed for deterministic trajectory generation.
            
        Returns:
            WeaponRolloutResponse: Response containing the total steps executed
                                   and number of trajectories generated.
            
        Raises:
            httpx.HTTPException: If the request fails or parameters are invalid.
        """
        request = WeaponRolloutRequest(
            num_vars=num_vars,
            width=width,
            size=size,
            steps_per_trajectory=steps_per_trajectory,
            num_trajectories=num_trajectories,
            initial_definition=initial_definition,
            initial_node_id=initial_node_id,
            seed=seed,
        )
        
        response = await self._client.post("/weapon/rollout", json=request.model_dump(exclude_none=True))
        response.raise_for_status()
        
        return WeaponRolloutResponse(**response.json())

    async def close(self) -> None:
        """
        Closes the HTTP client connection.
        """
        await self._client.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - ensures client is closed."""
        await self.close()