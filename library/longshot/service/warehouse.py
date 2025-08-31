import httpx
from typing import Any, Dict, Optional
from datetime import datetime

class WarehouseClient:
    """
    The WarehouseClient class provides a high-level interface for interacting 
    with the Warehouse microservice, which manages the storage and retrieval of 
    trajectories.
    """

    def __init__(self, host: str, port: int, **config: Any):
        """
        Initializes the WarehouseClient with the given host and port.
        
        :param host: The warehouse server host address.
        :param port: The warehouse server port.
        :param config: Additional configuration parameters for the client.
        """
        base_url = f"http://{host}:{port}"
        self._client = httpx.Client(base_url=base_url)
        self._config = config

    def get_trajectory(self, traj_id: str) -> Dict[str, Any]:
        """
        Retrieves a trajectory by its ID.
        
        :param traj_id: The ID of the trajectory to retrieve.
        :return: A dictionary containing the trajectory information.
        :raises httpx.HTTPException: If the trajectory with the given ID does not exist in the warehouse.
        """
        response = self._client.get("/trajectory", params={"traj_id": traj_id})
        response.raise_for_status()
        return response.json()

    def post_trajectory(self, **body: Any) -> str:
        """
        Creates a new trajectory in the warehouse.
        
        :param body: The trajectory information to create.
        :return: The ID of the created trajectory.
        :raises httpx.HTTPException: If the request body is not in the correct format or missing required fields.
        """
        response = self._client.post("/trajectory", json=body)
        response.raise_for_status()
        return response.json()["traj_id"]

    def put_trajectory(self, **body: Any) -> None:
        """
        Updates an existing trajectory in the warehouse.
        
        :param body: The trajectory information to update.
        :raises httpx.HTTPException: If the trajectory with the given ID does not exist in the warehouse or if the request body is not in the correct format.
        """
        response = self._client.put("/trajectory", json=body)
        response.raise_for_status()

    def delete_trajectory(self, traj_id: str) -> None:
        """
        Deletes a trajectory from the warehouse by its ID.
        
        :param traj_id: The ID of the trajectory to delete.
        :raises httpx.HTTPException: If the trajectory with the given ID does not exist in the warehouse.
        """
        response = self._client.delete("/trajectory", params={"traj_id": traj_id})
        response.raise_for_status()
    
    def get_trajectory_dataset(self, 
                             num_vars: Optional[int] = None,
                             width: Optional[int] = None,
                             since: Optional[datetime] = None,
                             until: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Gets the trajectory dataset with optional filtering.
        
        :param num_vars: Filter trajectories by number of variables.
        :param width: Filter trajectories by width.
        :param since: Filter trajectories with timestamp after this date.
        :param until: Filter trajectories with timestamp before this date.
        :return: A dictionary containing the trajectory dataset.
        :raises httpx.HTTPException: If the request fails.
        """
        params = {}
        if num_vars is not None:
            params["num_vars"] = num_vars
        if width is not None:
            params["width"] = width
        if since is not None:
            params["since"] = since.isoformat() if hasattr(since, 'isoformat') else str(since)
        if until is not None:
            params["until"] = until.isoformat() if hasattr(until, 'isoformat') else str(until)
        
        response = self._client.get("/trajectory/dataset", params=params)
        response.raise_for_status()
        return response.json()
    
    def purge_trajectories(self) -> Dict[str, Any]:
        """
        Completely purges all trajectory data from the warehouse.
        
        WARNING: This operation cannot be undone.
        
        :return: A dictionary containing the purge result information.
        :raises httpx.HTTPException: If the purge operation fails.
        """
        response = self._client.delete("/trajectory/purge")
        response.raise_for_status()
        return response.json()

    def close(self) -> None:
        """
        Closes the connection to the Warehouse microservice.
        This method should be called to properly clean up resources when the WarehouseClient is no longer needed.
        """
        self._client.close()

    def __enter__(self) -> "WarehouseClient":
        """
        Enters the runtime context related to this object.
        This method is called when the object is used in a `with` statement.
        It returns the WarehouseClient instance itself."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exits the runtime context related to this object.
        This method is called when the `with` statement is exited, regardless of whether an exception occurred.
        """
        self.close()
        

class AsyncWarehouseClient:
    """Asynchronous client for interacting with the Longshot warehouse service.
    This class provides an async interface to the warehouse API, allowing for
    non-blocking I/O operations. It manages trajectory data storage and retrieval.
    It is recommended to use this class as an async context manager to ensure
    the underlying HTTP client is properly closed.
    
    Example:
        >>> async with AsyncWarehouseClient(host="localhost", port=8000) as client:
        ...     info = await client.get_trajectory("some_trajectory_id")
        
    Attributes:
        _client (httpx.AsyncClient): The asynchronous HTTP client for making requests.
        _config (Dict[str, Any]): Additional configuration options.
    """
    def __init__(self, host: str, port: int, **config: Any):
        """Initializes the AsyncWarehouseClient.
        
        Args:
            host (str): The hostname or IP address of the warehouse service.
            port (int): The port number of the warehouse service.
            **config (Any): Additional configuration parameters.
        """
        base_url = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(base_url=base_url, **config)
        self._config = config

    async def get_trajectory(self, traj_id: str) -> Dict[str, Any]:
        """Retrieves a trajectory by its ID.
        
        Args:
            traj_id (str): The unique identifier for the trajectory.
            
        Returns:
            Dict[str, Any]: A dictionary containing the trajectory data.
            
        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status code.
        """
        response = await self._client.get("/trajectory", params={"traj_id": traj_id})
        response.raise_for_status()
        return response.json()

    async def post_trajectory(self, **body: Any) -> str:
        """Creates a new trajectory entry.
        
        Args:
            **body (Any): Keyword arguments representing the trajectory data.
            
        Returns:
            str: The ID of the newly created trajectory.
            
        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status code.
        """
        response = await self._client.post("/trajectory", json=body)
        response.raise_for_status()
        return response.json()["traj_id"]

    async def put_trajectory(self, **body: Any) -> None:
        """Updates an existing trajectory.
        The body must contain the 'traj_id' of the trajectory to update.
        
        Args:
            **body (Any): Keyword arguments for the update, including the 'traj_id'.
            
        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status code.
        """
        response = await self._client.put("/trajectory", json=body)
        response.raise_for_status()

    async def delete_trajectory(self, traj_id: str) -> None:
        """Deletes a trajectory.
        
        Args:
            traj_id (str): The ID of the trajectory to delete.
            
        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status code.
        """
        response = await self._client.delete("/trajectory", params={"traj_id": traj_id})
        response.raise_for_status()
    
    async def get_trajectory_dataset(self, 
                                    num_vars: Optional[int] = None,
                                    width: Optional[int] = None,
                                    since: Optional[datetime] = None,
                                    until: Optional[datetime] = None) -> Dict[str, Any]:
        """Gets the trajectory dataset with optional filtering.
        
        Args:
            num_vars (Optional[int]): Filter trajectories by number of variables.
            width (Optional[int]): Filter trajectories by width.
            since (Optional[datetime]): Filter trajectories with timestamp after this date.
            until (Optional[datetime]): Filter trajectories with timestamp before this date.
            
        Returns:
            Dict[str, Any]: A dictionary containing the trajectory dataset.
            
        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status code.
        """
        params = {}
        if num_vars is not None:
            params["num_vars"] = num_vars
        if width is not None:
            params["width"] = width
        if since is not None:
            params["since"] = since.isoformat() if hasattr(since, 'isoformat') else str(since)
        if until is not None:
            params["until"] = until.isoformat() if hasattr(until, 'isoformat') else str(until)
        
        response = await self._client.get("/trajectory/dataset", params=params)
        response.raise_for_status()
        return response.json()
    
    async def purge_trajectories(self) -> Dict[str, Any]:
        """Completely purges all trajectory data from the warehouse.
        
        WARNING: This operation cannot be undone.
        
        Returns:
            Dict[str, Any]: A dictionary containing the purge result information.
            
        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status code.
        """
        response = await self._client.delete("/trajectory/purge")
        response.raise_for_status()
        return response.json()

    async def aclose(self) -> None:
        """Closes the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncWarehouseClient":
        """Enters the async context manager.
        
        Returns:
            AsyncWarehouseClient: The instance of the client.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exits the async context manager, ensuring the client is closed."""
        await self.aclose()