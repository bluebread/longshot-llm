import httpx
from typing import Any, Dict, Optional, List
from datetime import datetime
from pathlib import Path
import zipfile
import io

class WarehouseClient:
    """
    The WarehouseClient class provides a high-level interface for interacting 
    with the Warehouse microservice, which manages the storage and retrieval of 
    trajectories.
    """

    def __init__(self, host: str, port: int, models_output_folder: Optional[str] = None, **config: Any):
        """
        Initializes the WarehouseClient with the given host and port.
        
        :param host: The warehouse server host address.
        :param port: The warehouse server port.
        :param models_output_folder: Optional folder path for downloading and storing models.
                                    Defaults to "./warehouse_models" if not specified.
        :param config: Additional configuration parameters for the client.
        """
        base_url = f"http://{host}:{port}"
        self._client = httpx.Client(base_url=base_url)
        self._config = config
        
        # Set default models folder if not provided (for backward compatibility)
        if models_output_folder is None:
            models_output_folder = "./warehouse_models"
        
        self._models_output_folder = Path(models_output_folder)
        # Create the output folder if it doesn't exist
        self._models_output_folder.mkdir(parents=True, exist_ok=True)

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

    # Parameter Server methods
    def get_models(self, num_vars: int, width: int, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Retrieves models matching the specified criteria.
        
        :param num_vars: Number of variables (required).
        :param width: Width parameter (required).
        :param tags: Optional list of tags to filter models.
        :return: A dictionary containing the list of models and count.
        :raises httpx.HTTPException: If the request fails.
        """
        params = {"num_vars": num_vars, "width": width}
        if tags:
            params["tags"] = tags
        
        response = self._client.get("/models", params=params)
        response.raise_for_status()
        return response.json()
    
    def get_latest_model(self, num_vars: int, width: int) -> Dict[str, Any]:
        """
        Retrieves the most recently uploaded model for the specified parameters.
        
        :param num_vars: Number of variables (required).
        :param width: Width parameter (required).
        :return: A dictionary containing the latest model metadata.
        :raises httpx.HTTPException: If no model is found or the request fails.
        """
        params = {"num_vars": num_vars, "width": width}
        response = self._client.get("/models/latest", params=params)
        response.raise_for_status()
        return response.json()
    
    def download_model(self, model_id: str, output_filename: Optional[str] = None) -> Path:
        """
        Downloads a model file and saves it to the models output folder.
        
        :param model_id: The GridFS model ID to download.
        :param output_filename: Optional custom filename for the downloaded file.
        :return: Path to the downloaded file.
        :raises httpx.HTTPException: If the model is not found or download fails.
        """
        response = self._client.get(f"/models/download/{model_id}")
        response.raise_for_status()
        
        # Extract filename from Content-Disposition header if not provided
        if output_filename is None:
            content_disp = response.headers.get("content-disposition", "")
            if "filename=" in content_disp:
                output_filename = content_disp.split("filename=")[-1].strip('"')
            else:
                output_filename = f"model_{model_id}.zip"
        
        output_path = self._models_output_folder / output_filename
        output_path.write_bytes(response.content)
        return output_path
    
    def upload_model(self, zip_path: str, num_vars: int, width: int, 
                    tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Uploads a model ZIP archive with metadata.
        
        :param zip_path: Path to the ZIP file to upload.
        :param num_vars: Number of variables (required).
        :param width: Width parameter (required).
        :param tags: Optional list of tags for the model.
        :return: A dictionary containing the upload response with model ID.
        :raises httpx.HTTPException: If the upload fails.
        :raises FileNotFoundError: If the ZIP file doesn't exist.
        :raises zipfile.BadZipFile: If the file is not a valid ZIP archive.
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")
        
        # Validate it's a ZIP file
        with zipfile.ZipFile(zip_path, 'r'):
            pass  # Just validate
        
        # Prepare the multipart form data
        with open(zip_path, 'rb') as f:
            files = {"file": (zip_path.name, f, "application/zip")}
            data = {
                "num_vars": str(num_vars),
                "width": str(width)
            }
            if tags:
                data["tags"] = ",".join(tags)
            
            response = self._client.post("/models/upload", files=files, data=data)
            response.raise_for_status()
            return response.json()
    
    def create_model_zip(self, files_dict: Dict[str, bytes], output_name: str) -> Path:
        """
        Helper method to create a ZIP file from a dictionary of files.
        
        :param files_dict: Dictionary mapping filenames to file contents (bytes).
        :param output_name: Name for the output ZIP file.
        :return: Path to the created ZIP file in the models output folder.
        """
        output_path = self._models_output_folder / output_name
        
        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename, content in files_dict.items():
                zf.writestr(filename, content)
        
        return output_path
    
    def purge_models(self) -> Dict[str, Any]:
        """
        Completely purges all models from GridFS storage.
        
        WARNING: This operation cannot be undone.
        
        :return: A dictionary containing the purge result information.
        :raises httpx.HTTPException: If the purge operation fails.
        """
        response = self._client.delete("/models/purge")
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
    def __init__(self, host: str, port: int, models_output_folder: Optional[str] = None, **config: Any):
        """Initializes the AsyncWarehouseClient.
        
        Args:
            host (str): The hostname or IP address of the warehouse service.
            port (int): The port number of the warehouse service.
            models_output_folder (Optional[str]): Optional folder path for downloading and storing models.
                                                 Defaults to "./warehouse_models" if not specified.
            **config (Any): Additional configuration parameters.
        """
        base_url = f"http://{host}:{port}"
        self._client = httpx.AsyncClient(base_url=base_url, **config)
        self._config = config
        
        # Set default models folder if not provided (for backward compatibility)
        if models_output_folder is None:
            models_output_folder = "./warehouse_models"
        
        self._models_output_folder = Path(models_output_folder)
        # Create the output folder if it doesn't exist
        self._models_output_folder.mkdir(parents=True, exist_ok=True)

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

    # Parameter Server async methods
    async def get_models(self, num_vars: int, width: int, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Retrieves models matching the specified criteria.
        
        Args:
            num_vars (int): Number of variables (required).
            width (int): Width parameter (required).
            tags (Optional[List[str]]): Optional list of tags to filter models.
            
        Returns:
            Dict[str, Any]: A dictionary containing the list of models and count.
            
        Raises:
            httpx.HTTPStatusError: If the API returns a non-2xx status code.
        """
        params = {"num_vars": num_vars, "width": width}
        if tags:
            params["tags"] = tags
        
        response = await self._client.get("/models", params=params)
        response.raise_for_status()
        return response.json()
    
    async def get_latest_model(self, num_vars: int, width: int) -> Dict[str, Any]:
        """Retrieves the most recently uploaded model for the specified parameters.
        
        Args:
            num_vars (int): Number of variables (required).
            width (int): Width parameter (required).
            
        Returns:
            Dict[str, Any]: A dictionary containing the latest model metadata.
            
        Raises:
            httpx.HTTPStatusError: If no model is found or the request fails.
        """
        params = {"num_vars": num_vars, "width": width}
        response = await self._client.get("/models/latest", params=params)
        response.raise_for_status()
        return response.json()
    
    async def download_model(self, model_id: str, output_filename: Optional[str] = None) -> Path:
        """Downloads a model file and saves it to the models output folder.
        
        Args:
            model_id (str): The GridFS model ID to download.
            output_filename (Optional[str]): Optional custom filename for the downloaded file.
            
        Returns:
            Path: Path to the downloaded file.
            
        Raises:
            httpx.HTTPStatusError: If the model is not found or download fails.
        """
        response = await self._client.get(f"/models/download/{model_id}")
        response.raise_for_status()
        
        # Extract filename from Content-Disposition header if not provided
        if output_filename is None:
            content_disp = response.headers.get("content-disposition", "")
            if "filename=" in content_disp:
                output_filename = content_disp.split("filename=")[-1].strip('"')
            else:
                output_filename = f"model_{model_id}.zip"
        
        output_path = self._models_output_folder / output_filename
        
        # Use async file writing
        import aiofiles
        async with aiofiles.open(output_path, 'wb') as f:
            await f.write(response.content)
        
        return output_path
    
    async def upload_model(self, zip_path: str, num_vars: int, width: int, 
                          tags: Optional[List[str]] = None) -> Dict[str, Any]:
        """Uploads a model ZIP archive with metadata.
        
        Args:
            zip_path (str): Path to the ZIP file to upload.
            num_vars (int): Number of variables (required).
            width (int): Width parameter (required).
            tags (Optional[List[str]]): Optional list of tags for the model.
            
        Returns:
            Dict[str, Any]: A dictionary containing the upload response with model ID.
            
        Raises:
            httpx.HTTPStatusError: If the upload fails.
            FileNotFoundError: If the ZIP file doesn't exist.
            zipfile.BadZipFile: If the file is not a valid ZIP archive.
        """
        zip_path = Path(zip_path)
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP file not found: {zip_path}")
        
        # Validate it's a ZIP file
        with zipfile.ZipFile(zip_path, 'r'):
            pass  # Just validate
        
        # Read file content for async upload
        with open(zip_path, 'rb') as f:
            file_content = f.read()
        
        # Prepare the multipart form data
        files = {"file": (zip_path.name, file_content, "application/zip")}
        data = {
            "num_vars": str(num_vars),
            "width": str(width)
        }
        if tags:
            data["tags"] = ",".join(tags)
        
        response = await self._client.post("/models/upload", files=files, data=data)
        response.raise_for_status()
        return response.json()
    
    async def create_model_zip(self, files_dict: Dict[str, bytes], output_name: str) -> Path:
        """Helper method to create a ZIP file from a dictionary of files.
        
        Args:
            files_dict (Dict[str, bytes]): Dictionary mapping filenames to file contents (bytes).
            output_name (str): Name for the output ZIP file.
            
        Returns:
            Path: Path to the created ZIP file in the models output folder.
        """
        output_path = self._models_output_folder / output_name
        
        # Create ZIP in memory first
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename, content in files_dict.items():
                zf.writestr(filename, content)
        
        # Write to file asynchronously
        import aiofiles
        async with aiofiles.open(output_path, 'wb') as f:
            await f.write(zip_buffer.getvalue())
        
        return output_path
    
    async def purge_models(self) -> Dict[str, Any]:
        """Completely purges all models from GridFS storage.
        
        WARNING: This operation cannot be undone.
        
        Returns:
            Dict[str, Any]: A dictionary containing the purge result information.
            
        Raises:
            httpx.HTTPStatusError: If the purge operation fails.
        """
        response = await self._client.delete("/models/purge")
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