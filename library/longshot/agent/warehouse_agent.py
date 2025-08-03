import httpx
from typing import Any, Dict, List, Optional

class WarehouseAgent:
    """
    The WarehouseAgent class provides a high-level interface for interacting 
    with the Warehouse microservice, which manages the storage and retrieval of 
    formulas, trajectories, and the evolution graph.
    """

    def __init__(self, host: str, port: int, **config: Any):
        """
        Initializes the WarehouseAgent.

        Args:
            host (str): The hostname or IP address of the Warehouse service.
            port (int): The port of the Warehouse service.
            config (Any): Additional configuration parameters.
        """
        base_url = f"http://{host}:{port}"
        self._client = httpx.Client(base_url=base_url)
        self._config = config
        
    def get_formula_info(self, formula_id: str) -> Dict[str, Any]:
        """
        Retrieves information about a formula by its ID.

        Args:
            formula_id (str): The ID of the formula to retrieve.

        Returns:
            A dictionary containing the formula information.

        Raises:
            httpx.HTTPStatusError: If the request fails (e.g., 404 Not Found).
        """
        response = self._client.get("/formula/info", params={"id": formula_id})
        response.raise_for_status()
        return response.json()

    def post_formula_info(self, **body: Any) -> str:
        """
        Adds a new formula entry to the formula table.

        Args:
            body (Any): The formula information to create.

        Returns:
            The ID of the created formula.

        Raises:
            httpx.HTTPStatusError: If the request fails (e.g., 422 Unprocessable Entity).
        """
        response = self._client.post("/formula/info", json=body)
        response.raise_for_status()
        return response.json()["id"]

    def put_formula_info(self, **body: Any) -> None:
        """
        Updates an existing formula entry.

        Args:
            body (Any): The formula information to update.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = self._client.put("/formula/info", json=body)
        response.raise_for_status()

    def delete_formula_info(self, formula_id: str) -> None:
        """
        Deletes a formula entry.

        Args:
            formula_id (str): The ID of the formula to delete.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = self._client.delete("/formula/info", params={"id": formula_id})
        response.raise_for_status()

    def get_likely_isomorphic(self, wl_hash: str) -> list[str]:
        """
        Retrieves IDs of likely isomorphic formulas.

        Args:
            wl_hash (str): The Weisfeiler-Lehman hash.

        Returns:
            A list of formula IDs.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = self._client.get("/formula/likely_isomorphic", params={"wl_hash": wl_hash})
        response.raise_for_status()
        return response.json().get("isomorphic_ids", [])

    def post_likely_isomorphic(self, wl_hash: str, formula_id: str) -> None:
        """
        Adds a likely isomorphic formula.

        Args:
            wl_hash (str): The hash of the formula.
            formula_id (str): The ID of the formula.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        body = {"wl_hash": wl_hash, "formula_id": formula_id}
        response = self._client.post("/formula/likely_isomorphic", json=body)
        response.raise_for_status()

    def delete_likely_isomorphic(self, wl_hash: str) -> None:
        """
        Deletes a likely isomorphic entry by its hash.

        Args:
            wl_hash (str): The hash of the formula to delete.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = self._client.delete("/formula/likely_isomorphic", params={"wl_hash": wl_hash})
        response.raise_for_status()

    def get_trajectory(self, traj_id: str) -> Dict[str, Any]:
        """
        Retrieves a trajectory by its ID.

        Args:
            traj_id (str): The ID of the trajectory to retrieve.

        Returns:
            A dictionary containing the trajectory information.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = self._client.get("/trajectory", params={"id": traj_id})
        response.raise_for_status()
        return response.json()

    def post_trajectory(self, **body: Any) -> str:
        """
        Adds a new trajectory.

        Args:
            body (Any): The trajectory information to create.

        Returns:
            The ID of the created trajectory.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = self._client.post("/trajectory", json=body)
        response.raise_for_status()
        return response.json()["id"]

    def put_trajectory(self, **body: Any) -> None:
        """
        Updates an existing trajectory.

        Args:
            body (Any): The trajectory information to update.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = self._client.put("/trajectory", json=body)
        response.raise_for_status()

    def delete_trajectory(self, traj_id: str) -> None:
        """
        Deletes a trajectory.

        Args:
            traj_id (str): The ID of the trajectory to delete.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        response = self._client.delete("/trajectory", params={"id": traj_id})
        response.raise_for_status()

    def get_formula_definition(self, formula_id: str | None) -> list[int]:
        """
        Retrieves the full definition of a formula by its ID.

        Args:
            formula_id (str): The ID of the formula.

        Returns:
            The definition of the formula. Return a empty list if the ID is None.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        if formula_id is None:
            return []
        
        response = self._client.get("/formula/definition", params={"id": formula_id})
        response.raise_for_status()
        return response.json()['definition']

    def close(self) -> None:
        """
        Closes the underlying HTTP client.
        """
        self._client.close()

    def __enter__(self) -> "WarehouseAgent":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()