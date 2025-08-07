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
        Initializes the WarehouseAgent with the given host and port.
        
        :param host: The RabbitMQ server host address.
        :param port: The RabbitMQ server port (default: 5672).
        :param config: Additional configuration parameters for the agent.
        """
        base_url = f"http://{host}:{port}"
        self._client = httpx.Client(base_url=base_url)
        self._config = config
        
    def get_formula_info(self, formula_id: str) -> Dict[str, Any]:
        """
        Retrieves information about a formula by its ID.
        
        :param formula_id: The ID of the formula to retrieve information for.
        :return: A dictionary containing the formula information.
        :raises httpx.HTTPException: If the formula with the given ID does not exist in the warehouse.
        """
        response = self._client.get("/formula/info", params={"id": formula_id})
        response.raise_for_status()
        return response.json()

    def post_formula_info(self, **body: Any) -> str:
        """
        Creates or updates formula information in the warehouse.
        
        :param body: The formula information to create or update.
        :return: The ID of the created or updated formula.
        :raises httpx.HTTPException: If the request body is not in the correct format or missing required fields.
        """
        response = self._client.post("/formula/info", json=body)
        response.raise_for_status()
        return response.json()["id"]

    def put_formula_info(self, **body: Any) -> None:
        """
        Updates formula information in the warehouse.
        
        :param body: The formula information to update.
        :raises httpx.HTTPException: If the formula with the given ID does not exist in the warehouse or if the request body is not in the correct format.
        """
        response = self._client.put("/formula/info", json=body)
        response.raise_for_status()

    def delete_formula_info(self, formula_id: str) -> None:
        """
        Deletes a formula from the warehouse by its ID.
        
        :param formula_id: The ID of the formula to delete.
        :raises httpx.HTTPException: If the formula with the given ID does not exist in the warehouse.
        """
        response = self._client.delete("/formula/info", params={"id": formula_id})
        response.raise_for_status()

    def get_likely_isomorphic(self, wl_hash: str) -> list[str]:
        """
        Retrieves a list of formula IDs that are likely isomorphic to the given formula hash.
        
        :param wl_hash: The hash of the formula to retrieve likely isomorphic formulas for.
        :return: A list of formula IDs that are likely isomorphic to the given formula.
        :raises httpx.HTTPException: If the formula with the given hash does not exist in the warehouse.
        """
        response = self._client.get("/formula/likely_isomorphic", params={"wl_hash": wl_hash})
        response.raise_for_status()
        return response.json().get("isomorphic_ids", [])

    def post_likely_isomorphic(self, wl_hash: str, formula_id: str) -> None:
        """
        Adds a formula to the list of likely isomorphic formulas for a given formula hash.
        
        :param wl_hash: The hash of the formula.
        :param formula_id: The ID of the formula to add to the list of likely isomorphic formulas.
        :raises httpx.HTTPException: If the formula with the given hash does not exist in the warehouse or if the request body is not in the correct format.
        """
        body = {"wl_hash": wl_hash, "formula_id": formula_id}
        response = self._client.post("/formula/likely_isomorphic", json=body)
        response.raise_for_status()

    def delete_likely_isomorphic(self, wl_hash: str) -> None:
        """
        Deletes the list of likely isomorphic formulas for a given formula hash.
        
        :param wl_hash: The hash of the formula to delete likely isomorphic formulas for.
        :raises httpx.HTTPException: If the formula with the given hash does not exist in the warehouse.
        """
        response = self._client.delete("/formula/likely_isomorphic", params={"wl_hash": wl_hash})
        response.raise_for_status()

    def get_trajectory(self, traj_id: str) -> Dict[str, Any]:
        """
        Retrieves a trajectory by its ID.
        
        :param traj_id: The ID of the trajectory to retrieve.
        :return: A dictionary containing the trajectory information.
        :raises httpx.HTTPException: If the trajectory with the given ID does not exist in the warehouse.
        """
        response = self._client.get("/trajectory", params={"id": traj_id})
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
        return response.json()["id"]

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
        response = self._client.delete("/trajectory", params={"id": traj_id})
        response.raise_for_status()
    
    def get_evolution_graph_node(self, formula_id: str) -> Dict[str, Any]:
        """
        Retrieves the evolution graph node for a given formula ID.

        :param formula_id: The ID of the formula to retrieve the evolution graph node for.
        :return: A dictionary containing the evolution graph node information.
        """
        response = self._client.get("/evolution_graph/node", params={"id": formula_id})
        response.raise_for_status()
        return response.json()
    
    def post_evolution_graph_node(self, **body: Any) -> str:
        """
        Creates a new evolution graph node in the warehouse.

        :param body: The evolution graph node information to create.
        :return: The ID of the created evolution graph node.
        :raises httpx.HTTPException: If the request body is not in the correct format or missing required fields.
        """
        response = self._client.post("/evolution_graph/node", json=body)
        response.raise_for_status()
        return response.json()["formula_id"]
    
    def put_evolution_graph_node(self, **body: Any) -> None:
        """
        Updates an existing evolution graph node in the warehouse.

        :param body: The evolution graph node information to update.
        :raises httpx.HTTPException: If the evolution graph node with the given ID does not exist in the warehouse or if the request body is not in the correct format.
        """
        response = self._client.put("/evolution_graph/node", json=body)
        response.raise_for_status()
        
    def delete_evolution_graph_node(self, formula_id: str) -> None:
        """
        Deletes an evolution graph node from the warehouse by its formula ID.

        :param formula_id: The ID of the evolution graph node to delete.
        :raises httpx.HTTPException: If the evolution graph node with the given ID does not exist in the warehouse.
        """
        response = self._client.delete("/evolution_graph/node", params={"formula_id": formula_id})
        response.raise_for_status()
    
    def get_formula_definition(self, formula_id: str | None) -> list[int]:
        """
        Retrieves the definition of a formula by its ID.
        
        :param formula_id: The ID of the formula to retrieve the definition for.
        :return: A list of integers representing the formula's definition.
        :raises httpx.HTTPException: If the formula with the given ID does not exist in the warehouse.
        """
        if formula_id is None:
            return []
        
        response = self._client.get("/formula/definition", params={"id": formula_id})
        response.raise_for_status()
        return response.json()['definition']

    def post_evolution_graph_path(self, path: List[str]) -> None:
        """
        Posts an evolution path to the warehouse.
        
        :param evo_path: A list of formula IDs representing the evolution path.
        :raises httpx.HTTPException: If the request body is not in the correct format or missing required fields.
        """
        body = {"path": path}
        response = self._client.post("/evolution_graph/path", json=body)
        response.raise_for_status()

    def download_nodes(self, num_vars: int, width: int, size_constraint: int | None = None) -> List[Dict[str, Any]]:
        """
        Downloads nodes from the warehouse based on the given constraints.
        
        :param num_vars: The number of variables in the formulas.
        :param width: The width of the formulas.
        :param size_constraint: An optional size constraint for the formulas.
        :return: A list of dictionaries containing the downloaded nodes.
        :raises httpx.HTTPException: If the request body is not in the correct format or missing required fields.
        """
        params = {"num_vars": num_vars, "width": width}
        if size_constraint is not None:
            params["size_constraint"] = size_constraint
        
        response = self._client.get("/evolution_graph/download_nodes", params=params)
        response.raise_for_status()
        return response.json()["nodes"]

    def download_hypernodes(self, num_vars: int, width: int, size_constraint: int | None = None) -> List[Dict[str, Any]]:
        """
        Downloads hypernodes from the warehouse.
        
        :param hypernodes: A list of hypernode IDs to download.
        :return: A dictionary containing the downloaded hypernodes.
        :raises httpx.HTTPException: If the request body is not in the correct format or missing required fields.
        """
        params = {"num_vars": num_vars, "width": width}
        if size_constraint is not None:
            params["size_constraint"] = size_constraint
        response = self._client.post("/evolution_graph/download_hypernodes", params=params)
        response.raise_for_status()
        return response.json()["hypernodes"]

    def close(self) -> None:
        """
        Closes the connection to the Warehouse microservice.
        This method should be called to properly clean up resources when the WarehouseAgent is no longer needed.
        """
        self._client.close()

    def __enter__(self) -> "WarehouseAgent":
        """
        Enters the runtime context related to this object.
        This method is called when the object is used in a `with` statement.
        It returns the WarehouseAgent instance itself."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Exits the runtime context related to this object.
        This method is called when the `with` statement is exited, regardless of whether an exception occurred.
        """
        self.close()