import json
import os
from typing import Optional, Dict, Any
from datetime import datetime
from torch.utils.data import Dataset
from longshot.service import WarehouseClient


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        num_vars: int,
        width: int,
        warehouse_host: str = "localhost",
        warehouse_port: int = 8000,
        cache_dir: Optional[str] = None,
        force_download: bool = False,
        timeout: float = 30.0,
    ):
        """
        Initialize the TrajectoryDataset that downloads data from warehouse.
        
        Args:
            num_vars: Number of variables to filter trajectories
            width: Width to filter trajectories
            warehouse_host: Warehouse service host
            warehouse_port: Warehouse service port
            cache_dir: Directory to cache downloaded datasets (default: ./cache)
            force_download: Force re-download even if cache exists
            timeout: HTTP timeout in seconds for warehouse requests (default: 30.0)
            transform: Optional transform to apply to samples
        """
        self.num_vars = num_vars
        self.width = width
        self.warehouse_host = warehouse_host
        self.warehouse_port = warehouse_port
        self.cache_dir = cache_dir or "./cache"
        self.timeout = timeout
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Generate cache filename based on parameters
        self.cache_file = os.path.join(
            self.cache_dir,
            f"trajectories_n{num_vars}_w{width}.json"
        )
        
        # Load or download the dataset
        if not force_download and os.path.exists(self.cache_file):
            print(f"Loading cached dataset from {self.cache_file}")
            self._load_from_cache()
        else:
            print(f"Downloading dataset from warehouse (num_vars={num_vars}, width={width})")
            self._download_from_warehouse()
            self._save_to_cache()
    
    def _download_from_warehouse(self) -> None:
        """Download trajectory dataset from warehouse service."""
        try:
            with WarehouseClient(self.warehouse_host, self.warehouse_port) as client:
                # Build query parameters
                params = {
                    "num_vars": self.num_vars,
                    "width": self.width
                }
                
                # Use the internal client directly with timeout
                response = client._client.get(
                    "/trajectory/dataset", 
                    params=params,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Get the JSON response
                dataset_response: dict = response.json()
                
                # Process trajectories to extract useful features
                trajectories = dataset_response.get("trajectories", [])
                
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset from warehouse: {e}")
    
        for traj in trajectories:
            assert "num_vars" in traj, "Trajectory missing 'num_vars'"
            assert "width" in traj, "Trajectory missing 'width'"
            assert traj["num_vars"] == self.num_vars, "Mismatch in num_vars"
            assert traj["width"] == self.width, "Mismatch in width"
            
        self.data = [traj["steps"] for traj in trajectories]
        
        # Store metadata
        self.metadata = {
            "download_timestamp": datetime.now().isoformat(),
            "warehouse_host": self.warehouse_host,
            "warehouse_port": self.warehouse_port,
            "num_vars": self.num_vars,
            "width": self.width,
            "trajectory_count": len(trajectories)
        }
        
        print(f"Downloaded {len(trajectories)} trajectories")
        
    def _save_to_cache(self) -> None:
        """Save the dataset to cache file."""
        cache_data = {
            "metadata": self.metadata,
            "data": self.data
        }
        
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"Dataset cached to {self.cache_file}")
    
    def _load_from_cache(self) -> None:
        """Load dataset from cache file."""
        with open(self.cache_file, "r", encoding="utf-8") as f:
            cache_data = json.load(f)
        
        self.metadata = cache_data.get("metadata", {})
        self.data = cache_data.get("data", [])
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            The sample at the given index
        """
        func = lambda t, l: l if t == 0 else -l
        
        return {
            "input_ids": [func(t, l) for t, l, _ in self.data[idx]],
            "attention_mask": [1] * len(self.data[idx]),
            "labels": [q for _, _, q in self.data[idx]],
        }
    
if __name__ == "__main__":
    # Example usage
    dataset = TrajectoryDataset(num_vars=3, width=2)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"First sample: {sample}")
    
    from torch.utils.data import random_split
    
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    print(f"Train size: {len(train_dataset)}, Eval size: {len(eval_dataset)}")