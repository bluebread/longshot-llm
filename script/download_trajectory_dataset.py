#!/usr/bin/env python3
"""
Download Trajectory Dataset from Warehouse

Downloads the complete trajectory dataset from the warehouse service and saves it to a specified file.

Usage:
    python download_trajectory_dataset.py --output trajectories.json
    python download_trajectory_dataset.py --output ../data/dataset.json --host localhost --port 8000
"""

import json
import argparse
import sys
from datetime import datetime

from longshot.agent import WarehouseAgent


def download_dataset(warehouse_host: str, warehouse_port: int, output_path: str) -> dict:
    """
    Download trajectory dataset from warehouse service
    
    Args:
        warehouse_host: Warehouse service host
        warehouse_port: Warehouse service port  
        output_path: Path to save the dataset
        
    Returns:
        The downloaded dataset dictionary
    """
    warehouse = WarehouseAgent(warehouse_host, warehouse_port)
    
    try:
        print(f"Connecting to warehouse at {warehouse_host}:{warehouse_port}...")
        print("Downloading trajectory dataset...")
        
        response = warehouse._client.get("/trajectory/dataset")
        response.raise_for_status()
        
        dataset = response.json()
        trajectories = dataset.get("trajectories", [])
        
        print(f"Downloaded {len(trajectories)} trajectories")
        
        # Add download metadata
        dataset["metadata"] = {
            "download_timestamp": datetime.now().isoformat(),
            "warehouse_host": warehouse_host,
            "warehouse_port": warehouse_port,
            "trajectory_count": len(trajectories)
        }
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print(f"Dataset saved to: {output_path}")
        
        # Print summary statistics
        if trajectories:
            total_steps = sum(len(t.get("steps", [])) for t in trajectories)
            avg_steps = total_steps / len(trajectories)
            print(f"\nDataset Statistics:")
            print(f"  Total trajectories: {len(trajectories)}")
            print(f"  Total steps: {total_steps}")
            print(f"  Average steps per trajectory: {avg_steps:.2f}")
            
            # Check for avgQ values
            has_avgq = any(
                any(step.get("cur_avgQ") is not None 
                    for step in t.get("steps", []))
                for t in trajectories
            )
            if has_avgq:
                print(f"  Contains avgQ values: Yes")
        
        return dataset
        
    except Exception as e:
        print(f"Error downloading dataset: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download trajectory dataset from warehouse service"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path for the dataset JSON"
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Warehouse service host (default: localhost)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Warehouse service port (default: 8000)"
    )
    
    args = parser.parse_args()
    
    # Download the dataset
    download_dataset(args.host, args.port, args.output)


if __name__ == "__main__":
    main()