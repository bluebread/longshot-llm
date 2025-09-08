#!/usr/bin/env python3
"""
Script to calculate and visualize avgQ distributions from trajectory data.

This script can load trajectory data from either:
1. A local JSON file (in the format of service/trainer/data/n3w2.json)
2. The Warehouse service via WarehouseClient

It calculates distributions of avgQ values at both trajectory and step levels,
then generates visualization plots saved as PNG files.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import httpx

# Add parent directory to path to import longshot library
sys.path.insert(0, str(Path(__file__).parent.parent / "library"))

from longshot.service.warehouse import WarehouseClient

# Constants
DEFAULT_ROUNDING_PRECISION = 6
DEFAULT_BAR_WIDTH = 0.01
DEFAULT_PLOT_DPI = 150
DEFAULT_FIGSIZE = (12, 10)
DEFAULT_TIMEOUT = 30.0
DEFAULT_WAREHOUSE_PORT = 8000


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Calculate and visualize avgQ distributions from trajectory data"
    )
    
    # Data source arguments
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--file",
        type=Path,
        help="Path to local JSON file containing trajectory data"
    )
    source_group.add_argument(
        "--warehouse",
        action="store_true",
        help="Download trajectories from warehouse service"
    )
    
    # Warehouse connection arguments
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Warehouse service host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_WAREHOUSE_PORT,
        help=f"Warehouse service port (default: {DEFAULT_WAREHOUSE_PORT})"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds for warehouse client requests (default: {DEFAULT_TIMEOUT})"
    )
    
    # Filter arguments
    parser.add_argument(
        "--num-vars",
        type=int,
        help="Filter trajectories by number of variables"
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Filter trajectories by width"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./"),
        help="Directory to save output plots (default: current directory)"
    )
    
    return parser.parse_args()


def load_trajectories_from_file(file_path: Path) -> Dict[str, Any]:
    """
    Load trajectory data from a local JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing trajectory data and metadata
    """
    print(f"Loading trajectories from file: {file_path}")
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data.get('trajectories', []))} trajectories from file")
    return data


class TimeoutWarehouseClient(WarehouseClient):
    """WarehouseClient with configurable timeout."""
    
    def __init__(self, host: str, port: int, timeout: float = DEFAULT_TIMEOUT, **config: Any):
        """
        Initialize WarehouseClient with timeout configuration.
        
        Args:
            host: Warehouse service host
            port: Warehouse service port
            timeout: Request timeout in seconds
            **config: Additional configuration
        """
        base_url = f"http://{host}:{port}"
        timeout_config = httpx.Timeout(timeout)
        self._client = httpx.Client(base_url=base_url, timeout=timeout_config)
        self._config = config


def load_trajectories_from_warehouse(
    host: str,
    port: int,
    timeout: float,
    num_vars: Optional[int] = None,
    width: Optional[int] = None
) -> Dict[str, Any]:
    """
    Download trajectory data from warehouse service.
    
    Args:
        host: Warehouse service host
        port: Warehouse service port
        timeout: Request timeout in seconds
        num_vars: Optional filter for number of variables
        width: Optional filter for width
        
    Returns:
        Dictionary containing trajectory data and metadata
        
    Raises:
        httpx.HTTPError: If the warehouse request fails
        ValueError: If the response data is invalid
    """
    print(f"Connecting to warehouse at {host}:{port} with timeout={timeout}s")
    
    try:
        with TimeoutWarehouseClient(host, port, timeout) as client:
            print(f"Downloading trajectories (num_vars={num_vars}, width={width})...")
            data = client.get_trajectory_dataset(num_vars=num_vars, width=width)
            
            # Convert to expected format if needed
            if "trajectories" not in data:
                # Assume the response is a list of trajectories
                data = {"trajectories": data}
            
            # Add metadata if not present
            if "metadata" not in data:
                data["metadata"] = {
                    "num_vars": num_vars,
                    "width": width,
                    "warehouse_host": host,
                    "warehouse_port": port
                }
            
            print(f"Downloaded {len(data.get('trajectories', []))} trajectories")
            return data
            
    except httpx.HTTPError as e:
        raise httpx.HTTPError(f"Failed to connect to warehouse: {e}")
    except Exception as e:
        raise ValueError(f"Invalid response from warehouse: {e}")


def validate_trajectory_data(data: Dict[str, Any]) -> None:
    """
    Validate trajectory data structure.
    
    Args:
        data: Trajectory data dictionary
        
    Raises:
        ValueError: If data structure is invalid
    """
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary")
    
    trajectories = data.get("trajectories", [])
    if not trajectories:
        raise ValueError("No trajectories found in data")
    
    for i, traj in enumerate(trajectories):
        if not isinstance(traj, dict) or "steps" not in traj:
            raise ValueError(f"Invalid trajectory structure at index {i}")
        
        steps = traj["steps"]
        if not isinstance(steps, list):
            raise ValueError(f"Steps must be a list in trajectory {i}")
        
        for j, step in enumerate(steps):
            if not isinstance(step, list) or len(step) < 3:
                raise ValueError(f"Invalid step format at trajectory {i}, step {j}")
            
            try:
                float(step[2])  # Validate avgQ is numeric
            except (ValueError, TypeError):
                raise ValueError(f"Invalid avgQ value at trajectory {i}, step {j}")


def extract_num_vars_width(data: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract num_vars and width from trajectory data.
    
    Args:
        data: Trajectory data dictionary
        
    Returns:
        Tuple of (num_vars, width)
    """
    # Try to get from metadata first
    metadata = data.get("metadata", {})
    num_vars = metadata.get("num_vars")
    width = metadata.get("width")
    
    # If not in metadata, try to infer from first trajectory
    if (num_vars is None or width is None) and data.get("trajectories"):
        # This would require analyzing the actual gate integers
        # For now, we'll just use the metadata values
        pass
    
    return num_vars, width


def calculate_avgq_distributions(data: Dict[str, Any]) -> Tuple[Counter, Counter]:
    """
    Calculate avgQ distributions at trajectory and step levels.
    
    Args:
        data: Trajectory data dictionary
        
    Returns:
        Tuple of (trajectory_distribution, step_distribution)
        - trajectory_distribution: Counter mapping avgQ -> number of trajectories
        - step_distribution: Counter mapping avgQ -> number of steps
    """
    trajectory_dist = Counter()
    step_dist = Counter()
    
    trajectories = data.get("trajectories", [])
    
    for traj in trajectories:
        # Get steps from trajectory
        steps = traj.get("steps", [])
        
        # Track max avgQ for this trajectory
        max_avgq = 0.0
        
        # Step format: [type, litint, avgQ]
        for _, _, avgq in steps:
            step_dist[avgq] += 1
            max_avgq = max(max_avgq, avgq)
        
        # For trajectory distribution, we count unique avgQ values per trajectory
        # or we might want the final avgQ value
        if steps:
            # Use the max avgQ value for the trajectory
            trajectory_dist[max_avgq] += 1
    
    print(f"\nDistribution Statistics:")
    print(f"  Unique avgQ values: {len(set(trajectory_dist.keys()) | set(step_dist.keys()))}")
    print(f"  Total trajectories: {sum(trajectory_dist.values())}")
    print(f"  Total steps: {sum(step_dist.values())}")
    
    return trajectory_dist, step_dist


def calculate_bar_width(values: list) -> float:
    """
    Calculate appropriate bar width based on data range.
    
    Args:
        values: List of numeric values
        
    Returns:
        Appropriate bar width
    """
    if len(values) < 2:
        return DEFAULT_BAR_WIDTH
    value_range = max(values) - min(values)
    # Use a width that's 1/10th of the average gap between values
    return max(0.001, value_range / (len(values) * 10))


def plot_distributions(
    trajectory_dist: Counter,
    step_dist: Counter,
    num_vars: Optional[int],
    width: Optional[int],
    output_dir: Path
) -> None:
    """
    Create and save distribution plots.
    
    Args:
        trajectory_dist: Counter of avgQ values at trajectory level
        step_dist: Counter of avgQ values at step level
        num_vars: Number of variables
        width: Width parameter
        output_dir: Directory to save plots
    """
    # Prepare data for plotting
    traj_avgq = sorted(trajectory_dist.keys())
    traj_counts = [trajectory_dist[q] for q in traj_avgq]
    
    step_avgq = sorted(step_dist.keys())
    step_counts = [step_dist[q] for q in step_avgq]
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=DEFAULT_FIGSIZE)
    
    # Title for the entire figure
    if num_vars is not None and width is not None:
        fig.suptitle(f"N = {num_vars}, W = {width}", fontsize=16, fontweight='bold')
    else:
        fig.suptitle("avgQ Distribution", fontsize=16, fontweight='bold')
    
    # Plot 1: Trajectory-level distribution
    bar_width = calculate_bar_width(traj_avgq)
    ax1.bar(traj_avgq, traj_counts, width=bar_width, edgecolor='black', alpha=0.7)
    ax1.set_xlabel("avgQ Value", fontsize=12)
    ax1.set_ylabel("Number of Trajectories", fontsize=12)
    ax1.set_title("Trajectory-level avgQ Distribution", fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    ax1.text(0.02, 0.98, 
             f"Total: {sum(traj_counts)} trajectories\n"
             f"Unique values: {len(traj_avgq)}\n"
             f"Min: {min(traj_avgq):.3f}\n"
             f"Max: {max(traj_avgq):.3f}",
             transform=ax1.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Step-level distribution
    bar_width = calculate_bar_width(step_avgq)
    ax2.bar(step_avgq, step_counts, width=bar_width, edgecolor='black', alpha=0.7, color='orange')
    ax2.set_xlabel("avgQ Value", fontsize=12)
    ax2.set_ylabel("Number of Steps", fontsize=12)
    ax2.set_title("Step-level avgQ Distribution", fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add statistics text
    ax2.text(0.02, 0.98,
             f"Total: {sum(step_counts)} steps\n"
             f"Unique values: {len(step_avgq)}\n"
             f"Min: {min(step_avgq):.3f}\n"
             f"Max: {max(step_avgq):.3f}",
             transform=ax2.transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if num_vars is not None and width is not None:
        filename = f"n{num_vars}w{width}_avgQ_distr.png"
    else:
        filename = "avgq_distr.png"
    
    output_path = output_dir / filename
    plt.savefig(output_path, dpi=DEFAULT_PLOT_DPI, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Clean up resources
    plt.close(fig)
    
    # Also create individual distribution plots
    create_individual_plots(trajectory_dist, step_dist, num_vars, width, output_dir)


def create_individual_plots(
    trajectory_dist: Counter,
    step_dist: Counter,
    num_vars: Optional[int],
    width: Optional[int],
    output_dir: Path
) -> None:
    """
    Create individual plots for trajectory and step distributions.
    
    Args:
        trajectory_dist: Counter of avgQ values at trajectory level
        step_dist: Counter of avgQ values at step level
        num_vars: Number of variables
        width: Width parameter
        output_dir: Directory to save plots
    """
    # Prepare data
    traj_avgq = sorted(trajectory_dist.keys())
    traj_counts = [trajectory_dist[q] for q in traj_avgq]
    
    step_avgq = sorted(step_dist.keys())
    step_counts = [step_dist[q] for q in step_avgq]
    
    # Base filename
    if num_vars is not None and width is not None:
        base_filename = f"n{num_vars}w{width}_avgQ_distr"
        title_suffix = f" (N = {num_vars}, W = {width})"
    else:
        base_filename = "avgq_distr"
        title_suffix = ""
    
    # Trajectory distribution plot
    plt.figure(figsize=(10, 6))
    bar_width = calculate_bar_width(traj_avgq)
    plt.bar(traj_avgq, traj_counts, width=bar_width, edgecolor='black', alpha=0.7, color='blue')
    plt.xlabel("avgQ Value", fontsize=12)
    plt.ylabel("Number of Trajectories", fontsize=12)
    plt.title(f"Trajectory-level avgQ Distribution{title_suffix}", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.text(0.02, 0.98,
             f"Total: {sum(traj_counts)} trajectories\n"
             f"Unique values: {len(traj_avgq)}\n"
             f"Mean: {np.average(traj_avgq, weights=traj_counts):.3f}\n"
             f"Min: {min(traj_avgq):.3f}\n"
             f"Max: {max(traj_avgq):.3f}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_filename}_trajectories.png", dpi=DEFAULT_PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    # Step distribution plot
    plt.figure(figsize=(10, 6))
    bar_width = calculate_bar_width(step_avgq)
    plt.bar(step_avgq, step_counts, width=bar_width, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel("avgQ Value", fontsize=12)
    plt.ylabel("Number of Steps", fontsize=12)
    plt.title(f"Step-level avgQ Distribution{title_suffix}", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.text(0.02, 0.98,
             f"Total: {sum(step_counts)} steps\n"
             f"Unique values: {len(step_avgq)}\n"
             f"Mean: {np.average(step_avgq, weights=step_counts):.3f}\n"
             f"Min: {min(step_avgq):.3f}\n"
             f"Max: {max(step_avgq):.3f}",
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_filename}_steps.png", dpi=DEFAULT_PLOT_DPI, bbox_inches='tight')
    plt.close()
    
    print(f"Additional plots saved:")
    print(f"  - {output_dir / f'{base_filename}_trajectories.png'}")
    print(f"  - {output_dir / f'{base_filename}_steps.png'}")


def main():
    """Main execution function."""
    args = parse_arguments()
    
    try:
        # Load trajectory data
        if args.file:
            data = load_trajectories_from_file(args.file)
        else:
            data = load_trajectories_from_warehouse(
                host=args.host,
                port=args.port,
                timeout=args.timeout,
                num_vars=args.num_vars,
                width=args.width
            )
        
        # Validate trajectory data
        validate_trajectory_data(data)
        
        # Extract num_vars and width
        num_vars, width = extract_num_vars_width(data)
        
        # Override with command line arguments if provided
        if args.num_vars is not None:
            num_vars = args.num_vars
        if args.width is not None:
            width = args.width
        
        # Calculate distributions
        trajectory_dist, step_dist = calculate_avgq_distributions(data)
        
        if not trajectory_dist and not step_dist:
            print("No trajectory data found to analyze!")
            return 1
        
        # Create and save plots
        plot_distributions(
            trajectory_dist,
            step_dist,
            num_vars,
            width,
            args.output_dir
        )
        
        print("\nAnalysis complete!")
        return 0
        
    except FileNotFoundError as e:
        print(f"File not found: {e}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"Invalid JSON data: {e}", file=sys.stderr)
        return 1
    except httpx.HTTPError as e:
        print(f"HTTP Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Data validation error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        import traceback
        print(f"Unexpected error: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())