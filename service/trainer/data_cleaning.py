#!/usr/bin/env python3
"""
Trajectory Data Cleaning Script

This script downloads trajectories from the warehouse service and performs
data cleaning operations including filtering by avgQ threshold and truncation.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

import httpx

# Set up the path to import from the library
sys.path.insert(0, '/root/longshot-llm/library')
from longshot.service.warehouse import WarehouseClient


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def download_trajectories(
    host: str, 
    port: int, 
    num_vars: int, 
    width: int,
    timeout: float,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Download trajectories from warehouse service.
    
    Args:
        host: Warehouse service host
        port: Warehouse service port
        num_vars: Filter by number of variables
        width: Filter by width parameter
        timeout: Request timeout in seconds (note: not used in WarehouseClient)
        logger: Logger instance
        
    Returns:
        List of trajectory dictionaries
        
    Raises:
        httpx.HTTPStatusError: If warehouse service request fails
    """
    logger.info(f"Connecting to warehouse at {host}:{port}")
    logger.info(f"Downloading trajectories with num_vars={num_vars}, width={width}")
    
    try:
        with WarehouseClient(host, port) as client:
            dataset = client.get_trajectory_dataset(num_vars=num_vars, width=width)
            trajectories = dataset.get("trajectories", [])
            logger.info(f"Downloaded {len(trajectories)} trajectories")
            
            # Validate trajectory format
            for i, traj in enumerate(trajectories):
                if not validate_trajectory_format(traj, logger):
                    logger.warning(f"Trajectory {i} has invalid format, skipping")
                    continue
                    
            return trajectories
            
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error downloading trajectories: {e.response.status_code} {e.response.text}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        raise


def get_max_avgq(steps: List[Tuple[int, int, float]]) -> float:
    """
    Get the maximum avgQ value from trajectory steps.
    
    Args:
        steps: List of trajectory steps (token_type, token_literals, cur_avgQ)
        
    Returns:
        Maximum avgQ value, or 0.0 if no steps
    """
    if not steps:
        return 0.0
    return max(step[2] for step in steps)


def validate_trajectory_format(traj: Dict[str, Any], logger: logging.Logger) -> bool:
    """
    Validate trajectory format to ensure it has the expected structure.
    
    Args:
        traj: Trajectory dictionary to validate
        logger: Logger instance
        
    Returns:
        True if trajectory format is valid, False otherwise
    """
    required_fields = ["traj_id", "steps"]
    for field in required_fields:
        if field not in traj:
            logger.warning(f"Trajectory missing required field: {field}")
            return False
    
    steps = traj["steps"]
    if not isinstance(steps, list):
        logger.warning("Trajectory steps is not a list")
        return False
    
    for i, step in enumerate(steps):
        if not isinstance(step, (list, tuple)) or len(step) != 3:
            logger.warning(f"Step {i} is not a 3-element tuple/list")
            return False
        
        try:
            token_type, token_literals, cur_avgq = step
            if not isinstance(token_type, int):
                logger.warning(f"Step {i} token_type is not int")
                return False
            if not isinstance(token_literals, int):
                logger.warning(f"Step {i} token_literals is not int")
                return False
            if not isinstance(cur_avgq, (int, float)):
                logger.warning(f"Step {i} cur_avgQ is not number")
                return False
        except (ValueError, TypeError) as e:
            logger.warning(f"Step {i} has invalid format: {e}")
            return False
    
    return True


def filter_trajectories_by_threshold(
    trajectories: List[Dict[str, Any]], 
    threshold: float,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Filter out trajectories with maximum avgQ below threshold.
    
    Args:
        trajectories: List of trajectory dictionaries
        threshold: Minimum threshold for maximum avgQ
        logger: Logger instance
        
    Returns:
        Filtered list of trajectories
    """
    logger.info(f"Filtering trajectories with max avgQ >= {threshold}")
    
    filtered = []
    for traj in trajectories:
        steps = traj.get("steps", [])
        max_avgq = get_max_avgq(steps)
        
        if max_avgq >= threshold:
            filtered.append(traj)
        else:
            logger.debug(f"Filtered out trajectory {traj.get('traj_id', 'unknown')} "
                        f"with max avgQ {max_avgq:.4f}")
    
    logger.info(f"Kept {len(filtered)} trajectories after filtering "
               f"(removed {len(trajectories) - len(filtered)})")
    return filtered


def truncate_trajectory_to_max_avgq(
    steps: List[Tuple[int, int, float]]
) -> List[Tuple[int, int, float]]:
    """
    Truncate trajectory from beginning to the last step with highest avgQ.
    
    If there are multiple steps with the same highest avgQ, truncate to the
    last one. If the trajectory has only one step, return it unchanged.
    
    Args:
        steps: List of trajectory steps (token_type, token_literals, cur_avgQ)
        
    Returns:
        Truncated list of steps
    """
    if not steps:
        return steps
    
    # Find the maximum avgQ value
    max_avgq = max(step[2] for step in steps)
    
    # Find the last step with maximum avgQ (using small epsilon for float comparison)
    last_max_index = -1
    for i, step in enumerate(steps):
        if abs(step[2] - max_avgq) < 1e-10:
            last_max_index = i
    
    # Truncate from beginning to the last step with max avgQ (inclusive)
    return steps[:last_max_index + 1]


def process_trajectories(
    trajectories: List[Dict[str, Any]], 
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Process trajectories by truncating them to the last highest avgQ step.
    
    Args:
        trajectories: List of trajectory dictionaries
        logger: Logger instance
        
    Returns:
        List of processed trajectories
    """
    logger.info("Processing trajectories - truncating to last highest avgQ step")
    
    processed = []
    for traj in trajectories:
        original_steps = traj.get("steps", [])
        truncated_steps = truncate_trajectory_to_max_avgq(original_steps)
        
        # Create processed trajectory with truncated steps
        processed_traj = {
            "traj_id": traj.get("traj_id"),
            "num_vars": traj.get("num_vars"),
            "width": traj.get("width"),
            "steps": truncated_steps
        }
        
        processed.append(processed_traj)
        
        logger.debug(f"Trajectory {traj.get('traj_id', 'unknown')}: "
                    f"{len(original_steps)} -> {len(truncated_steps)} steps")
    
    logger.info(f"Processed {len(processed)} trajectories")
    return processed


def create_output_dataset(
    trajectories: List[Dict[str, Any]],
    num_vars: int,
    width: int,
    threshold: float,
    host: str,
    port: int,
    original_count: int,
    filtered_count: int
) -> Dict[str, Any]:
    """
    Create the final output dataset with metadata.
    
    Args:
        trajectories: Processed trajectories
        num_vars: Number of variables parameter
        width: Width parameter
        threshold: avgQ threshold used
        host: Warehouse host
        port: Warehouse port
        original_count: Original number of trajectories downloaded
        filtered_count: Number of trajectories that passed filtering
        
    Returns:
        Complete dataset dictionary
    """
    return {
        "metadata": {
            "num_vars": num_vars,
            "width": width,
            "threshold": threshold,
            "warehouse_host": host,
            "warehouse_port": port,
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "downloaded_count": original_count,
            "filtered_count": filtered_count,
            "removed_count": original_count - filtered_count,
            "processed_count": len(trajectories)
        },
        "trajectories": trajectories
    }


def save_dataset_to_file(dataset: Dict[str, Any], output_file: str, logger: logging.Logger):
    """
    Save the dataset to a JSON file.
    
    Args:
        dataset: Complete dataset dictionary
        output_file: Output file path
        logger: Logger instance
        
    Raises:
        IOError: If file write fails
    """
    logger.info(f"Saving dataset to {output_file}")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        logger.info(f"Dataset saved successfully to {output_file}")
        logger.info(f"Total trajectories in output: {len(dataset['trajectories'])}")
        
    except IOError as e:
        logger.error(f"Failed to save dataset to {output_file}: {e}")
        raise


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed arguments
        
    Raises:
        ValueError: If any argument is invalid
    """
    if args.num_vars <= 0:
        raise ValueError(f"num_vars must be positive, got {args.num_vars}")
    
    if args.width <= 0:
        raise ValueError(f"width must be positive, got {args.width}")
    
    if args.port <= 0 or args.port > 65535:
        raise ValueError(f"port must be between 1 and 65535, got {args.port}")
    
    if args.timeout <= 0:
        raise ValueError(f"timeout must be positive, got {args.timeout}")
    
    # Check if output directory exists and is writable
    import os
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        raise ValueError(f"Output directory does not exist: {output_dir}")
    
    if output_dir and not os.access(output_dir, os.W_OK):
        raise ValueError(f"Output directory is not writable: {output_dir}")


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and clean trajectory dataset from warehouse service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --num-vars 4 --width 3 --threshold 0.5 --output cleaned_data.json
  %(prog)s --num-vars 3 --width 2 --threshold 0.8 --host remote.server --port 8080 --output data.json --verbose
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--num-vars", 
        type=int, 
        required=True,
        help="Number of variables to filter trajectories by (must be positive)"
    )
    
    parser.add_argument(
        "--width", 
        type=int, 
        required=True,
        help="Width parameter to filter trajectories by (must be positive)"
    )
    
    parser.add_argument(
        "--threshold", 
        type=float, 
        required=True,
        help="Minimum threshold for maximum avgQ in trajectories"
    )
    
    parser.add_argument(
        "--output", 
        type=str, 
        required=True,
        help="Output JSON file path for cleaned dataset"
    )
    
    # Optional arguments
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
    
    parser.add_argument(
        "--timeout", 
        type=float, 
        default=30.0,
        help="Request timeout in seconds (default: 30.0)"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    validate_arguments(args)
    return args


def main():
    """Main script entry point."""
    args = parse_arguments()
    logger = setup_logging(args.verbose)
    
    try:
        logger.info("Starting trajectory data cleaning script")
        logger.info(f"Parameters: num_vars={args.num_vars}, width={args.width}, "
                   f"threshold={args.threshold}")
        
        # Step 1: Download trajectories from warehouse
        trajectories = download_trajectories(
            args.host, args.port, args.num_vars, args.width, args.timeout, logger
        )
        original_count = len(trajectories)
        
        if not trajectories:
            logger.warning("No trajectories found matching the criteria")
            # Create empty dataset
            dataset = create_output_dataset(
                [], args.num_vars, args.width, args.threshold,
                args.host, args.port, 0, 0
            )
            save_dataset_to_file(dataset, args.output, logger)
            return
        
        # Step 2: Filter trajectories by avgQ threshold
        filtered_trajectories = filter_trajectories_by_threshold(
            trajectories, args.threshold, logger
        )
        
        if not filtered_trajectories:
            logger.warning(f"No trajectories have max avgQ >= {args.threshold}")
            # Create empty dataset
            dataset = create_output_dataset(
                [], args.num_vars, args.width, args.threshold,
                args.host, args.port, original_count, len(filtered_trajectories)
            )
            save_dataset_to_file(dataset, args.output, logger)
            return
        
        # Step 3: Truncate trajectories to last highest avgQ step
        processed_trajectories = process_trajectories(filtered_trajectories, logger)
        
        # Step 4: Create final output dataset
        dataset = create_output_dataset(
            processed_trajectories, args.num_vars, args.width, args.threshold,
            args.host, args.port, original_count, len(filtered_trajectories)
        )
        
        # Step 5: Save to file
        save_dataset_to_file(dataset, args.output, logger)
        
        logger.info("Data cleaning completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Script interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Script failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()