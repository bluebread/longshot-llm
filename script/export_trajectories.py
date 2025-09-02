#!/usr/bin/env python3
"""
Export Trajectories Script for longshot-llm

This script uses the WarehouseClient to export trajectory datasets from the warehouse service.
It supports filtering by num_vars and width, and outputs to a compact JSON format.

The output JSON structure:
{
    "trajectories": [
        {
            "traj_id": "uuid",
            "num_vars": int,
            "width": int,
            "steps": [
                [type, litint, avgQ],  # Each step as a 3-element list
                ...
            ]
        },
        ...
    ]
}

Usage:
    # Export all trajectories
    python script/export_trajectories.py --output trajectories.json
    
    # Export trajectories with specific num_vars and width
    python script/export_trajectories.py --output data.json --num-vars 4 --width 3
    
    # Export from remote warehouse
    python script/export_trajectories.py --output data.json --host remote.server --port 8080

Options:
    --output, -o        : Output JSON file path (required)
    --num-vars          : Filter by number of variables
    --width             : Filter by formula width
    --host              : Warehouse host (default: localhost)
    --port              : Warehouse port (default: 8000)
    --since             : Filter trajectories after this date (ISO format)
    --until             : Filter trajectories before this date (ISO format)
    --include-metadata  : Include traj_id, num_vars, width in output
    --pretty            : Pretty-print JSON output (larger file size)
    --verbose           : Enable verbose logging
"""

import argparse
import json
import sys
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from longshot.service.warehouse import WarehouseClient


def setup_logging(verbose: bool) -> logging.Logger:
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


def transform_trajectory(
    trajectory: Dict[str, Any], 
    include_metadata: bool,
    logger: logging.Logger
) -> Optional[Dict[str, Any]]:
    """
    Transform a trajectory from warehouse format to the export format.
    
    Args:
        trajectory: Raw trajectory from warehouse
        include_metadata: Whether to include metadata fields
        logger: Logger instance
        
    Returns:
        Transformed trajectory or None if invalid
    """
    try:
        traj_id = trajectory.get('traj_id', 'unknown')
        steps = trajectory.get('steps', [])
        
        if not steps:
            logger.debug(f"Trajectory {traj_id} has no steps, skipping")
            return None
        
        # Validate and keep steps as list of lists
        validated_steps = []
        for i, step in enumerate(steps):
            if isinstance(step, (list, tuple)) and len(step) >= 3:
                # Keep steps in format: [type, litint, avgQ]
                validated_steps.append([step[0], step[1], step[2]])
            else:
                logger.debug(f"Invalid step format in trajectory {traj_id}: {step}")
                return None
        
        # Build output dictionary
        result = {
            "steps": validated_steps
        }
        
        # Add metadata if requested
        if include_metadata:
            result = {
                "traj_id": traj_id,
                "num_vars": trajectory.get('num_vars'),
                "width": trajectory.get('width'),
                **result
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to transform trajectory {trajectory.get('traj_id', 'unknown')}: {e}")
        return None


def export_trajectories(
    client: WarehouseClient,
    output_file: str,
    num_vars: Optional[int],
    width: Optional[int],
    since: Optional[datetime],
    until: Optional[datetime],
    include_metadata: bool,
    pretty: bool,
    logger: logging.Logger
) -> tuple[int, int]:
    """
    Export trajectories from warehouse to JSON file.
    
    Returns:
        Tuple of (processed_count, failed_count)
    """
    processed_count = 0
    failed_count = 0
    
    try:
        # Get trajectory dataset with filters
        logger.info("📊 Fetching trajectory dataset from warehouse...")
        logger.info(f"   Filters: num_vars={num_vars}, width={width}")
        if since:
            logger.info(f"   Since: {since}")
        if until:
            logger.info(f"   Until: {until}")
        
        dataset = client.get_trajectory_dataset(
            num_vars=num_vars,
            width=width,
            since=since,
            until=until
        )
        
        trajectories = dataset.get('trajectories', [])
        logger.info(f"   Found {len(trajectories)} trajectories matching filters")
        
        # Transform trajectories
        logger.info("🔄 Processing trajectories...")
        transformed_trajectories = []
        
        for trajectory in trajectories:
            transformed = transform_trajectory(trajectory, include_metadata, logger)
            if transformed:
                transformed_trajectories.append(transformed)
                processed_count += 1
            else:
                failed_count += 1
        
        # Write to output file
        logger.info(f"💾 Writing to {output_file}...")
        with open(output_file, 'w') as f:
            output_data = {"trajectories": transformed_trajectories}
            if pretty:
                json.dump(output_data, f, indent=2)
            else:
                # Compact format with no spaces
                json.dump(output_data, f, separators=(',', ':'))
        
        logger.info(f"✅ Successfully exported {processed_count} trajectories to {output_file}")
        
        if failed_count > 0:
            logger.warning(f"⚠️  {failed_count} trajectories failed to process")
        
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        raise
    
    return processed_count, failed_count


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Export trajectory datasets from warehouse service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all trajectories
  %(prog)s --output trajectories.json

  # Export trajectories with specific parameters
  %(prog)s --output data.json --num-vars 4 --width 3

  # Export with metadata included
  %(prog)s --output data.json --include-metadata

  # Export from remote warehouse
  %(prog)s --output data.json --host remote.server --port 8080
        """
    )
    
    # Output arguments
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSON file path"
    )
    
    # Filter arguments
    parser.add_argument(
        "--num-vars",
        type=int,
        help="Filter by number of variables"
    )
    parser.add_argument(
        "--width",
        type=int,
        help="Filter by formula width"
    )
    parser.add_argument(
        "--since",
        type=str,
        help="Filter trajectories after this date (ISO format, e.g., 2024-01-01T00:00:00)"
    )
    parser.add_argument(
        "--until",
        type=str,
        help="Filter trajectories before this date (ISO format, e.g., 2024-12-31T23:59:59)"
    )
    
    # Warehouse connection arguments
    parser.add_argument(
        "--host",
        default="localhost",
        help="Warehouse host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Warehouse port (default: 8000)"
    )
    
    # Output format arguments
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include traj_id, num_vars, width in output"
    )
    parser.add_argument(
        "--pretty", "-p",
        action="store_true",
        help="Pretty-print JSON output (larger file size)"
    )
    
    # Logging arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Parse date filters if provided
    since = None
    until = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since)
        except ValueError:
            logger.error(f"Invalid date format for --since: {args.since}")
            logger.error("Use ISO format, e.g., 2024-01-01T00:00:00")
            return 1
    
    if args.until:
        try:
            until = datetime.fromisoformat(args.until)
        except ValueError:
            logger.error(f"Invalid date format for --until: {args.until}")
            logger.error("Use ISO format, e.g., 2024-12-31T23:59:59")
            return 1
    
    logger.info("🚀 Starting trajectory export from warehouse")
    logger.info(f"   Warehouse: http://{args.host}:{args.port}")
    
    # Initialize warehouse client
    client = WarehouseClient(host=args.host, port=args.port)
    
    try:
        # Export trajectories
        processed_count, failed_count = export_trajectories(
            client,
            args.output,
            args.num_vars,
            args.width,
            since,
            until,
            args.include_metadata,
            args.pretty,
            logger
        )
        
        # Print summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("📋 EXPORT SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Warehouse: http://{args.host}:{args.port}")
        logger.info(f"Output file: {args.output}")
        logger.info(f"Trajectories exported: {processed_count}")
        if failed_count > 0:
            logger.info(f"Trajectories failed: {failed_count}")
        if args.num_vars:
            logger.info(f"Filter num_vars: {args.num_vars}")
        if args.width:
            logger.info(f"Filter width: {args.width}")
        if since:
            logger.info(f"Filter since: {since.isoformat()}")
        if until:
            logger.info(f"Filter until: {until.isoformat()}")
        logger.info(f"Include metadata: {args.include_metadata}")
        logger.info(f"Compact format: {not args.pretty}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info("=" * 60)
        
        return 0 if processed_count > 0 or failed_count == 0 else 1
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)