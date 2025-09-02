#!/usr/bin/env python3
"""
Export Trajectories Script for longshot-llm

This script directly connects to MongoDB and exports trajectory data to a JSON dataset format.
Credentials are loaded from environment variables for security.

The output JSON structure:
{
    "trajectories": [
        {
            "type": [int, ...],        # Token types for each step
            "litint": [int, ...],      # Token literals for each step  
            "avgQ": [float, ...]       # Average Q values for each step
        },
        ...
    ]
}

Usage:
    # Set environment variables first:
    export MONGO_USER=your_username
    export MONGO_PASSWORD=your_password
    
    # Then run:
    python export_trajectories.py --output-file trajectories.json
    
Options:
    --output-file       : Output JSON file path (required)
    --limit             : Maximum number of trajectories to export
    --all               : Export all trajectories (ignores --limit)
    --mongo-host        : MongoDB host (default: from MONGO_HOST env or localhost)
    --mongo-port        : MongoDB port (default: from MONGO_PORT env or 27017)
    --mongo-db          : MongoDB database name (default: from MONGO_DB env or LongshotWarehouse)
    --verbose           : Enable verbose logging
    --pretty            : Pretty-print JSON output (larger file size)
    --batch-size        : Process trajectories in batches (default: 1000)
    
Example:
    # Export ALL trajectories
    python export_trajectories.py --output-file trajectories.json --all
    
    # Export with limit and pretty formatting
    python export_trajectories.py --output-file data.json --limit 100 --pretty
    
    # Export with custom MongoDB connection
    python export_trajectories.py --output-file trajectories.json --mongo-host localhost
"""

import argparse
import json
import sys
import os
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime
import logging
from contextlib import contextmanager
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

def setup_logging(verbose: bool):
    """Configure logging based on verbosity."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)

def get_mongo_config(args) -> Dict[str, Any]:
    """Get MongoDB configuration from environment and arguments."""
    config = {
        'host': args.mongo_host or os.getenv('MONGO_HOST', 'localhost'),
        'port': args.mongo_port or int(os.getenv('MONGO_PORT', '27017')),
        'user': os.getenv('MONGO_USER'),
        'password': os.getenv('MONGO_PASSWORD'),
        'database': args.mongo_db or os.getenv('MONGO_DB', 'LongshotWarehouse')
    }
    
    # Override with test defaults if in test mode (for backwards compatibility)
    if os.getenv('LONGSHOT_TEST_MODE') == '1':
        config['host'] = config.get('host') or 'mongo-bread'
        config['user'] = config.get('user') or 'haowei'
        config['password'] = config.get('password') or 'bread861122'
    
    return config

@contextmanager
def mongodb_connection(config: Dict[str, Any], logger: logging.Logger):
    """Context manager for MongoDB connections."""
    client = None
    try:
        if not config['user'] or not config['password']:
            raise ValueError(
                "MongoDB credentials not provided. "
                "Please set MONGO_USER and MONGO_PASSWORD environment variables, "
                "or use LONGSHOT_TEST_MODE=1 for test defaults."
            )
        
        connection_string = f"mongodb://{config['user']}:{config['password']}@{config['host']}:{config['port']}"
        logger.debug(f"Connecting to MongoDB at {config['host']}:{config['port']} as user '{config['user']}'")
        
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        logger.info(f"✅ Successfully connected to MongoDB")
        yield client
        
    except ConnectionFailure as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        raise
    except ServerSelectionTimeoutError as e:
        logger.error(f"❌ MongoDB connection timeout: {e}")
        logger.error(f"   Check if MongoDB is running at {config['host']}:{config['port']}")
        raise
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error connecting to MongoDB: {e}")
        raise
    finally:
        if client:
            client.close()
            logger.debug("Closed MongoDB connection")

def fetch_trajectories_streaming(
    client: MongoClient, 
    db_name: str, 
    limit: Optional[int], 
    batch_size: int,
    logger: logging.Logger
) -> Iterator[Dict[str, Any]]:
    """Fetch trajectories from MongoDB with streaming and projection."""
    try:
        db = client[db_name]
        trajectory_collection = db["TrajectoryTable"]
        
        # Use projection to only fetch required fields
        projection = {
            '_id': 1,
            'steps.token_type': 1,
            'steps.token_literals': 1,
            'steps.cur_avgQ': 1
        }
        
        # Count total trajectories
        total_count = trajectory_collection.count_documents({})
        logger.info(f"📊 Found {total_count} trajectories in database")
        
        # Build cursor with projection
        cursor = trajectory_collection.find({}, projection)
        
        if limit and limit > 0:
            cursor = cursor.limit(limit)
            logger.info(f"   Limiting export to {limit} trajectories")
        
        # Configure batch size for cursor
        cursor = cursor.batch_size(batch_size)
        
        # Return iterator
        return cursor
        
    except Exception as e:
        logger.error(f"❌ Failed to fetch trajectories from database: {e}")
        return iter([])  # Return empty iterator

def transform_trajectory(trajectory: Dict[str, Any], logger: logging.Logger) -> Optional[Dict[str, List]]:
    """
    Transform a trajectory from MongoDB format to the desired output format.
    
    Input format (from MongoDB):
    {
        "_id": "uuid",
        "steps": [
            {
                "token_type": int,
                "token_literals": int,
                "cur_avgQ": float
            },
            ...
        ]
    }
    
    Output format:
    {
        "type": [token_type, ...],
        "litint": [token_literals, ...],
        "avgQ": [cur_avgQ, ...]
    }
    """
    try:
        trajectory_id = trajectory.get('_id', 'unknown')
        steps = trajectory.get("steps", [])
        
        if not steps:
            logger.debug(f"Trajectory {trajectory_id} has no steps, skipping")
            return None
        
        # Initialize lists for each field
        types = []
        litints = []
        avgQs = []
        
        # Extract data from each step
        for i, step in enumerate(steps):
            if isinstance(step, dict):
                # Handle dictionary format (legacy MongoDB format)
                token_type = step.get("token_type", 0)  # Default to 0 if missing
                token_literals = step.get("token_literals")
                cur_avgQ = step.get("cur_avgQ")
                
                if token_literals is None or cur_avgQ is None:
                    logger.debug(f"Missing required fields in step {i} of trajectory {trajectory_id}")
                    return None
                    
                types.append(token_type)
                litints.append(token_literals)
                avgQs.append(cur_avgQ)
            elif isinstance(step, (list, tuple)) and len(step) == 3:
                # Handle tuple/list format (token_type, token_literals, cur_avgQ)
                types.append(step[0])
                litints.append(step[1])
                avgQs.append(step[2])
            else:
                logger.debug(f"Invalid step format in trajectory {trajectory_id}: {step}")
                return None
        
        # Validate that all lists have the same length
        if not (len(types) == len(litints) == len(avgQs)):
            logger.error(f"Inconsistent field lengths in trajectory {trajectory_id}")
            return None
        
        return {
            "type": types,
            "litint": litints,
            "avgQ": avgQs
        }
        
    except Exception as e:
        logger.error(f"Failed to transform trajectory {trajectory.get('_id', 'unknown')}: {e}")
        return None

def process_and_export_streaming(
    cursor: Iterator[Dict[str, Any]],
    output_file: str,
    pretty: bool,
    batch_size: int,
    logger: logging.Logger
) -> tuple[int, int]:
    """Process trajectories in a streaming fashion to minimize memory usage."""
    processed_count = 0
    failed_count = 0
    
    try:
        with open(output_file, 'w') as f:
            # Start JSON structure
            f.write('{"trajectories":[')
            first_trajectory = True
            
            batch_count = 0
            for trajectory in cursor:
                transformed = transform_trajectory(trajectory, logger)
                
                if transformed:
                    if not first_trajectory:
                        f.write(',')
                    
                    if pretty:
                        f.write('\n  ')
                        json.dump(transformed, f, indent=2)
                    else:
                        json.dump(transformed, f, separators=(',', ':'))
                    
                    first_trajectory = False
                    processed_count += 1
                else:
                    failed_count += 1
                
                batch_count += 1
                if batch_count % batch_size == 0:
                    logger.debug(f"Processed {batch_count} trajectories...")
            
            # Close JSON structure
            if pretty:
                f.write('\n]}\n')
            else:
                f.write(']}')
        
        logger.info(f"✅ Successfully exported {processed_count} trajectories to {output_file}")
        
    except IOError as e:
        logger.error(f"❌ Failed to write output file: {e}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during export: {e}")
        raise
    
    return processed_count, failed_count

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Export trajectory data directly from MongoDB to JSON format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Output arguments
    parser.add_argument(
        "--output-file", "-o",
        required=True,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Maximum number of trajectories to export"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Export all trajectories (ignores --limit)"
    )
    
    # MongoDB arguments
    parser.add_argument(
        "--mongo-host",
        help="MongoDB host (overrides MONGO_HOST env)"
    )
    parser.add_argument(
        "--mongo-port",
        type=int,
        help="MongoDB port (overrides MONGO_PORT env)"
    )
    parser.add_argument(
        "--mongo-db",
        help="MongoDB database name (overrides MONGO_DB env)"
    )
    
    # Processing arguments
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1000,
        help="Process trajectories in batches (default: 1000)"
    )
    
    # Formatting and logging arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--pretty", "-p",
        action="store_true",
        help="Pretty-print JSON output (larger file size)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Get configuration
    config = get_mongo_config(args)
    
    logger.info("🚀 Starting trajectory export from MongoDB")
    logger.info(f"   Database: {config['database']} at {config['host']}:{config['port']}")
    
    try:
        with mongodb_connection(config, logger) as client:
            # Determine limit based on --all flag
            limit = None if args.all else args.limit
            if args.all and args.limit:
                logger.warning("--all flag specified, ignoring --limit")
            
            # Fetch trajectories with streaming
            cursor = fetch_trajectories_streaming(
                client,
                config['database'],
                limit,
                args.batch_size,
                logger
            )
            
            # Process and export with streaming
            logger.info("🔄 Processing and exporting trajectories...")
            processed_count, failed_count = process_and_export_streaming(
                cursor,
                args.output_file,
                args.pretty,
                args.batch_size,
                logger
            )
            
            # Print summary
            logger.info("")
            logger.info("=" * 60)
            logger.info("📋 EXPORT SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Database: {config['database']}@{config['host']}:{config['port']}")
            logger.info(f"Output file: {args.output_file}")
            logger.info(f"Trajectories exported: {processed_count}")
            if failed_count > 0:
                logger.info(f"Trajectories failed: {failed_count}")
            logger.info(f"Batch size: {args.batch_size}")
            logger.info(f"Timestamp: {datetime.now().isoformat()}")
            logger.info("=" * 60)
            
            return 0 if processed_count > 0 or failed_count == 0 else 1
            
    except (ConnectionFailure, ServerSelectionTimeoutError, ValueError) as e:
        logger.error("Cannot proceed - MongoDB connection failed")
        return 1
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return 1

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