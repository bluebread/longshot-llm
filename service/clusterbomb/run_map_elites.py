#!/usr/bin/env python3
"""
Autonomous MAP-Elites runner for clusterbomb service.

This script runs MAP-Elites algorithm as a standalone job executor that
continuously collects trajectories and stores them in the warehouse.
No public API endpoints - runs as an autonomous container.

Usage:
    python run_map_elites.py [OPTIONS]

Basic Examples:
    # Run with default settings (100 iterations, 4 vars, width 3)
    python run_map_elites.py
    
    # Run with custom iterations and formula space
    python run_map_elites.py --iterations 500 --num-vars 5 --width 4
    
    # Run with synchronization enabled (for multi-instance deployment)
    python run_map_elites.py --enable-sync --sync-interval 10
    
    # Connect to remote warehouse
    python run_map_elites.py --warehouse-host warehouse.example.com --warehouse-port 8000
    
    # Run with increased parallelization
    python run_map_elites.py --num-workers 16 --batch-size 20

Core Algorithm Parameters:
    --iterations N        Number of MAP-Elites iterations (default: 100)
    --cell-density N      Maximum elites per cell (default: 1)
    
Formula Space Parameters:
    --num-vars N          Number of boolean variables (default: 4)
    --width N             Maximum formula width (default: 3)
    
Mutation Parameters:
    --num-steps N         Steps per trajectory mutation (default: 10)
    --num-trajectories N  Trajectories per mutation (default: 5)
    
Parallelization:
    --num-workers N       Number of parallel workers (default: all CPU cores)
    --batch-size N        Number of elites to mutate per iteration (default: 10)
    
Algorithm Strategy:
    --strategy TYPE       Elite selection strategy: uniform, curiosity, performance (default: uniform)
    --init-strategy TYPE  Initialization: warehouse, random (default: warehouse)
    
Synchronization (for distributed runs):
    --enable-sync         Enable synchronization with other instances via warehouse
    --sync-interval N     Iterations between syncs when enabled (default: 10)
    
Warehouse Connection:
    --warehouse-host HOST Warehouse service host (default: localhost)
    --warehouse-port PORT Warehouse service port (default: 8000)
    
Output Options:
    --output FILE         Archive output file (default: map_elites_archive.json)
    --quiet               Reduce output verbosity
    --no-save             Don't save archive to file

Advanced Examples:
    # High-performance configuration for large formula spaces
    python run_map_elites.py \\
        --iterations 1000 \\
        --num-vars 6 \\
        --width 4 \\
        --num-workers 32 \\
        --batch-size 50 \\
        --num-trajectories 10 \\
        --strategy performance
    
    # Distributed multi-instance deployment
    python run_map_elites.py \\
        --enable-sync \\
        --sync-interval 5 \\
        --warehouse-host central-warehouse.cluster.local \\
        --init-strategy warehouse \\
        --cell-density 3
    
    # Exploration-focused configuration
    python run_map_elites.py \\
        --strategy curiosity \\
        --num-steps 20 \\
        --num-trajectories 8 \\
        --batch-size 15 \\
        --output exploration_archive.json

    # Typical research setup
    python run_map_elites.py 
        --iterations 40 \\
        --cell-density 5 \\ 
        --num-vars 6 \\ 
        --width 6 \\
        --num-steps 128 \\ # 2 * (2^width)
        --batch-size 32 \\ # num_workers
        --init-strategy warehouse \\
        --timeout 60 \\
        --strategy performance \\
        2> log.txt &

Docker Deployment:
    # Build and run as container
    docker build -t clusterbomb-map-elites .
    docker run --network host clusterbomb-map-elites \\
        --warehouse-host warehouse \\
        --iterations 500

Environment Variables:
    WAREHOUSE_HOST        Alternative to --warehouse-host
    WAREHOUSE_PORT        Alternative to --warehouse-port
    MAP_ELITES_WORKERS    Alternative to --num-workers

Signal Handling:
    - SIGINT (Ctrl+C): Graceful shutdown, saves current archive
    - SIGTERM: Graceful shutdown for container orchestration

Notes:
    - The algorithm will automatically initialize from existing warehouse trajectories
    - Archive is saved periodically and on shutdown (unless --no-save)
    - Progress is logged to stdout with configurable verbosity
    - Multi-instance synchronization allows distributed exploration
    - Each instance maintains its own archive while sharing discoveries
"""

import asyncio
import argparse
import logging
import signal
import sys
from datetime import datetime

from models import MAPElitesConfig
from map_elites_service import MAPElitesService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    global shutdown_requested
    logger.info(f"Received signal {signum}, requesting shutdown...")
    shutdown_requested = True


async def run_map_elites(config: MAPElitesConfig):
    """
    Run MAP-Elites algorithm autonomously.
    
    Args:
        config: MAP-Elites configuration
    """
    logger.info("=" * 60)
    logger.info("Clusterbomb MAP-Elites Job Executor")
    logger.info("=" * 60)
    logger.info(f"Starting autonomous MAP-Elites execution")
    logger.info(f"Configuration:")
    logger.info(f"  - Iterations: {config.num_iterations}")
    logger.info(f"  - Formula space: {config.num_vars} vars, width {config.width}")
    logger.info(f"  - Synchronization: {'Enabled' if config.enable_sync else 'Disabled'}")
    if config.enable_sync:
        logger.info(f"  - Sync interval: every {config.sync_interval} iterations")
    logger.info(f"  - Warehouse: {config.warehouse_host}:{config.warehouse_port}")
    logger.info("=" * 60)
    
    # Create and run MAP-Elites service
    service = MAPElitesService(config)
    
    try:
        await service.run()
        logger.info("MAP-Elites execution completed successfully")
    except Exception as e:
        logger.error(f"MAP-Elites execution failed: {e}", exc_info=True)
        raise
    finally:
        # Save final archive
        if config.save_archive:
            service.save_archive()
            logger.info(f"Archive saved to {config.archive_path}")


def main():
    """Main entry point for the clusterbomb MAP-Elites executor"""
    
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Clusterbomb MAP-Elites Job Executor - Autonomous trajectory collection using quality-diversity optimization"
    )
    
    # Core algorithm parameters
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of MAP-Elites iterations (default: 100)")
    parser.add_argument("--cell-density", type=int, default=1,
                       help="Maximum elites per cell (default: 1)")
    
    # Formula space parameters
    parser.add_argument("--num-vars", type=int, default=4,
                       help="Number of boolean variables (default: 4)")
    parser.add_argument("--width", type=int, default=3,
                       help="Maximum formula width (default: 3)")
    
    # Mutation parameters
    parser.add_argument("--num-steps", type=int, default=10,
                       help="Steps per trajectory mutation (default: 10)")
    parser.add_argument("--num-trajectories", type=int, default=5,
                       help="Trajectories per mutation (default: 5)")
    
    # Parallelization
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of parallel workers (default: all CPU cores)")
    
    # Algorithm strategy
    parser.add_argument("--batch-size", type=int, default=10,
                       help="Number of elites to mutate per iteration (default: 10)")
    parser.add_argument("--strategy", type=str, default="uniform",
                       choices=["uniform", "curiosity", "performance"],
                       help="Elite selection strategy (default: uniform)")
    parser.add_argument("--init-strategy", type=str, default="warehouse",
                       choices=["warehouse", "random"],
                       help="Initialization strategy (default: warehouse)")
    
    # Synchronization settings
    parser.add_argument("--enable-sync", action="store_true",
                       help="Enable synchronization with other instances via warehouse")
    parser.add_argument("--sync-interval", type=int, default=10,
                       help="Iterations between syncs when enabled (default: 10)")
    
    # Service configuration
    parser.add_argument("--warehouse-host", type=str, 
                       default="localhost",
                       help="Warehouse service host (default: localhost)")
    parser.add_argument("--warehouse-port", type=int, 
                       default=8000,
                       help="Warehouse service port (default: 8000)")
    parser.add_argument("--timeout", type=float,
                       default=30.0,
                       help="HTTP timeout for warehouse requests in seconds (default: 30.0)")
    
    # Output options
    parser.add_argument("--output", type=str, 
                       default=None,
                       help="Archive output file (default: output/archive-n{N}w{W}.json based on num_vars and width)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save archive to file")
    
    args = parser.parse_args()
    
    # Generate default output filename if not specified
    if args.output is None:
        args.output = f"output/archive-n{args.num_vars}w{args.width}.json"
    
    # Create configuration
    config = MAPElitesConfig(
        num_iterations=args.iterations,
        cell_density=args.cell_density,
        num_vars=args.num_vars,
        width=args.width,
        num_steps=args.num_steps,
        num_trajectories=args.num_trajectories,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        elite_selection_strategy=args.strategy,
        initialization_strategy=args.init_strategy,
        enable_sync=args.enable_sync,
        sync_interval=args.sync_interval,
        warehouse_host=args.warehouse_host,
        warehouse_port=args.warehouse_port,
        warehouse_timeout=args.timeout,
        verbose=not args.quiet,
        save_archive=not args.no_save,
        archive_path=args.output
    )
    
    # Log startup information
    logger.info(f"Starting Clusterbomb MAP-Elites Job Executor")
    logger.info(f"Process started at: {datetime.now().isoformat()}")
    
    # Run the async main function
    try:
        asyncio.run(run_map_elites(config))
        logger.info("Clusterbomb MAP-Elites executor finished successfully")
        sys.exit(0)
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()