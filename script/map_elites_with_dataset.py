#!/usr/bin/env python3
"""
MAP-Elites with Dataset Download

Extends the MAP-Elites algorithm to download trajectory datasets after completion.
Supports downloading specific datasets or all trajectories generated during optimization.

Ussage: 

python map_elites_with_dataset.py --iterations 20 --num-vars 4 --width 2 --strategy performance --batch-size 10 --size 20 --mutate-length 10 --cell-density 3 --output ../output/me-n4w2.json --elites-only --dataset-output ../output/traj-n4w2.json --init-strategy random
"""

import json
import argparse
import time
from datetime import datetime
from typing import Optional

from map_elites import MAPElites, MAPElitesConfig
from longshot.agent import WarehouseAgent


def download_trajectory_dataset(
    warehouse_host: str = "localhost",
    warehouse_port: int = 8000,
    output_path: str = "trajectories_dataset.json",
    elite_traj_ids: Optional[set] = None
):
    """
    Download trajectory dataset from warehouse
    
    Args:
        warehouse_host: Warehouse service host
        warehouse_port: Warehouse service port
        output_path: Path to save the dataset
        elite_traj_ids: Optional set of specific trajectory IDs to filter
    
    Returns:
        Dictionary containing the downloaded dataset
    """
    warehouse = WarehouseAgent(warehouse_host, warehouse_port)
    
    retry_count = 0
    
    while True:
        try:
            # Always download the complete dataset from warehouse
            print(f"Downloading trajectory dataset from warehouse... (attempt {retry_count + 1})")
            response = warehouse._client.get("/trajectory/dataset")
            response.raise_for_status()
            
            dataset = response.json()
            all_trajectories = dataset.get("trajectories", [])
            break  # Success, exit retry loop
            
        except Exception as e:
            retry_count += 1
            print(f"Download attempt {retry_count} failed: {e}")
            print("Retrying in 10 seconds...")
            time.sleep(10)
            continue
    
    try:
        
        # Filter by elite IDs if provided
        if elite_traj_ids:
            print(f"Filtering for {len(elite_traj_ids)} elite trajectories...")
            filtered_trajectories = [
                traj for traj in all_trajectories
                if traj.get("traj_id") in elite_traj_ids
            ]
            dataset["trajectories"] = filtered_trajectories
            dataset["original_count"] = len(all_trajectories)
            dataset["filtered_count"] = len(filtered_trajectories)
            print(f"Found {len(filtered_trajectories)} elite trajectories out of {len(all_trajectories)} total")
        
        # Add metadata
        dataset["download_timestamp"] = datetime.now().isoformat()
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        trajectory_count = len(dataset.get("trajectories", []))
        print(f"Saved {trajectory_count} trajectories to {output_path}")
        
        return dataset
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None
    finally:
        warehouse._client.close()


def run_map_elites_with_dataset(config: MAPElitesConfig, dataset_args: dict, bootstrap: bool = False):
    """
    Run MAP-Elites and download dataset after completion
    
    Args:
        config: MAP-Elites configuration
        dataset_args: Arguments for dataset download
        bootstrap: If True, generate initial trajectories if archive is empty
    """
    # Run MAP-Elites
    map_elites = MAPElites(config)
    
    try:
        # Execute the algorithm
        map_elites.run()
        
        # If archive is empty and bootstrap is enabled, generate initial trajectories
        if bootstrap and map_elites.archive.get_statistics()["total_cells"] == 0:
            print("\n" + "="*60)
            print("Archive empty - generating bootstrap trajectories...")
            print("="*60)
            
            # Generate initial trajectories using clusterbomb
            from longshot.agent import ClusterbombAgent
            clusterbomb = ClusterbombAgent(config.clusterbomb_host, config.clusterbomb_port)
            
            try:
                # Generate random initial trajectories
                for i in range(5):  # Generate 5 batches
                    print(f"Generating bootstrap batch {i+1}/5...")
                    response = clusterbomb.weapon_rollout(
                        num_vars=config.max_num_vars,
                        width=config.max_width,
                        size=config.max_size,
                        steps_per_trajectory=20,  # Longer initial trajectories
                        num_trajectories=10,
                        prefix_traj=[],  # Start from scratch
                        early_stop=True,
                        trajproc_config={
                            "iterations": config.trajproc_iterations,
                            "granularity": config.trajproc_granularity,
                            "num_summits": config.trajproc_num_summits
                        }
                    )
                    print(f"  Generated {response.num_trajectories} trajectories")
                
                # Re-initialize from warehouse with the new trajectories
                print("\nRe-initializing MAP-Elites with bootstrap trajectories...")
                map_elites.initialize_from_warehouse()
                
                # Run evolution again with populated archive
                if map_elites.archive.get_statistics()["total_cells"] > 0:
                    print("Running evolution with bootstrap data...")
                    for iteration in range(config.num_iterations):
                        map_elites.current_iteration = iteration + 1
                        print(f"\n--- Iteration {map_elites.current_iteration}/{config.num_iterations} ---")
                        
                        start_time = time.time()
                        map_elites.evolution_step()
                        elapsed = time.time() - start_time
                        
                        # Report statistics
                        stats = map_elites.archive.get_statistics()
                        discoveries = map_elites.archive.iteration_discoveries.get(map_elites.current_iteration, 0)
                        
                        print(f"  Time: {elapsed:.2f}s")
                        print(f"  New discoveries: {discoveries}")
                        print(f"  Archive size: {stats['total_cells']} cells, {stats['total_elites']} elites")
                        print(f"  Performance: avg={stats['avg_avgQ']:.4f}, max={stats['max_avgQ']:.4f}")
                
            finally:
                clusterbomb.close()
        
        # Extract elite trajectory IDs from the archive
        elite_traj_ids = set()
        for cell_elites in map_elites.archive.cells.values():
            for elite in cell_elites:
                elite_traj_ids.add(elite.traj_id)
        
        print(f"\n{'='*60}")
        print("Downloading trajectory dataset...")
        print(f"{'='*60}")
        
        # Prepare dataset download arguments
        download_args = {
            "warehouse_host": config.warehouse_host,
            "warehouse_port": config.warehouse_port,
            "output_path": dataset_args.get("output_path", "trajectories_dataset.json"),
        }
        
        # Optionally download only elite trajectories
        if dataset_args.get("elites_only", False):
            download_args["elite_traj_ids"] = elite_traj_ids  # Pass as set
            print(f"Filtering dataset for {len(elite_traj_ids)} elite trajectories...")
        
        # Download the dataset
        dataset = download_trajectory_dataset(**download_args)
        
        if dataset:
            # Add MAP-Elites metadata to dataset
            dataset["map_elites_metadata"] = {
                "archive_path": config.archive_path,
                "num_iterations": config.num_iterations,
                "final_archive_stats": map_elites.archive.get_statistics(),
                "total_elite_trajectories": len(elite_traj_ids),
                "elite_traj_ids": list(elite_traj_ids)  # Always include the elite IDs for reference
            }
            
            # Save updated dataset with metadata
            with open(download_args["output_path"], 'w') as f:
                json.dump(dataset, f, indent=2)
            
            print(f"\nDataset with MAP-Elites metadata saved to: {download_args['output_path']}")
            
            # Print summary statistics
            if "trajectories" in dataset:
                total_steps = sum(len(t.get("steps", [])) for t in dataset["trajectories"])
                print(f"\nDataset Statistics:")
                print(f"  - Total trajectories: {len(dataset['trajectories'])}")
                print(f"  - Total steps: {total_steps}")
                print(f"  - Elite trajectories: {len(elite_traj_ids)}")
                
    finally:
        map_elites.close()


def main():
    """Main entry point with dataset download support"""
    parser = argparse.ArgumentParser(
        description="MAP-Elites with trajectory dataset download",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run MAP-Elites and download all trajectories
  %(prog)s --iterations 50
  
  # Run MAP-Elites and download only elite trajectories
  %(prog)s --iterations 50 --elites-only
  
  # Run with random initialization if warehouse is empty
  %(prog)s --iterations 50 --init-strategy random
  
  # Run with bootstrap (generates initial trajectories if warehouse is empty)
  %(prog)s --iterations 50 --bootstrap
  
  # Specify custom output paths
  %(prog)s --iterations 50 --output archive.json --dataset-output trajectories.json
        """
    )
    
    # MAP-Elites parameters
    parser.add_argument("--iterations", type=int, default=50, help="Number of MAP-Elites iterations")
    parser.add_argument("--cell-density", type=int, default=1, help="Max elites per cell")
    parser.add_argument("--num-vars", type=int, default=4, help="Max number of variables")
    parser.add_argument("--width", type=int, default=3, help="Max formula width")
    parser.add_argument("--size", type=int, default=5, help="Max formula size")
    parser.add_argument("--mutate-length", type=int, default=10, help="Steps per mutation")
    parser.add_argument("--num-mutate", type=int, default=5, help="Trajectories per mutation")
    parser.add_argument("--batch-size", type=int, default=10, help="Mutations per iteration")
    parser.add_argument("--strategy", type=str, default="uniform",
                       choices=["uniform", "curiosity", "performance"],
                       help="Elite selection strategy")
    parser.add_argument("--init-strategy", type=str, default="warehouse",
                       choices=["warehouse", "random"],
                       help="Initialization strategy (warehouse: use existing, random: generate new)")
    
    # Service configuration
    parser.add_argument("--warehouse-host", type=str, default="localhost", help="Warehouse host")
    parser.add_argument("--warehouse-port", type=int, default=8000, help="Warehouse port")
    parser.add_argument("--clusterbomb-host", type=str, default="localhost", help="Clusterbomb host")
    parser.add_argument("--clusterbomb-port", type=int, default=8060, help="Clusterbomb port")
    
    # TrajectoryProcessor configuration
    parser.add_argument("--trajproc-iterations", type=int, default=5, help="WL hash iterations")
    parser.add_argument("--trajproc-granularity", type=int, default=20, help="Q-value discretization")
    parser.add_argument("--trajproc-num-summits", type=int, default=5, help="Number of summits")
    
    # Output configuration
    parser.add_argument("--output", type=str, default="map_elites_archive.json",
                       help="MAP-Elites archive output file")
    parser.add_argument("--dataset-output", type=str, default="trajectories_dataset.json",
                       help="Trajectory dataset output file")
    
    # Dataset download options
    parser.add_argument("--elites-only", action="store_true",
                       help="Download only elite trajectories from the archive")
    parser.add_argument("--skip-dataset", action="store_true",
                       help="Skip dataset download after MAP-Elites")
    parser.add_argument("--bootstrap", action="store_true",
                       help="Generate initial trajectories if warehouse is empty")
    
    # Other options
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    # Create MAP-Elites configuration
    config = MAPElitesConfig(
        num_iterations=args.iterations,
        cell_density=args.cell_density,
        max_num_vars=args.num_vars,
        max_width=args.width,
        max_size=args.size,
        mutate_length=args.mutate_length,
        num_mutate=args.num_mutate,
        batch_size=args.batch_size,
        elite_selection_strategy=args.strategy,
        initialization_strategy=args.init_strategy,
        warehouse_host=args.warehouse_host,
        warehouse_port=args.warehouse_port,
        clusterbomb_host=args.clusterbomb_host,
        clusterbomb_port=args.clusterbomb_port,
        trajproc_iterations=args.trajproc_iterations,
        trajproc_granularity=args.trajproc_granularity,
        trajproc_num_summits=args.trajproc_num_summits,
        archive_path=args.output,
        verbose=not args.quiet
    )
    
    # Prepare dataset download arguments
    dataset_args = {
        "output_path": args.dataset_output,
        "elites_only": args.elites_only
    }
    
    if args.skip_dataset:
        # Run MAP-Elites without dataset download
        map_elites = MAPElites(config)
        try:
            map_elites.run()
        finally:
            map_elites.close()
    else:
        # Run MAP-Elites with dataset download
        run_map_elites_with_dataset(config, dataset_args, bootstrap=args.bootstrap)


if __name__ == "__main__":
    main()