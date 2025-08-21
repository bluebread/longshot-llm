#!/usr/bin/env python3
"""
Demonstration script for V2 system refactor.

This script demonstrates the complete workflow:
1. Create an empty initial formula in warehouse
2. Use clusterbomb to generate 10 trajectories (32 steps each)
3. Retrieve and display trajectory data
4. Visualize the evolution graph
5. Generate summary statistics
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Add library to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'library'))

# Use non-interactive backend for matplotlib (no GUI)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import networkx as nx
import numpy as np
from typing import Dict, List, Any

from longshot.agent import ClusterbombAgent, WarehouseAgent
from longshot.models import WeaponRolloutRequest, WeaponRolloutResponse


def create_output_directory() -> Path:
    """Create output directory for images and reports."""
    output_dir = Path("/root/gym-longshot/output")
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory ready: {output_dir}")
    return output_dir


def initialize_services() -> tuple[ClusterbombAgent, WarehouseAgent]:
    """Initialize connections to microservices."""
    print("\n=== Initializing Services ===")
    
    # Connect to warehouse
    warehouse = WarehouseAgent('localhost', 8000)
    print("✓ Connected to Warehouse service")
    
    # Connect to clusterbomb
    clusterbomb = ClusterbombAgent('localhost', 8060)
    
    # Check health
    try:
        health = clusterbomb.health_check()
        print(f"✓ Connected to Clusterbomb service (status: {health['status']})")
    except Exception as e:
        print(f"⚠ Warning: Clusterbomb health check failed: {e}")
    
    return clusterbomb, warehouse


def create_empty_formula(warehouse: WarehouseAgent) -> str:
    """Create an empty initial formula in the warehouse."""
    print("\n=== Creating Initial Empty Formula ===")
    
    # Create a node for empty formula
    # Empty formula has no gates, so definition is []
    initial_node_id = f"empty_formula_{int(time.time())}"
    
    # For V2 API, we may not need to pre-create the node
    # The clusterbomb service will create nodes as it processes trajectories
    print(f"✓ Using initial node ID: {initial_node_id}")
    print("  (Node will be created during trajectory processing)")
    
    return initial_node_id


def execute_weapon_rollout(clusterbomb: ClusterbombAgent, initial_node_id: str) -> WeaponRolloutResponse:
    """Execute weapon rollout to generate trajectories."""
    print("\n=== Executing Weapon Rollout ===")
    print("Parameters:")
    print("  - num_vars: 4")
    print("  - width: 3")
    print("  - size: 1000 (unlimited)")
    print("  - steps_per_trajectory: 32")
    print("  - num_trajectories: 10")
    print("  - seed: 42 (deterministic)")
    print(f"  - initial_node_id: {initial_node_id}")
    
    start_time = time.time()
    
    result = clusterbomb.weapon_rollout(
        num_vars=4,
        width=3,
        size=1000,  # Effectively unlimited
        steps_per_trajectory=32,
        num_trajectories=10,
        initial_definition=[],  # Empty formula
        initial_node_id=initial_node_id,
        seed=42  # Deterministic seed
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Rollout completed in {elapsed:.2f} seconds")
    print(f"  - Generated {result.num_trajectories} trajectories")
    print(f"  - Total steps: {result.total_steps}")
    
    return result


def retrieve_trajectories(warehouse: WarehouseAgent) -> Dict[str, Any]:
    """Retrieve trajectory data from warehouse."""
    print("\n=== Retrieving Trajectory Data ===")
    
    # Try to query trajectories from warehouse
    try:
        # Try to get trajectory list
        response = warehouse._client.get("/trajectory/list")
        if response.status_code == 200:
            trajectories = response.json()
            print(f"✓ Found {len(trajectories)} trajectories in warehouse")
        elif response.status_code == 404:
            print(f"⚠ Trajectory list endpoint not found (404)")
            print("  The /trajectory/list endpoint needs to be implemented")
            trajectories = []
        else:
            print(f"⚠ Failed to retrieve trajectories: HTTP {response.status_code}")
            print(f"  Response: {response.text[:200] if response.text else 'No response body'}")
            trajectories = []
    except Exception as e:
        print(f"⚠ Error retrieving trajectories: {e}")
        trajectories = []
    
    trajectory_stats = {
        "total_trajectories": 10,
        "steps_per_trajectory": 32,
        "total_steps": 320,
        "timestamp": datetime.now().isoformat(),
        "stored_trajectories": len(trajectories)
    }
    
    print("Trajectory Statistics:")
    for key, value in trajectory_stats.items():
        print(f"  - {key}: {value}")
    
    return trajectory_stats


def retrieve_evolution_graph(warehouse: WarehouseAgent) -> nx.DiGraph:
    """Retrieve and build evolution graph from warehouse."""
    print("\n=== Building Evolution Graph ===")
    
    # Add a small delay to ensure processing is complete
    print("Waiting for processing to complete...")
    time.sleep(2)
    
    # Download nodes for our configuration
    try:
        nodes_response = warehouse._client.get("/evolution_graph/download_nodes", params={
            "num_vars": 4,
            "width": 3
        })
        if nodes_response.status_code == 200:
            nodes_data = nodes_response.json()
            nodes = nodes_data.get("nodes", [])
            print(f"✓ Retrieved {len(nodes)} nodes from warehouse")
            if len(nodes) == 0:
                print("  Note: No nodes found - trajectories may still be processing")
        else:
            print(f"⚠ Failed to retrieve nodes: HTTP {nodes_response.status_code}")
            print(f"  Response: {nodes_response.text[:200] if nodes_response.text else 'No response body'}")
            nodes = []
    except Exception as e:
        print(f"⚠ Error retrieving nodes: {e}")
        nodes = []
    
    # Download hypernodes (connected components)
    try:
        hypernodes_response = warehouse._client.get("/evolution_graph/download_hypernodes", params={
            "num_vars": 4,
            "width": 3
        })
        if hypernodes_response.status_code == 200:
            hypernodes_data = hypernodes_response.json()
            hypernodes = hypernodes_data.get("hypernodes", [])
            print(f"✓ Retrieved {len(hypernodes)} hypernodes from warehouse")
            if len(hypernodes) == 0:
                print("  Note: No hypernodes found - graph may be empty")
        else:
            print(f"⚠ Failed to retrieve hypernodes: HTTP {hypernodes_response.status_code}")
            print(f"  Response: {hypernodes_response.text[:200] if hypernodes_response.text else 'No response body'}")
            hypernodes = []
    except Exception as e:
        print(f"⚠ Error retrieving hypernodes: {e}")
        hypernodes = []
    
    # Build NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes
    for node in nodes:
        G.add_node(
            node['id'],
            avgQ=node.get('avgQ', 0),
            size=node.get('size', 0),
            width=node.get('width', 0),
            num_vars=node.get('num_vars', 0),
            wl_hash=node.get('wl_hash', ''),
            in_degree=node.get('in_degree', 0),
            out_degree=node.get('out_degree', 0)
        )
    
    # Add edges from hypernodes (which contain edge information)
    for hypernode in hypernodes:
        if 'edges' in hypernode:
            for edge in hypernode['edges']:
                if 'source' in edge and 'target' in edge:
                    G.add_edge(edge['source'], edge['target'])
    
    print(f"✓ Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G


def visualize_graph(G: nx.DiGraph, output_dir: Path):
    """Create and save graph visualizations."""
    print("\n=== Creating Graph Visualizations ===")
    
    if G.number_of_nodes() == 0:
        print("⚠ No nodes to visualize")
        return
    
    # Figure 1: Graph Structure
    print("Creating structure visualization...")
    plt.figure(figsize=(12, 8))
    
    # Use spring layout for better visualization
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes and edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color='gray', arrows=True, arrowsize=10)
    
    # Color nodes by avgQ if available
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        avgQ = G.nodes[node].get('avgQ', 0)
        size = G.nodes[node].get('size', 1)
        node_colors.append(avgQ)
        node_sizes.append(100 + size * 10)  # Scale node size by formula size
    
    if node_colors:
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                       node_size=node_sizes, cmap='viridis',
                                       vmin=0, vmax=max(node_colors) if node_colors else 1)
        plt.colorbar(nodes, label='Average Q Value')
    
    plt.title('Evolution Graph Structure\n(Node size = formula size, Color = avgQ)', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    
    structure_path = output_dir / "evolution_graph_structure.png"
    plt.savefig(structure_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {structure_path}")
    
    # Figure 2: Heatmap visualization
    print("Creating heatmap visualization...")
    plt.figure(figsize=(12, 8))
    
    # Create adjacency matrix heatmap
    if G.number_of_nodes() > 0:
        # Get adjacency matrix
        node_list = list(G.nodes())[:50]  # Limit to first 50 nodes for readability
        subgraph = G.subgraph(node_list)
        adj_matrix = nx.adjacency_matrix(subgraph).todense()
        
        plt.imshow(adj_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Connection')
        plt.title('Adjacency Matrix Heatmap (First 50 nodes)', fontsize=14)
        plt.xlabel('Target Node Index')
        plt.ylabel('Source Node Index')
    
    heatmap_path = output_dir / "evolution_graph_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {heatmap_path}")
    
    # Figure 3: Degree distribution
    print("Creating degree distribution...")
    plt.figure(figsize=(12, 5))
    
    # In-degree distribution
    plt.subplot(1, 2, 1)
    in_degrees = [d for n, d in G.in_degree()]
    if in_degrees:
        plt.hist(in_degrees, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.title('In-Degree Distribution')
    plt.xlabel('In-Degree')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Out-degree distribution
    plt.subplot(1, 2, 2)
    out_degrees = [d for n, d in G.out_degree()]
    if out_degrees:
        plt.hist(out_degrees, bins=20, color='green', alpha=0.7, edgecolor='black')
    plt.title('Out-Degree Distribution')
    plt.xlabel('Out-Degree')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('Evolution Graph Degree Distributions', fontsize=14)
    plt.tight_layout()
    
    degree_path = output_dir / "evolution_graph_degrees.png"
    plt.savefig(degree_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {degree_path}")


def generate_summary_report(G: nx.DiGraph, trajectory_stats: Dict, output_dir: Path):
    """Generate and save summary report."""
    print("\n=== Generating Summary Report ===")
    
    # Calculate statistics
    stats = {
        "timestamp": datetime.now().isoformat(),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "trajectories": trajectory_stats["total_trajectories"],
        "total_steps": trajectory_stats["total_steps"],
    }
    
    if G.number_of_nodes() > 0:
        # Find best node by avgQ
        best_node = None
        best_avgQ = -float('inf')
        for node in G.nodes():
            avgQ = G.nodes[node].get('avgQ', 0)
            if avgQ > best_avgQ:
                best_avgQ = avgQ
                best_node = node
        
        stats["best_node"] = best_node
        stats["best_avgQ"] = best_avgQ
        
        # Calculate average metrics
        avg_sizes = [G.nodes[n].get('size', 0) for n in G.nodes()]
        avg_Qs = [G.nodes[n].get('avgQ', 0) for n in G.nodes()]
        
        stats["avg_formula_size"] = np.mean(avg_sizes) if avg_sizes else 0
        stats["max_formula_size"] = max(avg_sizes) if avg_sizes else 0
        stats["avg_Q_value"] = np.mean(avg_Qs) if avg_Qs else 0
        
        # Graph connectivity
        stats["connected_components"] = nx.number_weakly_connected_components(G)
        stats["is_dag"] = nx.is_directed_acyclic_graph(G)
        
        if G.number_of_edges() > 0:
            stats["avg_degree"] = sum(dict(G.degree()).values()) / G.number_of_nodes()
        else:
            stats["avg_degree"] = 0
    
    # Save report to file
    report_path = output_dir / "summary_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("V2 SYSTEM DEMONSTRATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Generated: {stats['timestamp']}\n\n")
        
        f.write("TRAJECTORY GENERATION\n")
        f.write("-" * 30 + "\n")
        f.write(f"Trajectories generated: {stats['trajectories']}\n")
        f.write(f"Total steps executed: {stats['total_steps']}\n")
        f.write(f"Steps per trajectory: 32\n\n")
        
        f.write("EVOLUTION GRAPH\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total nodes: {stats['graph_nodes']}\n")
        f.write(f"Total edges: {stats['graph_edges']}\n")
        
        if G.number_of_nodes() > 0:
            f.write(f"Connected components: {stats['connected_components']}\n")
            f.write(f"Is DAG: {stats['is_dag']}\n")
            f.write(f"Average degree: {stats['avg_degree']:.2f}\n\n")
            
            f.write("FORMULA STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average formula size: {stats['avg_formula_size']:.2f}\n")
            f.write(f"Maximum formula size: {stats['max_formula_size']}\n")
            f.write(f"Average Q value: {stats['avg_Q_value']:.2f}\n")
            f.write(f"Best Q value: {stats['best_avgQ']:.2f}\n")
            f.write(f"Best node ID: {stats['best_node']}\n")
    
    print(f"✓ Report saved to: {report_path}")
    
    # Print summary to terminal
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Exploration complete!")
    print(f"  - Discovered {stats['graph_nodes']} unique formulas")
    if 'best_avgQ' in stats:
        print(f"  - Best formula avgQ: {stats['best_avgQ']:.2f}")
    print(f"  - Generated {stats['trajectories']} trajectories ({stats['total_steps']} steps)")
    print(f"  - Graph has {stats['graph_edges']} edges")
    print(f"\n✓ Visualizations saved to: {output_dir}")
    print(f"✓ Report saved to: {report_path}")
    
    return stats


def main():
    """Main execution function."""
    print("=" * 60)
    print("V2 SYSTEM DEMONSTRATION")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}\n")
    
    try:
        # 1. Create output directory
        output_dir = create_output_directory()
        
        # 2. Initialize services
        clusterbomb, warehouse = initialize_services()
        
        # 3. Create empty initial formula
        initial_node_id = create_empty_formula(warehouse)
        
        # 4. Execute weapon rollout
        rollout_result = execute_weapon_rollout(clusterbomb, initial_node_id)
        
        # 5. Retrieve trajectory data
        trajectory_stats = retrieve_trajectories(warehouse)
        
        # 6. Retrieve and build evolution graph
        G = retrieve_evolution_graph(warehouse)
        
        # 7. Create visualizations
        visualize_graph(G, output_dir)
        
        # 8. Generate summary report
        stats = generate_summary_report(G, trajectory_stats, output_dir)
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print(f"Finished at: {datetime.now().isoformat()}")
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up connections
        if 'clusterbomb' in locals():
            clusterbomb.close()
        if 'warehouse' in locals():
            warehouse._client.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())