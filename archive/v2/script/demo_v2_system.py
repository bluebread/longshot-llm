#!/usr/bin/env python3
"""
Demonstration script for V2 system refactor.

This script demonstrates the complete workflow:
1. Create an empty initial formula in warehouse
2. Use clusterbomb to generate 10 trajectories (32 steps each)
3. Retrieve and display trajectory data
4. Visualize the evolution graph
5. Generate summary statistics
6. List trajectories with detailed analysis (optional)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict

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

from longshot.service import ClusterbombAgent, WarehouseClient
from longshot.models import WeaponRolloutRequest, WeaponRolloutResponse
from longshot.literals import Literals


def create_output_directory() -> Path:
    """Create output directory for images and reports."""
    output_dir = Path("/root/longshot-llm/output")
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory ready: {output_dir}")
    return output_dir


def initialize_services() -> tuple[ClusterbombAgent, WarehouseClient]:
    """Initialize connections to microservices."""
    print("\n=== Initializing Services ===")
    
    # Connect to warehouse
    warehouse = WarehouseClient('localhost', 8000)
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


def create_empty_formula(warehouse: WarehouseClient) -> str:
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


def execute_weapon_rollout(clusterbomb: ClusterbombAgent, initial_node_id: str, early_stop: bool = True) -> WeaponRolloutResponse:
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
    print(f"  - early_stop: {early_stop}")
    
    start_time = time.time()
    
    # Create empty prefix trajectory for V2 API
    empty_prefix_traj = []  # Empty formula = no steps needed
    
    result = clusterbomb.weapon_rollout(
        num_vars=4,
        width=3,
        size=1000,  # Effectively unlimited
        steps_per_trajectory=32,
        num_trajectories=10,
        prefix_traj=empty_prefix_traj,  # V2: Use prefix trajectory
        seed=42,  # Deterministic seed
        early_stop=early_stop
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n✓ Rollout completed in {elapsed:.2f} seconds")
    print(f"  - Generated {result.num_trajectories} trajectories")
    print(f"  - Total steps: {result.total_steps}")
    print(f"  - Processed formulas: {result.processed_formulas}")
    print(f"  - New nodes created: {len(result.new_nodes_created)} nodes")
    print(f"  - Evolution paths: {len(result.evopaths)} paths")
    print(f"  - Base formula exists: {result.base_formula_exists}")
    
    return result


def retrieve_trajectories(warehouse: WarehouseClient) -> Dict[str, Any]:
    """Retrieve trajectory data from warehouse using dataset endpoint."""
    print("\n=== Retrieving Trajectory Data ===")
    
    # Use the new trajectory dataset endpoint
    try:
        response = warehouse._client.get("/trajectory/dataset")
        if response.status_code == 200:
            dataset = response.json()
            trajectories = dataset.get("trajectories", [])
            print(f"✓ Found {len(trajectories)} trajectories in warehouse")
            
            # Calculate statistics from actual data
            total_steps = 0
            for traj in trajectories:
                steps = traj.get("steps", [])
                total_steps += len(steps)
            
            trajectory_stats = {
                "total_trajectories": len(trajectories),
                "steps_per_trajectory": total_steps // len(trajectories) if trajectories else 0,
                "total_steps": total_steps,
                "timestamp": datetime.now().isoformat(),
                "stored_trajectories": len(trajectories)
            }
            
        else:
            print(f"⚠ Failed to retrieve trajectories: HTTP {response.status_code}")
            print(f"  Response: {response.text[:200] if response.text else 'No response body'}")
            trajectories = []
            trajectory_stats = {
                "total_trajectories": 0,
                "steps_per_trajectory": 0,
                "total_steps": 0,
                "timestamp": datetime.now().isoformat(),
                "stored_trajectories": 0
            }
    except Exception as e:
        print(f"⚠ Error retrieving trajectories: {e}")
        trajectories = []
        trajectory_stats = {
            "total_trajectories": 0,
            "steps_per_trajectory": 0,
            "total_steps": 0,
            "timestamp": datetime.now().isoformat(),
            "stored_trajectories": 0
        }
    
    print("Trajectory Statistics:")
    for key, value in trajectory_stats.items():
        print(f"  - {key}: {value}")
    
    return trajectory_stats


def retrieve_evolution_graph(warehouse: WarehouseClient) -> nx.DiGraph:
    """Retrieve and build evolution graph from warehouse using dataset endpoint."""
    print("\n=== Building Evolution Graph ===")
    
    # Add a small delay to ensure processing is complete
    print("Waiting for processing to complete...")
    time.sleep(2)
    
    # Use the new dataset endpoint to get complete graph data
    try:
        # Request specific fields to optimize data transfer
        dataset_response = warehouse._client.get("/evolution_graph/dataset", params={
            "required_fields": ["node_id", "avgQ", "num_vars", "width", "size", "wl_hash"]
        })
        if dataset_response.status_code == 200:
            dataset = dataset_response.json()
            nodes = dataset.get("nodes", [])
            edges = dataset.get("edges", [])
            print(f"✓ Retrieved {len(nodes)} nodes and {len(edges)} edges from warehouse")
            if len(nodes) == 0:
                print("  Note: No nodes found - trajectories may still be processing")
        else:
            print(f"⚠ Failed to retrieve graph dataset: HTTP {dataset_response.status_code}")
            print(f"  Response: {dataset_response.text[:200] if dataset_response.text else 'No response body'}")
            nodes = []
            edges = []
    except Exception as e:
        print(f"⚠ Error retrieving graph dataset: {e}")
        nodes = []
        edges = []
    
    # Build NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes with their attributes
    for node in nodes:
        G.add_node(
            node['node_id'],
            avgQ=node.get('avgQ', 0),
            size=node.get('size', 0),
            width=node.get('width', 0),
            num_vars=node.get('num_vars', 0),
            wl_hash=node.get('wl_hash', '')
        )
    
    # Add edges using the new format (src, dst, type)
    for edge in edges:
        if 'src' in edge and 'dst' in edge:
            G.add_edge(edge['src'], edge['dst'], edge_type=edge.get('type', 'EVOLVED_TO'))
    
    print(f"✓ Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G


def retrieve_detailed_node_info(warehouse: WarehouseClient) -> List[Dict[str, Any]]:
    """Retrieve detailed information for all nodes including definitions."""
    print("\n=== Retrieving Detailed Node Information ===")
    
    # Get all node data with complete fields
    try:
        dataset_response = warehouse._client.get("/evolution_graph/dataset", params={
            "required_fields": ["node_id", "avgQ", "num_vars", "width", "size", "wl_hash", "timestamp", "traj_id", "traj_slice"]
        })
        if dataset_response.status_code == 200:
            dataset = dataset_response.json()
            nodes = dataset.get("nodes", [])
            print(f"✓ Retrieved detailed info for {len(nodes)} nodes")
        else:
            print(f"⚠ Failed to retrieve detailed node data: HTTP {dataset_response.status_code}")
            return []
    except Exception as e:
        print(f"⚠ Error retrieving detailed node data: {e}")
        return []
    
    # Enrich nodes with formula definitions
    detailed_nodes = []
    for node in nodes:
        node_id = node.get('node_id')
        if node_id:
            try:
                # Get formula definition for this node
                def_response = warehouse._client.get("/formula/definition", params={"node_id": node_id})
                if def_response.status_code == 200:
                    def_data = def_response.json()
                    node['definition'] = def_data.get('definition', [])
                else:
                    node['definition'] = []
                    print(f"  ⚠ No definition found for node {node_id}")
            except Exception as e:
                print(f"  ⚠ Error getting definition for {node_id}: {e}")
                node['definition'] = []
        
        detailed_nodes.append(node)
    
    print(f"✓ Enriched {len(detailed_nodes)} nodes with formula definitions")
    return detailed_nodes


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
        # Handle None values
        if avgQ is None:
            avgQ = 0
        if size is None:
            size = 1
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
    in_degrees = [d for _, d in G.in_degree()]
    if in_degrees:
        plt.hist(in_degrees, bins=20, color='blue', alpha=0.7, edgecolor='black')
    plt.title('In-Degree Distribution')
    plt.xlabel('In-Degree')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    
    # Out-degree distribution
    plt.subplot(1, 2, 2)
    out_degrees = [d for _, d in G.out_degree()]
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


def convert_gate_definition_to_string(gate_int: int) -> str:
    """Convert integer gate representation to human-readable string using Literals."""
    try:
        # Extract positive and negative literals from the 64-bit integer
        # First 32 bits are positive literals, last 32 bits are negative literals
        pos = gate_int & ((1 << 32) - 1)  # Extract lower 32 bits
        neg = (gate_int >> 32) & ((1 << 32) - 1)  # Extract upper 32 bits
        
        # Create Literals object and convert to string
        literals = Literals(pos=pos, neg=neg)
        formula_str = str(literals)
        
        # Replace negation symbol ¬ with ~ for better display compatibility
        formula_str = formula_str.replace('¬', '~')
        
        return formula_str
    except Exception as e:
        return f"[Error converting {gate_int}: {e}]"


def format_definition_as_strings(definition: List[int]) -> List[str]:
    """Convert a list of integer gate definitions to human-readable strings."""
    if not definition:
        return []
    
    return [convert_gate_definition_to_string(gate) for gate in definition]


def generate_detailed_node_report(detailed_nodes: List[Dict[str, Any]], output_dir: Path):
    """Generate detailed report with all node information including definitions."""
    print("\n=== Generating Detailed Node Report ===")
    
    if not detailed_nodes:
        print("⚠ No nodes to report")
        return
    
    # Save detailed node report
    node_report_path = output_dir / "detailed_node_report.txt"
    with open(node_report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED NODE INFORMATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total nodes: {len(detailed_nodes)}\n\n")
        
        # Sort nodes by avgQ (best first)
        sorted_nodes = sorted(detailed_nodes, key=lambda x: x.get('avgQ') or 0, reverse=True)
        
        for i, node in enumerate(sorted_nodes, 1):
            f.write(f"NODE {i:02d}: {node.get('node_id', 'Unknown')}\n")
            f.write("-" * 50 + "\n")
            
            # Basic attributes
            f.write(f"Average Q-value: {node.get('avgQ', 'N/A')}\n")
            f.write(f"Number of variables: {node.get('num_vars', 'N/A')}\n")
            f.write(f"Width: {node.get('width', 'N/A')}\n")
            f.write(f"Size: {node.get('size', 'N/A')}\n")
            f.write(f"WL Hash: {node.get('wl_hash', 'N/A')}\n")
            f.write(f"Timestamp: {node.get('timestamp', 'N/A')}\n")
            f.write(f"Trajectory ID: {node.get('traj_id', 'N/A')}\n")
            f.write(f"Trajectory Slice: {node.get('traj_slice', 'N/A')}\n")
            
            # Formula definition (convert integers to readable strings)
            definition = node.get('definition', [])
            if definition:
                f.write(f"Formula Definition (raw): {definition}\n")
                string_definition = format_definition_as_strings(definition)
                f.write(f"Formula Definition (readable): {string_definition}\n")
                f.write(f"Definition Length: {len(definition)} gates\n")
            else:
                f.write("Formula Definition: [Empty or not available]\n")
            
            f.write("\n")
    
    # Also create a JSON version for programmatic access
    # Enrich the nodes with string representations for JSON export
    enriched_nodes = []
    for node in sorted_nodes:
        enriched_node = node.copy()
        definition = node.get('definition', [])
        if definition:
            enriched_node['definition_strings'] = format_definition_as_strings(definition)
        else:
            enriched_node['definition_strings'] = []
        enriched_nodes.append(enriched_node)
    
    json_report_path = output_dir / "detailed_node_report.json"
    with open(json_report_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_nodes": len(detailed_nodes),
            "nodes": enriched_nodes
        }, f, indent=2, default=str)
    
    print(f"✓ Detailed node report saved to: {node_report_path}")
    print(f"✓ JSON node report saved to: {json_report_path}")
    
    # Print summary statistics
    print("\nNode Summary Statistics:")
    print(f"  - Total nodes analyzed: {len(detailed_nodes)}")
    
    # Count nodes with definitions
    nodes_with_def = sum(1 for n in detailed_nodes if n.get('definition'))
    print(f"  - Nodes with definitions: {nodes_with_def}")
    
    # avgQ statistics
    avg_qs = [n.get('avgQ') for n in detailed_nodes if n.get('avgQ') is not None]
    if avg_qs:
        print(f"  - Best avgQ: {max(avg_qs):.4f}")
        print(f"  - Average avgQ: {np.mean(avg_qs):.4f}")
        print(f"  - Worst avgQ: {min(avg_qs):.4f}")
    
    # Size statistics
    sizes = [n.get('size') for n in detailed_nodes if n.get('size') is not None]
    if sizes:
        print(f"  - Largest formula size: {max(sizes)}")
        print(f"  - Average formula size: {np.mean(sizes):.2f}")
        print(f"  - Smallest formula size: {min(sizes)}")


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
            if avgQ is None:
                avgQ = 0
            if avgQ > best_avgQ:
                best_avgQ = avgQ
                best_node = node
        
        stats["best_node"] = best_node
        stats["best_avgQ"] = best_avgQ
        
        # Calculate average metrics
        avg_sizes = [G.nodes[n].get('size', 0) or 0 for n in G.nodes()]
        avg_Qs = [G.nodes[n].get('avgQ', 0) or 0 for n in G.nodes()]
        
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


def list_trajectories_with_analysis(warehouse: WarehouseClient, output_dir: Path):
    """Generate detailed trajectory analysis showing literal/complexity sequences and evolution paths."""
    print("\n=== Trajectory Analysis and Listing ===")
    
    # Retrieve all trajectories and nodes from warehouse
    try:
        # Get trajectory dataset
        traj_response = warehouse._client.get("/trajectory/dataset")
        if traj_response.status_code == 200:
            traj_dataset = traj_response.json()
            trajectories = traj_dataset.get("trajectories", [])
            print(f"✓ Retrieved {len(trajectories)} trajectories")
        else:
            print(f"⚠ Failed to retrieve trajectories: HTTP {traj_response.status_code}")
            return
        
        # Get evolution graph dataset with trajectory linkage
        node_response = warehouse._client.get("/evolution_graph/dataset", params={
            "required_fields": ["node_id", "avgQ", "num_vars", "width", "size", "wl_hash", "traj_id", "traj_slice"]
        })
        if node_response.status_code == 200:
            node_dataset = node_response.json()
            nodes = node_dataset.get("nodes", [])
            print(f"✓ Retrieved {len(nodes)} evolution graph nodes")
        else:
            print(f"⚠ Failed to retrieve nodes: HTTP {node_response.status_code}")
            nodes = []
        
    except Exception as e:
        print(f"⚠ Error retrieving data: {e}")
        return
    
    if not trajectories:
        print("⚠ No trajectories found to analyze")
        return
    
    # Group nodes by trajectory ID and sort by traj_slice
    traj_to_nodes = defaultdict(list)
    for node in nodes:
        traj_id = node.get('traj_id')
        if traj_id:
            traj_to_nodes[traj_id].append(node)
    
    # Sort nodes within each trajectory by traj_slice
    for traj_id in traj_to_nodes:
        traj_to_nodes[traj_id].sort(key=lambda x: x.get('traj_slice', 0))
    
    # Analyze trajectories
    analysis_results = []
    
    for i, trajectory in enumerate(trajectories, 1):
        traj_id = trajectory.get("traj_id", f"traj_{i}")
        steps = trajectory.get("steps", [])
        
        print(f"Processing trajectory {i}/{len(trajectories)}: {traj_id}")
        
        # Extract literal sequences and avgQ from trajectory steps
        literal_sequence = []
        avgQ_sequence = []
        
        for step in steps:
            # Step format: [token_type, token_literals, cur_avgQ]
            if len(step) >= 3:
                token_type = step[0]
                token_literals = step[1]
                cur_avgQ = step[2]
                
                # Convert token type to human-readable string
                token_type_str = "ADD" if token_type == 0 else "DEL"
                
                # Convert token_literals to human-readable string using Literals tool
                if isinstance(token_literals, int) and token_literals != 0:
                    # Convert integer gate representation to string
                    try:
                        literals_str = convert_gate_definition_to_string(token_literals)
                    except Exception:
                        literals_str = f"[Error: {token_literals}]"
                elif isinstance(token_literals, list) and token_literals:
                    # Handle list of literals (convert each and join)
                    try:
                        literal_strs = [convert_gate_definition_to_string(lit) for lit in token_literals]
                        literals_str = ".".join(literal_strs)
                    except Exception:
                        literals_str = f"[Error: {token_literals}]"
                else:
                    literals_str = "[Empty]"
                
                # Combine token type with literals
                formatted_literal = f"{token_type_str} {literals_str}"
                literal_sequence.append(formatted_literal)
                
                # Store avgQ values (avgQ represents complexity)
                avgQ_sequence.append(cur_avgQ)
            else:
                # Handle malformed steps gracefully
                literal_sequence.append("INVALID")
                avgQ_sequence.append(0)
        
        # Build evolution path from linked nodes
        evolution_path = []
        linked_nodes = traj_to_nodes.get(traj_id, [])
        
        for node in linked_nodes:
            evolution_path.append({
                "node_id": node.get("node_id"),
                "traj_slice": node.get("traj_slice", 0),
                "avgQ": node.get("avgQ", 0)
            })
        
        # Compile trajectory analysis
        traj_analysis = {
            "trajectory_id": traj_id,
            "sequence_number": i,
            "total_steps": len(steps),
            "literal_sequence": literal_sequence,
            "avgQ_sequence": avgQ_sequence,
            "evolution_path": evolution_path,
            "final_avgQ": avgQ_sequence[-1] if avgQ_sequence else 0,
            "avgQ_improvement": (avgQ_sequence[-1] - avgQ_sequence[0]) if len(avgQ_sequence) > 1 else 0,
            "linked_nodes_count": len(linked_nodes)
        }
        
        analysis_results.append(traj_analysis)
    
    # Generate comprehensive report
    generate_trajectory_analysis_report(analysis_results, output_dir)
    
    return analysis_results


def generate_trajectory_analysis_report(analysis_results: List[Dict], output_dir: Path):
    """Generate comprehensive trajectory analysis report."""
    print("\n=== Generating Trajectory Analysis Report ===")
    
    if not analysis_results:
        print("⚠ No trajectory analysis results to report")
        return
    
    # Save detailed text report
    report_path = output_dir / "trajectory_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE TRAJECTORY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Total trajectories analyzed: {len(analysis_results)}\n\n")
        
        # Summary statistics
        total_steps = sum(r["total_steps"] for r in analysis_results)
        total_nodes = sum(r["linked_nodes_count"] for r in analysis_results)
        avg_final_avgQ = sum(r["final_avgQ"] for r in analysis_results) / len(analysis_results)
        
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 50 + "\n")
        f.write(f"Total steps across all trajectories: {total_steps}\n")
        f.write(f"Total evolution graph nodes created: {total_nodes}\n")
        f.write(f"Average final avgQ: {avg_final_avgQ:.4f}\n\n")
        
        # Sort trajectories by final avgQ (best first)
        sorted_results = sorted(analysis_results, key=lambda x: x["final_avgQ"], reverse=True)
        
        # Detailed trajectory analysis
        for result in sorted_results:
            f.write("=" * 60 + "\n")
            f.write(f"TRAJECTORY {result['sequence_number']}: {result['trajectory_id']}\n")
            f.write("=" * 60 + "\n")
            
            f.write("TRAJECTORY OVERVIEW\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total steps: {result['total_steps']}\n")
            f.write(f"Final avgQ: {result['final_avgQ']:.4f}\n")
            f.write(f"avgQ improvement: {result['avgQ_improvement']:+.4f}\n")
            f.write(f"Linked evolution nodes: {result['linked_nodes_count']}\n\n")
            
            # Literal sequence analysis
            f.write("LITERAL SEQUENCE (First 10 steps)\n")
            f.write("-" * 30 + "\n")
            for i, literals in enumerate(result["literal_sequence"][:10]):
                f.write(f"Step {i+1:2d}: {literals}\n")
            if len(result["literal_sequence"]) > 10:
                f.write(f"... ({len(result['literal_sequence']) - 10} more steps)\n")
            f.write("\n")
            
            # avgQ progression
            f.write("AVGQ SEQUENCE (Every 4th step)\n")
            f.write("-" * 30 + "\n")
            avgQ_seq = result["avgQ_sequence"]
            for i in range(0, len(avgQ_seq), 4):
                step_num = i + 1
                avgQ_val = avgQ_seq[i]
                f.write(f"Step {step_num:2d}: {avgQ_val:8.4f}")
                if i + 4 < len(avgQ_seq):
                    f.write(f"    Step {i+5:2d}: {avgQ_seq[i+4]:8.4f}")
                f.write("\n")
            f.write("\n")
            
            # Evolution path
            f.write("EVOLUTION PATH\n")
            f.write("-" * 30 + "\n")
            if result["evolution_path"]:
                f.write("(node_id, traj_slice, avgQ)\n")
                for path_entry in result["evolution_path"]:
                    node_id = path_entry["node_id"]
                    traj_slice = path_entry["traj_slice"]
                    avgQ = path_entry["avgQ"]
                    f.write(f"({node_id}, {traj_slice}, {avgQ:.4f})\n")
            else:
                f.write("No evolution path nodes found for this trajectory\n")
            f.write("\n\n")
    
    # Save JSON report for programmatic access
    json_report_path = output_dir / "trajectory_analysis_report.json"
    with open(json_report_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "total_trajectories": len(analysis_results),
            "summary_statistics": {
                "total_steps": sum(r["total_steps"] for r in analysis_results),
                "total_nodes": sum(r["linked_nodes_count"] for r in analysis_results),
                "avg_final_avgQ": sum(r["final_avgQ"] for r in analysis_results) / len(analysis_results)
            },
            "trajectories": analysis_results
        }, f, indent=2, default=str)
    
    print(f"✓ Trajectory analysis report saved to: {report_path}")
    print(f"✓ JSON analysis report saved to: {json_report_path}")
    
    # Print summary to console
    print("\nTrajectory Analysis Summary:")
    print(f"  - Total trajectories analyzed: {len(analysis_results)}")
    print(f"  - Best final avgQ: {max(r['final_avgQ'] for r in analysis_results):.4f}")
    print(f"  - Average final avgQ: {sum(r['final_avgQ'] for r in analysis_results) / len(analysis_results):.4f}")
    print(f"  - Total evolution path nodes: {sum(r['linked_nodes_count'] for r in analysis_results)}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="V2 System Demonstration and Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_v2_system.py                    # Run full demo with early stopping (default)
  python demo_v2_system.py --no-early-stop    # Run demo without early stopping
  python demo_v2_system.py --list-trajectories # Run demo + trajectory analysis
  python demo_v2_system.py --only-list-trajectories # Only trajectory analysis
        """
    )
    
    parser.add_argument(
        "--list-trajectories",
        action="store_true",
        help="Generate detailed trajectory analysis after running the demo"
    )
    
    parser.add_argument(
        "--only-list-trajectories", 
        action="store_true",
        help="Only run trajectory analysis (skip demo, requires existing data)"
    )
    
    parser.add_argument(
        "--early-stop",
        action="store_true",
        default=True,
        help="Enable early stopping when avgQ reaches 0 (default: True)"
    )
    
    parser.add_argument(
        "--no-early-stop",
        action="store_false",
        dest="early_stop",
        help="Disable early stopping (run full trajectory length)"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_arguments()
    
    print("=" * 60)
    if args.only_list_trajectories:
        print("V2 TRAJECTORY ANALYSIS")
    else:
        print("V2 SYSTEM DEMONSTRATION")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}\n")
    
    try:
        # 1. Create output directory
        output_dir = create_output_directory()
        
        # 2. Initialize services
        if args.only_list_trajectories:
            # Only initialize warehouse for trajectory analysis
            warehouse = WarehouseClient('localhost', 8000)
            print("✓ Connected to Warehouse service")
            clusterbomb = None
        else:
            clusterbomb, warehouse = initialize_services()
        
        if not args.only_list_trajectories:
            # 3. Create empty initial formula
            initial_node_id = create_empty_formula(warehouse)
            
            # 4. Execute weapon rollout
            execute_weapon_rollout(clusterbomb, initial_node_id, args.early_stop)
            
            # 5. Retrieve trajectory data
            trajectory_stats = retrieve_trajectories(warehouse)
            
            # 6. Retrieve and build evolution graph
            G = retrieve_evolution_graph(warehouse)
            
            # 7. Create visualizations
            visualize_graph(G, output_dir)
            
            # 8. Retrieve detailed node information
            detailed_nodes = retrieve_detailed_node_info(warehouse)
            
            # 9. Generate detailed node report
            generate_detailed_node_report(detailed_nodes, output_dir)
            
            # 10. Generate summary report
            generate_summary_report(G, trajectory_stats, output_dir)
        
        # 11. Trajectory analysis (if requested)
        if args.list_trajectories or args.only_list_trajectories:
            list_trajectories_with_analysis(warehouse, output_dir)
        
        print("\n" + "=" * 60)
        if args.only_list_trajectories:
            print("TRAJECTORY ANALYSIS COMPLETE")
        else:
            print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print(f"Finished at: {datetime.now().isoformat()}")
        
    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    finally:
        # Clean up connections
        if 'clusterbomb' in locals() and clusterbomb:
            clusterbomb.close()
        if 'warehouse' in locals():
            warehouse._client.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())