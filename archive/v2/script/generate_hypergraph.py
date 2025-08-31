#!/usr/bin/env python3
"""
Generate Evolution Graph Dataset in GraphML Format

This script preprocesses an evolution graph by directly connecting to Neo4j database:
1. Downloads nodes and edges directly from Neo4j database
2. Contracts nodes connected by SAME_Q edges into hypernodes
3. Creates NGE_Q directed edges between hypernodes based on avgQ comparisons
4. Exports the result to GraphML format

Requirements:
- All nodes in a connected component induced by SAME_Q edges are contracted into a hypernode
- Nodes connected by SAME_Q edges share the same avgQ value and form a hypernode
- Isolated nodes (no SAME_Q edges) form their own hypernode
- Two hypernodes are connected by NGE_Q edge if:
  - Their constituent nodes have EVOLVED_TO edges between them
  - The source hypernode has lower avgQ than target hypernode
- Only NGE_Q edges appear in the result (no SAME_Q or EVOLVED_TO)
- Each hypernode contains avgQ value and list of contracted node IDs

Usage:
    # Using environment variables for credentials
    export NEO4J_URI=bolt://localhost:7687
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=password
    python generate_hypergraph.py --output hypergraph.graphml
    
    # Using command line arguments
    python generate_hypergraph.py --output hypergraph.graphml --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password password
    
    # Test mode with default credentials
    LONGSHOT_TEST_MODE=1 python generate_hypergraph.py --output hypergraph.graphml
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any
from neo4j import GraphDatabase
import networkx as nx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnionFind:
    """Efficient Union-Find (Disjoint Set) data structure for finding connected components."""
    
    def __init__(self, elements: Set[str]):
        """Initialize Union-Find with a set of elements."""
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}
    
    def find(self, x: str) -> str:
        """Find root of element x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: str, y: str) -> bool:
        """Union two elements. Returns True if they were in different components."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return False
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        
        return True
    
    def get_components(self) -> Dict[str, List[str]]:
        """Get all connected components as {root: [members]}."""
        components = defaultdict(list)
        for elem in self.parent:
            root = self.find(elem)
            components[root].append(elem)
        return dict(components)


class HypergraphProcessor:
    """Main processor for converting evolution graph to hypernode graph."""
    
    def __init__(self, neo4j_uri: str = None, neo4j_user: str = None, neo4j_password: str = None):
        """Initialize processor with Neo4j connection parameters."""
        # Get Neo4j connection details from environment or parameters
        self.neo4j_uri = neo4j_uri or os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = neo4j_user or os.getenv('NEO4J_USER', 'neo4j')
        self.neo4j_password = neo4j_password or os.getenv('NEO4J_PASSWORD', 'password')
        
        # Test mode for backwards compatibility
        if os.getenv('LONGSHOT_TEST_MODE') == '1':
            self.neo4j_uri = 'neo4j://neo4j-bread:7687'
            self.neo4j_user = 'neo4j'
            self.neo4j_password = 'bread861122'
        
        self.driver = None
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.same_q_edges: List[Tuple[str, str]] = []
        self.evolved_to_edges: List[Tuple[str, str]] = []
        self.hypernodes: Dict[str, Dict[str, Any]] = {}
        self.nge_q_edges: Set[Tuple[str, str]] = set()
        self.skipped_nodes_count = 0  # Track nodes skipped due to missing avgQ
    
    def connect_to_neo4j(self) -> None:
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            logger.info(f"Connected to Neo4j at {self.neo4j_uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    def close_connection(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.debug("Closed Neo4j connection")
    
    def download_graph_data(self, required_fields: List[str]) -> None:
        """Download nodes and edges directly from Neo4j database."""
        logger.info("Downloading graph data from Neo4j database...")
        
        # Ensure required fields always include node_id and avgQ
        if "node_id" not in required_fields:
            required_fields.append("node_id")
        if "avgQ" not in required_fields:
            required_fields.append("avgQ")
        
        try:
            with self.driver.session() as session:
                # Query to get all nodes with selected fields
                node_fields = []
                for field in required_fields:
                    if field in ["node_id", "avgQ", "num_vars", "width", "size", "wl_hash", "traj_id", "traj_slice"]:
                        node_fields.append(f"n.{field} AS {field}")
                
                nodes_query = f"""
                MATCH (n:FormulaNode)
                RETURN {', '.join(node_fields)}
                """
                
                # Execute nodes query
                result = session.run(nodes_query)
                for record in result:
                    node_data = dict(record)
                    node_id = node_data["node_id"]
                    self.nodes[node_id] = node_data
                
                logger.info(f"Downloaded {len(self.nodes)} nodes")
                
                # Query to get all edges
                edges_query = """
                MATCH (n:FormulaNode)-[r]->(m:FormulaNode)
                RETURN n.node_id AS src, m.node_id AS dst, type(r) AS edge_type
                """
                
                # Execute edges query
                result = session.run(edges_query)
                for record in result:
                    src = record["src"]
                    dst = record["dst"]
                    edge_type = record["edge_type"]
                    
                    if edge_type == "SAME_Q":
                        self.same_q_edges.append((src, dst))
                    elif edge_type == "EVOLVED_TO":
                        self.evolved_to_edges.append((src, dst))
                
                logger.info(f"Downloaded {len(self.same_q_edges)} SAME_Q edges")
                logger.info(f"Downloaded {len(self.evolved_to_edges)} EVOLVED_TO edges")
                
        except Exception as e:
            logger.error(f"Failed to download graph data: {e}")
            raise
    
    def create_hypernodes(self) -> None:
        """Create hypernodes by contracting SAME_Q connected components."""
        logger.info("Creating hypernodes from SAME_Q connected components...")
        
        # Initialize Union-Find with all nodes
        all_nodes = set(self.nodes.keys())
        uf = UnionFind(all_nodes)
        
        # Union nodes connected by SAME_Q edges
        for src, dst in self.same_q_edges:
            if src in all_nodes and dst in all_nodes:
                uf.union(src, dst)
        
        # Get connected components
        components = uf.get_components()
        
        # Create hypernodes from components
        hypernode_id = 0
        self.node_to_hypernode = {}  # Mapping from original node to hypernode
        
        for root, members in components.items():
            hypernode_id += 1
            h_id = f"hn{hypernode_id}"
            
            # Get avgQ from the first member (all should have same avgQ)
            first_member = members[0]
            avgQ = self.nodes[first_member].get("avgQ")
            
            # Skip nodes without avgQ values
            if avgQ is None:
                logger.warning(f"Skipping hypernode {h_id}: no avgQ value for nodes {members}")
                self.skipped_nodes_count += len(members)  # Count all members as skipped
                continue
            
            # Verify all members have same avgQ (sanity check)
            for member in members:
                member_avgQ = self.nodes[member].get("avgQ")
                if member_avgQ is None:
                    logger.warning(f"Node {member} in hypernode {h_id} has no avgQ value")
                    continue
                if abs(member_avgQ - avgQ) > 1e-9:
                    logger.warning(
                        f"avgQ mismatch in hypernode {h_id}: "
                        f"node {member} has avgQ={member_avgQ}, expected {avgQ}"
                    )
            
            # Create hypernode
            self.hypernodes[h_id] = {
                "id": h_id,
                "avgQ": avgQ,
                "members": sorted(members)  # Sort for consistent output
            }
            
            # Update node to hypernode mapping
            for member in members:
                self.node_to_hypernode[member] = h_id
        
        logger.info(f"Created {len(self.hypernodes)} hypernodes from {len(self.nodes)} nodes")
        if self.skipped_nodes_count > 0:
            logger.info(f"Skipped {self.skipped_nodes_count} nodes due to missing avgQ values")
        
        # Log statistics
        sizes = [len(h["members"]) for h in self.hypernodes.values()]
        if sizes:
            logger.info(f"Hypernode sizes: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.2f}")
    
    def create_nge_q_edges(self) -> None:
        """Create NGE_Q edges between hypernodes based on EVOLVED_TO relationships."""
        logger.info("Creating NGE_Q edges between hypernodes...")
        
        edge_candidates = defaultdict(set)  # Track potential edges to avoid duplicates
        
        # Process each EVOLVED_TO edge
        for src_node, dst_node in self.evolved_to_edges:
            # Skip if nodes not in our dataset
            if src_node not in self.node_to_hypernode or dst_node not in self.node_to_hypernode:
                continue
            
            src_hypernode = self.node_to_hypernode[src_node]
            dst_hypernode = self.node_to_hypernode[dst_node]
            
            # Skip self-loops
            if src_hypernode == dst_hypernode:
                continue
            
            # Record this edge candidate
            edge_candidates[(src_hypernode, dst_hypernode)].add((src_node, dst_node))
        
        # Create NGE_Q edges based on avgQ comparison
        for (src_h, dst_h), original_edges in edge_candidates.items():
            # Skip if either hypernode doesn't exist (could happen if nodes were skipped due to missing avgQ)
            if src_h not in self.hypernodes or dst_h not in self.hypernodes:
                continue
                
            src_avgQ = self.hypernodes[src_h]["avgQ"]
            dst_avgQ = self.hypernodes[dst_h]["avgQ"]
            
            # Create NGE_Q edge from lower avgQ to higher avgQ hypernode
            # regardless of the original EVOLVED_TO direction
            if src_avgQ < dst_avgQ:
                # Source has lower avgQ, create edge as-is
                self.nge_q_edges.add((src_h, dst_h))
            elif src_avgQ > dst_avgQ:
                # Source has higher avgQ, reverse the edge direction
                self.nge_q_edges.add((dst_h, src_h))
            # Note: If avgQ values are equal, no edge is created (shouldn't happen after SAME_Q contraction)
        
        logger.info(f"Created {len(self.nge_q_edges)} NGE_Q edges")
        
        # Log edge statistics
        out_degrees = defaultdict(int)
        in_degrees = defaultdict(int)
        for src, dst in self.nge_q_edges:
            out_degrees[src] += 1
            in_degrees[dst] += 1
        
        if out_degrees:
            out_vals = list(out_degrees.values())
            logger.info(f"Out-degrees: min={min(out_vals)}, max={max(out_vals)}, avg={sum(out_vals)/len(out_vals):.2f}")
        if in_degrees:
            in_vals = list(in_degrees.values())
            logger.info(f"In-degrees: min={min(in_vals)}, max={max(in_vals)}, avg={sum(in_vals)/len(in_vals):.2f}")
    
    def export_to_graphml(self, output_file: str) -> None:
        """Export hypergraph to GraphML format using NetworkX."""
        logger.info(f"Exporting hypergraph to {output_file}...")
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add hypernodes with attributes
        for h_id, h_data in self.hypernodes.items():
            # Join member IDs with comma for GraphML storage
            members_str = ",".join(h_data["members"])
            G.add_node(h_id, avgQ=h_data["avgQ"], members=members_str)
        
        # Add NGE_Q edges
        for src, dst in self.nge_q_edges:
            G.add_edge(src, dst)
        
        # Write to GraphML
        nx.write_graphml(G, output_file, prettyprint=True)
        
        logger.info(f"Successfully exported hypergraph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    def process(self, output_file: str, required_fields: List[str]) -> None:
        """Execute complete processing pipeline."""
        logger.info("="*60)
        logger.info("HYPERGRAPH GENERATION PIPELINE")
        logger.info("="*60)
        
        try:
            # Connect to Neo4j
            self.connect_to_neo4j()
            
            # Step 1: Download data
            self.download_graph_data(required_fields)
            
            # Step 2: Create hypernodes
            self.create_hypernodes()
            
            # Step 3: Create NGE_Q edges
            self.create_nge_q_edges()
            
            # Step 4: Export to GraphML
            self.export_to_graphml(output_file)
            
            # Print summary
            print("\n" + "="*60)
            print("PROCESSING SUMMARY")
            print("="*60)
            print(f"Original nodes: {len(self.nodes)}")
            if self.skipped_nodes_count > 0:
                print(f"Skipped nodes (no avgQ): {self.skipped_nodes_count}")
                print(f"Processed nodes: {len(self.nodes) - self.skipped_nodes_count}")
            print(f"Original SAME_Q edges: {len(self.same_q_edges)}")
            print(f"Original EVOLVED_TO edges: {len(self.evolved_to_edges)}")
            print(f"Hypernodes created: {len(self.hypernodes)}")
            print(f"NGE_Q edges created: {len(self.nge_q_edges)}")
            if self.nodes:
                effective_nodes = len(self.nodes) - self.skipped_nodes_count
                if effective_nodes > 0:
                    print(f"Compression ratio: {len(self.hypernodes)/effective_nodes:.3f}")
            print(f"Output file: {output_file}")
            print("="*60)
            
        finally:
            # Always close connection
            self.close_connection()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate evolution graph dataset in GraphML format with hypernode preprocessing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings (uses environment variables for Neo4j)
  python generate_hypergraph.py --output hypergraph.graphml
  
  # Specify Neo4j connection details
  python generate_hypergraph.py --output hypergraph.graphml --neo4j-uri bolt://localhost:7687 --neo4j-user neo4j --neo4j-password mypassword
  
  # Include additional node fields in the download
  python generate_hypergraph.py --output hypergraph.graphml --fields avgQ num_vars width size wl_hash
  
  # Test mode with default test credentials
  LONGSHOT_TEST_MODE=1 python generate_hypergraph.py --output hypergraph.graphml
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output GraphML file path"
    )
    
    parser.add_argument(
        "--neo4j-uri",
        help="Neo4j database URI (default: from NEO4J_URI env or bolt://localhost:7687)"
    )
    
    parser.add_argument(
        "--neo4j-user",
        help="Neo4j username (default: from NEO4J_USER env or 'neo4j')"
    )
    
    parser.add_argument(
        "--neo4j-password",
        help="Neo4j password (default: from NEO4J_PASSWORD env or 'password')"
    )
    
    parser.add_argument(
        "--fields", "-f",
        nargs="+",
        default=["avgQ"],
        help="Required node fields to download (default: avgQ)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create processor and run
        processor = HypergraphProcessor(
            neo4j_uri=args.neo4j_uri,
            neo4j_user=args.neo4j_user,
            neo4j_password=args.neo4j_password
        )
        processor.process(output_file=args.output, required_fields=args.fields)
        
        logger.info("Hypergraph generation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())