import networkx as nx

class EvolutionGraphManager:
    """
    Manages the evolution graph for trajectory processing.
    """
    def __init__(self, **config):
        self.config = config
        self.evograph = nx.DiGraph()
    
    def update_graph(self, new_formulas: list[dict], evo_path: list[str]):
        """
        Updates the evolution graph with a new trajectory.
        """
        self.evograph.add_nodes_from({
            (
                formula['id'],
                {
                    'id': formula['id'],
                    'avgQ': formula['avgQ'],
                    'size': formula['size'],
                    'visited_counter': 0,
                }
            )
            for formula in new_formulas if formula['id'] not in self.evograph
        })
        
        if evo_path[0] is None:
            # If the first node is None, we skip adding edges
            evo_path = evo_path[1:]
        if len(evo_path) <= 1:
            return
        
        self.evograph.add_edges_from(
            (
                (evo_path[i - 1], evo_path[i])
                for i in range(1, len(evo_path))
            )
        )
        involved_nodes = set(evo_path)
        nodes = self.evograph.nodes(data=True)
        
        # increment the visited counter for involved nodes
        for node_id in involved_nodes:
            nodes[node_id]['visited_counter'] += 1

    def get_active_nodes(self) -> list[dict]:
        """
        Returns a list of active nodes in the evolution graph.
        """
        # This is a stub implementation.
        # In a real implementation, you would return the active nodes from the graph.
        return [
            {
                'id': node,
                'in_degree': self.evograph.in_degree(node),
                'out_degree': self.evograph.out_degree(node),
                **data,
            }
            for node, data in self.evograph.nodes(data=True)
        ]