import networkx as nx
from typing import Dict, List


def compute_metrics(graph: nx.Graph) -> Dict[str, float]:
    """Compute quality metrics for a graph.

    Parameters
    ----------
    graph : nx.Graph
        The graph to analyze.

    Returns
    -------
    Dict[str, float]
        Dictionary containing metrics like average degree, number of connected
        components, component sizes and the average shortest path length of the
        largest component.
    """
    metrics: Dict[str, float] = {}
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        return {
            "average_degree": 0.0,
            "num_components": 0.0,
            "largest_component_size": 0.0,
            "average_shortest_path_length": 0.0,
        }

    degrees = [deg for _, deg in graph.degree()]
    metrics["average_degree"] = sum(degrees) / float(num_nodes)

    component_sizes: List[int] = [len(c) for c in nx.connected_components(graph)]
    metrics["num_components"] = float(len(component_sizes))
    metrics["largest_component_size"] = float(max(component_sizes))

    largest_component = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_component)
    if subgraph.number_of_nodes() > 1:
        metrics["average_shortest_path_length"] = nx.average_shortest_path_length(subgraph)
    else:
        metrics["average_shortest_path_length"] = 0.0

    # attach the full list of component sizes for completeness
    metrics["component_sizes"] = component_sizes  # type: ignore
    return metrics
