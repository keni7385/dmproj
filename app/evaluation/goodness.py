import networkx as nx
from typing import Dict
from tqdm import tqdm


def balanced_partition(part_graph: nx.Graph, k: int):
    score = 0
    partitions = get_partitions(part_graph, k)

    for ith_cluster, part in partitions.items():
        v_i = part.nodes
        v_i_bar = part_graph.nodes - v_i
        number_of_edges = 0
        for node_v_i in tqdm(v_i, desc="Cluster %d-th of %d: " % (ith_cluster, k)):
            for node_v_i_bar in v_i_bar:
                number_of_edges = number_of_edges + \
                                  part_graph.number_of_edges(node_v_i, node_v_i_bar)
        score = score + number_of_edges / len(v_i)

    return score


def get_partitions(graph: nx.Graph, k: int) -> Dict[int, nx.Graph]:
    """
    Returns an array of sub-graphs. In order to filter the nodes, the attribute 'partition' is considered,
    and it should contain the node cluster ID. The graph is already partitioned through the attributes
    :param graph: the partitioned graph
    :param k: the number of clusters
    :return: the dictionary of sub-graph. i => sub-graph of community i-th
    """
    partitions = {}
    for i in range(0, k):
        nodes = (node
                 for node, data in graph.nodes(data=True)
                 if data.get('partition') == i)
        partitions[i] = graph.subgraph(nodes)
    return partitions
