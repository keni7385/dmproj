import networkx as nx
import random


class RandomClustering:

    def __init__(self, seed=-1):
        self.seed = seed
        self.name = "RandomClustering"

    def run(self, graph: nx.Graph, k: int):
        """
        Perform a random partitioning of the graph based on the seed given in the constructor
        :param graph: graph to be partitioned
        :param k: number of clusters
        :return: partitioned graph
        """
        if self.seed > 0:
            random.seed(self.seed)
        for n in graph.nodes:
            graph.add_node(n, partition=random.randint(0, k-1))
        return graph
