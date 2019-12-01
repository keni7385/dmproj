import networkx as nx
import random


class RandomClustering:

    def __init__(self, seed=-1):
        self.seed = seed
        self.name = "RandomClustering"

    def run(self, graph: nx.Graph, k: int):
        if self.seed > 0:
            random.seed(self.seed)
        for n in graph.nodes:
            graph.add_node(n, partition=random.randint(0, k-1))
        return graph
