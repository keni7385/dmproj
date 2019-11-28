import networkx as nx
import random


class RandomClustering:

    def __init__(self, seed):
        self.seed = seed
        self.name = "RandomClustering"

    def run(self, graph: nx.Graph, k: int):
        random.seed(self.seed)
        for n in graph.nodes:
            graph.add_node(n, partition=random.randint(0, k-1))
        return graph
