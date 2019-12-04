import networkx as nx
from sklearn.cluster import SpectralClustering


class FasterSpectralClustering:

    def __init__(self):
        self.name = "FasterSpectralClustering"

    def run(self, graph: nx.Graph, k: int):
        pred_k = SpectralClustering(n_clusters=k,
                                    eigen_solver="amg",
                                    random_state=1,
                                    affinity="precomputed",
                                    n_jobs=-1).fit_predict(nx.adjacency_matrix(graph))
        print("Done, partitioning now...\n")

        partitioned = nx.Graph()
        for index, node in enumerate(graph.nodes):
            partitioned.add_node(node, partition=pred_k[index])
        partitioned.add_edges_from(graph.edges)

        return partitioned
