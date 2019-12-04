import networkx as nx
from sklearn.cluster.spectral import SpectralClustering
import os


class FasterSpectralClustering:

    def __init__(self, offset):
        self.name = "FasterSpectralClustering"
        self.offset = offset

    def run(self, graph: nx.Graph, k: int):
        pred_k = SpectralClustering(n_clusters=k,
                                    eigen_solver="amg",
                                    random_state=int(os.environ["random_state"]),
                                    n_components=self.offset+k,
                                    affinity="precomputed",
                                    n_jobs=-1).fit_predict(nx.adjacency_matrix(graph))
        print("Done, partitioning now...\n")

        partitioned = nx.Graph()
        for index, node in enumerate(graph.nodes):
            partitioned.add_node(node, partition=pred_k[index])
        partitioned.add_edges_from(graph.edges)

        return partitioned
