import networkx as nx
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
import numpy as np


class SpectralClustering:

    def __init__(self):
        self.name = "SpectralClustering"

    def run(self, graph: nx.Graph, k: int):
        norm_laplacian = nx.normalized_laplacian_matrix(graph)
        vals, vecs = eigsh(norm_laplacian, which='SM')
        x = vecs[:, 0:k]  # eig vect of the second smallest eigenvalue

        emb_nodes = [x[int(node), :] for node in graph.nodes]  # nodes in the embedded space
        emb_nodes = np.array(emb_nodes).reshape(-1, k)

        kmeans = KMeans(n_clusters=k, random_state=1).fit(emb_nodes)
        predictions = kmeans.predict(emb_nodes)

        partitioned = nx.Graph()
        for node in graph.nodes:
            partitioned.add_node(node, partition=predictions[int(node)])
        partitioned.add_edges_from(graph.edges)
        return partitioned
