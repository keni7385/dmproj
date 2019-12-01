import networkx as nx
from scipy.sparse.linalg import eigs
from sklearn.cluster import KMeans
import numpy as np


class SpectralClustering:

    def __init__(self, normalised=True):
        self.normalised = normalised
        self.name = "SpectralClustering"

    def run(self, graph: nx.Graph, k: int):
        laplacian = nx.laplacian_matrix(graph) if not self.normalised else nx.normalized_laplacian_matrix(graph)
        laplacian = laplacian.asfptype()

        # TODO ensure to have at least k eigenvectors, eigsh only reports 6
        vals, vecs = eigs(laplacian, k=k+1, which='SR')
        x = vecs.real  # eig vect of the second smallest eigenvalue

        emb_nodes = [x[int(node), :] for node in graph.nodes]  # nodes in the embedded space
        emb_nodes = np.array(emb_nodes).reshape(-1, k)

        kmeans = KMeans(n_clusters=k, random_state=1).fit(emb_nodes)
        predictions = kmeans.predict(emb_nodes)

        partitioned = nx.Graph()
        for node in graph.nodes:
            partitioned.add_node(node, partition=predictions[int(node)])
        partitioned.add_edges_from(graph.edges)
        return partitioned
