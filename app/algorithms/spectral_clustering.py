import networkx as nx
from scipy.sparse import linalg
from sklearn.cluster import KMeans
import os

class SpectralClustering:

    def __init__(self, embedding):
        self.name = "SpectralClustering"
        self.embedding = embedding

    def run(self, graph: nx.Graph, k: int):

        kmeans = KMeans(n_clusters=k, random_state=int(os.environ["random_state"]), max_iter=500).fit(self.embedding)
        pred_k = kmeans.predict(self.embedding)

        partitioned = nx.Graph()
        for index, node in enumerate(graph.nodes):
            partitioned.add_node(node, partition=pred_k[index])
        partitioned.add_edges_from(graph.edges)

        return partitioned


def compute_eigenvectors(graph: nx.Graph,  num: int, normalised: bool = False):
    """
    Compute the real part of the eigenvecors of an undirected Graph
    :param graph: target graph
    :param normalised: define whether the laplacian matrix should be normalised or not
    :param num: number of eigenvectors to compute
    :return: eigenvectors relative to the target graph
    """
    if normalised:
        laplacian = nx.normalized_laplacian_matrix(graph)
    else:
        laplacian = nx.laplacian_matrix(graph)
        laplacian = laplacian.asfptype()

    _, vecs = linalg.eigsh(laplacian, k=num, which='SM')

    return vecs.real
