import networkx as nx
from scipy.sparse import linalg
from scipy.sparse.linalg import LinearOperator
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding
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

    # from scipy.sparse import eye
    # from scipy.sparse.linalg import LinearOperator, eigsh, minres
    # sigma = 1
    # OP = laplacian - sigma * eye(laplacian.shape[0])
    # OPinv = LinearOperator(shape=laplacian.shape, dtype=laplacian.dtype, matvec=lambda v: minres(OP, v, tol=1e-3)[0])
    # w, v = eigsh(laplacian, sigma=sigma, k=num, which='SM', tol=1e-4, OPinv=OPinv)
    # return v

    _, vecs = linalg.eigsh(laplacian, k=num, which='SM')

    return vecs.real


def compute_manifold_eigenvector(graph: nx.Graph,  num: int, normalised: bool = False):
    embedding = spectral_embedding(nx.adjacency_matrix(graph), n_components=num,
                                   eigen_solver='amg',
                                   random_state=0,  # int(os.environ["random_state_embedding"]),
                                   eigen_tol=0.0, drop_first=False, norm_laplacian=normalised)
    return embedding
