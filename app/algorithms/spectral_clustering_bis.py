import networkx as nx
from scipy.sparse import linalg
from sklearn.cluster import KMeans


class SpectralClusteringBis:

    def __init__(self, normalised=True):
        self.normalised = normalised
        self.name = "SpectralBis"

    def run(self, graph: nx.Graph, k: int, offset: int = 0):
        if self.normalised:
            laplacian = nx.normalized_laplacian_matrix(graph)
        else:
            laplacian = nx.laplacian_matrix(graph)

        laplacian = laplacian.asfptype()

        vals, vecs = linalg.eigs(laplacian, k=k+offset, which='SR')
        embedding = vecs.real

        kmeans = KMeans(n_clusters=k, random_state=1, max_iter=500).fit(embedding)
        pred_k = kmeans.predict(embedding)

        partitioned = nx.Graph()
        for index, node in enumerate(graph.nodes):
            partitioned.add_node(node, partition=pred_k[index])
        partitioned.add_edges_from(graph.edges)

        return partitioned
