import networkx as nx
from scipy.sparse import linalg
from sklearn.cluster import KMeans


class SpectralClusteringBis:
    def __init__(self):
        self.name = "SpectralLuca"

    def run(self, net: nx.Graph, k: int):
        L = nx.laplacian_matrix(net)
        L = L.asfptype()
        vals, vecs = linalg.eigs(L, k=k+5, which='SR')
        embedding = vecs.real

        kmeans = KMeans(n_clusters=k, random_state=1, max_iter=500).fit(embedding)
        pred_k = kmeans.predict(embedding)

        partitioned = nx.Graph()
        for index, node in enumerate(net.nodes):
            partitioned.add_node(node, partition=pred_k[index])
        partitioned.add_edges_from(net.edges)

        return partitioned
