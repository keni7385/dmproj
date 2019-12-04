import os

import networkx as nx
from sklearn.cluster import KMeans
from sklearn.manifold import spectral_embedding


class UnNormalizedFasterSpectralClustering:

    def __init__(self, offset, random_seed_embedding=0, random_seed_clustering=1, assign_labels="kmeans"):
        self.random_seed_clustering = random_seed_clustering
        self.random_seed_embedding = random_seed_embedding
        self.name = "FasterSpectralClustering"
        self.offset = offset
        self.assign_labels = assign_labels

    def run(self, graph: nx.Graph, k: int):
        embedding = spectral_embedding(nx.adjacency_matrix(graph), n_components=k+self.offset,
                                       eigen_solver='amg',
                                       random_state=int(os.environ["random_state"]),
                                       eigen_tol=0.0, drop_first=False, norm_laplacian=False)

        # if self.assign_labels == 'kmeans':
        est = KMeans(n_clusters=k, n_init=10, max_iter=500,
                     random_state=0, copy_x=True).fit(embedding)
        pred_k = est.labels_
        # else:
        #    pred_k = discretize(maps, random_state=random_state)

        # SpectralClustering(n_clusters=k,
        #                            eigen_solver="amg",
        #                            random_state=int(os.environ["random_state"]),
        #                            n_components=self.offset+k,
        #                            affinity="precomputed",
        #                            n_jobs=-1).fit_predict(nx.adjacency_matrix(graph))
        print("Done, partitioning now...\n")

        partitioned = nx.Graph()
        for index, node in enumerate(graph.nodes):
            partitioned.add_node(node, partition=pred_k[index])
        partitioned.add_edges_from(graph.edges)

        return partitioned
