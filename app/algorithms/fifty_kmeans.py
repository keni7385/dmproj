import networkx as nx
from scipy.sparse.linalg import eigs
from scipy.spatial import distance
import numpy as np
import random


class BalancedSpectralClustering:

    def __init__(self, normalised=True, seed=-1):
        self.normalised = normalised
        self.name = "SpectralClustering"
        self.seed = seed

    def run(self, graph: nx.Graph, k: int):
        if self.seed > 0:
            random.seed(self.seed)

        laplacian = nx.laplacian_matrix(graph) if not self.normalised else nx.normalized_laplacian_matrix(graph)
        laplacian = laplacian.asfptype()

        # TODO ensure to have at least k eigenvectors, eigsh only reports 6
        vals, vecs = eigs(laplacian, which='SM')
        x = vecs[:, 0:k].real  # eig vect of the second smallest eigenvalue

        size_of_clusters = graph.number_of_nodes() / k
        centroids = np.array(BalancedSpectralClustering.incremental_farthest_search(x, 4))
        clusters = [{'centroid_idx': idx,
                     'centroid_coord': centroid_point,
                     'cluster_pts_idx': [x.index(centroid_point.tolist())],
                     'visited_clusters': []
                     } for idx, centroid_point in enumerate(centroids)]

        dist = []
        centroids_indexes = [cluster['cluster_pts_idx'][0] for cluster in clusters]
        for i, point in enumerate(x):
            if i not in centroids_indexes:
                dist.append([])
                closest_centroid_distance = np.inf
                # closest_centroid_idx = -1
                closest_cluster_obj = {}
                for cluster in clusters:
                    # print(cluster["cluster_pts_idx"])
                    # print(size_of_clusters)
                    if len(cluster["cluster_pts_idx"]) < size_of_clusters:
                        # print("here")
                        euclidean_distance = distance.euclidean(point, cluster['centroid_coord'])
                        if euclidean_distance <= closest_centroid_distance:
                            closest_centroid_distance = euclidean_distance
                            # closest_centroid_idx = cluster['centroid_idx']
                            closest_cluster_obj = cluster

                if closest_cluster_obj:
                    closest_cluster_obj["cluster_pts_idx"].append(i)

        partitioned = nx.Graph()
        for cluster in clusters:
            partitioned.add_nodes_from(cluster['cluster_pts_idx'], partition=cluster['centroid_idx'])
        return partitioned

    @staticmethod
    def incremental_farthest_search(points, k):
        remaining_points = points.tolist()
        #remaining_points = points[:]
        solution_set = [remaining_points.pop(random.randint(0, len(remaining_points) - 1))]
        for _ in range(k - 1):
            distances = [BalancedSpectralClustering.distance(p, solution_set[0]) for p in remaining_points]
            for i, p in enumerate(remaining_points):
                for j, s in enumerate(solution_set):
                    distances[i] = min(distances[i], BalancedSpectralClustering.distance(p, s))
            solution_set.append(remaining_points.pop(distances.index(max(distances))))
        return solution_set

    @staticmethod
    def distance(a, b):
        a1 = np.array(a)
        b1 = np.array(b)
        dist = abs(np.linalg.norm(a1 - b1))
        return dist
