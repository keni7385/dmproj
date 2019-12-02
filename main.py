from app.algorithms.random_clustering import RandomClustering
from app.algorithms.spectral_clustering import SpectralClustering, compute_eigenvectors
from app.algorithms.fifty_kmeans import BalancedSpectralClustering
from app.data.reader import Reader
from app.data.task import GraphPartitioningTask

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

files = ["ca-GrQc", "Oregon-1", "roadNet-CA", "soc-Epinions1", "web-NotreDame"]
path = "graphs_processed/%s.txt"
paths = [path % file for file in files[2:]]
max_offset = 24
negative_offset = 4

for filepath in paths:
    print("Started {}".format(filepath))
    task_params = Reader.read(filepath)  # name, k, graph
    # embedding = compute_eigenvectors(task_params[2], task_params[1] + max_offset)
    embedding = compute_eigenvectors(task_params[2], task_params[1] + max_offset - 4, normalised=True)

    for offset in range(max_offset):
        task = GraphPartitioningTask(*task_params, offset - negative_offset)
        task.solve(SpectralClustering(embedding[:, :task_params[1] + offset - negative_offset]))
