from app.algorithms.random_clustering import RandomClustering
from app.algorithms.spectral_clustering import SpectralClustering
from app.algorithms.fifty_kmeans import BalancedSpectralClustering
from app.data.reader import Reader
from app.data.task import GraphPartitioningTask
import os
from numpy import inf

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

files = ["ca-GrQc", "Oregon-1", "soc-Epinions1", "web-NotreDame", "roadNet-CA"]
path = "graphs_processed/%s.txt"
output_directory = "results"
paths = [path % file for file in files[0:2]]
max_offset = 30
negative_offset = 4

for filepath in paths:
    print("Started {}".format(filepath))
    task_params = Reader.read(filepath)
    curr_smallest_value = inf
    for normalised in [True, False]:
        embedding = Reader.load_embedding(output_directory, task_params, max_offset, negative_offset,
                                          normalised=normalised)
        for random_state in range(10):
            # Try different random states for the k-means algorithm
            os.environ["random_state"] = str(random_state)

            for offset in range(max_offset):
                task_params["offset"] = offset - negative_offset
                task = GraphPartitioningTask(**task_params)
                upper_bound = task_params["k"] + offset - negative_offset
                if upper_bound == 1:
                    # Use the second smallest eigenvec
                    curr_smallest_value = task.solve(SpectralClustering(embedding[:, 2].reshape(-1, 1)),
                                                     curr_smallest_value=curr_smallest_value, normalised=normalised)
                elif upper_bound > 1:
                    # Use the first k eigenvecs
                    curr_smallest_value = task.solve(SpectralClustering(embedding[:, :upper_bound]),
                                                     curr_smallest_value=curr_smallest_value, normalised=normalised)
