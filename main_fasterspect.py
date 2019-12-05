import logging
import os

from numpy import inf

from app.algorithms.faster_spectral_custering import FasterSpectralClustering
from app.data.reader import Reader
from app.data.task import GraphPartitioningTask

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

files = ["ca-GrQc", "Oregon-1", "soc-Epinions1", "web-NotreDame", "roadNet-CA"]
path = "graphs_processed/%s.txt"
output_directory = "results"
paths = [path % file for file in files]
max_offset = 10
negative_offset = 0

for filepath in paths[3:5]:
    print("Started {}".format(filepath))
    task_params = Reader.read(filepath)
    curr_smallest_value = inf
    for random_state in range(5):
        # Try different random states for the k-means algorithm
        os.environ["random_state"] = str(random_state)

        for offset in range(max_offset):
            task_params["offset"] = offset - negative_offset
            task = GraphPartitioningTask(**task_params)
            curr_smallest_value = task.solve(FasterSpectralClustering(offset - negative_offset),
                                             curr_smallest_value=curr_smallest_value, normalised=True)
