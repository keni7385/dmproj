from app.algorithms.random_clustering import RandomClustering
from app.algorithms.spectral_clustering import SpectralClustering
from app.data.reader import Reader
from app.data.task import GraphPartitioningTask
import os
from numpy import inf
import networkx as nx

import logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

########################################################################################################
# Supplementary code
# Run the same algorithms specified in main.py, but firstly all the self loops will be removed from any node.
# There are no improvements in this scores, you can simply run main.py instead.
# This code will run the algorithm on every graph. This might be computational intensive, be careful.
# Different graphs have different parameters,
# make sure to change them accordingly to the Results section of the report.
########################################################################################################

files = ["ca-GrQc", "Oregon-1", "soc-Epinions1", "web-NotreDame", "roadNet-CA"]
path = "graphs_processed/%s.txt"
output_directory = "results"
paths = [path % file for file in files]

# Parameters:
drop_first_eigenvector = True  # Defines whether the first eigenvector should be dropped
min_random_seed = 0     # minimum of the range of k-means random seeds for grid search
max_random_seed = 4     # maximum of the range of k-means random seeds for grid search
normalised_vals = True  # sets whether Laplacian matrix should normalised. possible values: True, False, [True, False]
max_offset = 26         # contributes defining the set of eigenvecs to be used (=k+max_offset-1+negative_offset)
negative_offset = 0     # contributes defining the set of eigenvecs to be used (=k+max_offset-1+negative_offset)

for filepath in paths:
    print("Started {}".format(filepath))
    task_params = Reader.read(filepath)
    # Removing loops does not change for ca-GrQc
    graph_no_loops = task_params["graph"].copy()
    graph_no_loops.remove_edges_from(nx.selfloop_edges(task_params["graph"]))
    curr_smallest_value = inf  # sets the current smallest score to infinite

    if type(normalised_vals) is bool and type(normalised_vals) is not list:
        normalised_vals = [normalised_vals]

    for normalised in normalised_vals:
        eval_graph = task_params["graph"]  # save the graph read from file
        task_params["graph"] = graph_no_loops  # swap to compute the embedding
        embedding = Reader.load_embedding(output_directory, task_params, max_offset, negative_offset,
                                          normalised=normalised, manifold_method=False)
        # restore the original graph for evaluation
        task_params["graph"] = eval_graph

        if drop_first_eigenvector:
            embedding = embedding[:, 1:]

        for random_state in range(min_random_seed, max_random_seed+1):
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
