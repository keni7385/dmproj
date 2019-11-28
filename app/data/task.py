import networkx as nx
from app.evaluation.goodness import balanced_partition
import logging
import os


class GraphPartitioningTask:

    def __init__(self, name: str, k: int, graph: nx.Graph):
        """
        Construct a graph partitioning task
        :param name: The name of the task, or the name of the graph
        :param k: the number of communities
        :param graph: the target graph
        """
        self.name = name
        self.k = k
        self.graph = graph

    def solve(self, algorithm, objective_function=balanced_partition, output_directory="results", save_results=True):
        logging.info("[Starting] to solve {%s} for k=%d clusters with algorithm {%s}",
                     self.name, self.k, algorithm.name)
        partitioned = algorithm.run(self.graph, self.k)
        logging.info("[Done] {%s} returned.", algorithm.name)

        logging.info("[Starting] clustering evaluation")
        score = objective_function(partitioned, self.k)
        logging.info("[Finished] Score obtained = %f", score)

        if save_results:
            self._save_results(output_directory, partitioned)

    def _save_results(self, output_directory, partitions):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        filename = output_directory + os.sep + self.name + ".txt"
        with open(filename, 'w') as f:
            f.write("# %s %d %d %d\n" %
                    (self.name, self.graph.number_of_nodes(), self.graph.number_of_edges(), self.k))
            for node, data in partitions.nodes(data=True):
                f.write("%s %d\n" % (node, data.get('partition')))
        logging.info("[Output Done] results on file %s\n" % filename)
