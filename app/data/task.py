import networkx as nx
from app.evaluation.goodness import balanced_partition
import logging
import os


class GraphPartitioningTask:

    def __init__(self, name: str, k: int, graph: nx.Graph, offset: int):
        """
        Construct a graph partitioning task
        :param name: The name of the task, or the name of the graph
        :param k: the number of communities
        :param graph: the target graph
        """
        self.name = name
        self.k = k
        self.graph = graph
        self.offset = offset

    def solve(self, algorithm, objective_function=balanced_partition, output_directory="results",
              output_directory_info="info", save_results=True, save_info=True, curr_smallest_value=None,
              normalised=True):
        """
        Runs the given algorithm, evaluates the outcomes, stores and returns the results
        :param algorithm: the clustering algorithm to run
        :param objective_function: function to evaluate the outcome
        :param output_directory: path in which the results are stored
        :param output_directory_info: path in which the logs are stored
        :param save_results: flag defining whether the results should be stored or not
        :param save_info: flag defining whether the logs should be stored or not
        :param curr_smallest_value: best score found so far; threshold for storing the results
        :param normalised: indicates whether the Laplacian matrix is normalized or not; used for logging only
        :return: the minimum between curr_smallest_value and the new score
        """

        logging.info("[Starting] to solve {} for k={} clusters with algorithm {}, offset={}, random_state={}, normalised={}".format(
                     self.name, self.k, algorithm.name, self.offset, os.environ["random_state"], normalised))
        data = "[Starting] to solve {} for k={} clusters with algorithm {} and offset={}, random_state={}, normalised={}\n".format(
            self.name, self.k, algorithm.name, self.offset, os.environ["random_state"], normalised)

        partitioned = algorithm.run(self.graph, self.k)

        logging.info("[Done] {} returned.".format(algorithm.name))
        data = data + "[Done] {} returned.\n".format(algorithm.name)

        logging.info("[Starting] clustering evaluation")
        score = objective_function(partitioned, self.k)
        logging.info("[Finished] Score obtained = {}".format(score))

        data = data + "[Finished] Score obtained = {}\n\n".format(score)

        if save_results and (curr_smallest_value > score or curr_smallest_value is None):
            self._save_results(output_directory, partitioned)
        else:
            print("\n")

        if save_info and (curr_smallest_value > score or curr_smallest_value is None):
            self._save_info(output_directory_info, data)

        if curr_smallest_value > score:
            return score
        else:
            return curr_smallest_value

    def _save_results(self, output_directory, partitions):
        """
        Saves the partition file into a file
        :param output_directory: path in which the results are stored
        :param partitions: nx.Graph() containing the nodes with the partitioned labels
        """
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
        filename = output_directory + os.sep + self.name + ".output"
        with open(filename, 'w') as f:
            f.write("# %s %d %d %d\n" %
                    (self.name, self.graph.number_of_nodes(), self.graph.number_of_edges(), self.k))
            for node, data in partitions.nodes(data=True):
                f.write("%s %d\n" % (node, data.get('partition')))
        logging.info("[Output Done] results on file %s" % filename)

    def _save_info(self, output_directory_info, data):
        """
        Saves the logs of the results
        :param output_directory_info: path in which the logs are stored
        :param data: content of the logs
        """
        if not os.path.exists(output_directory_info):
            os.makedirs(output_directory_info)
        filename = output_directory_info + os.sep + self.name + ".txt"
        with open(filename, 'a+') as f:
            f.write(data)
        logging.info("[Output Done] info in file {}\n".format(filename))
