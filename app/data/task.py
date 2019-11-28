import networkx as nx
from app.evaluation.goodness import balanced_partition


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

    def solve(self, algorithm, objective_function=balanced_partition):
        print("[Starting] to solve {%s} for k=%d clusters\n" % (self.name, self.k))

        partitioned = algorithm.run(self.graph, self.k)
        score = objective_function(partitioned, self.k)

        print("[Finished] Score obtained = %f \n" % score)
        # TODO: write results to output file
        print("[Output Done] results on file %s\n" % "fake.txt")
