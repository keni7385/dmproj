import networkx as nx
import re
from app.data.task import GraphPartitioningTask


class Reader:

    @staticmethod
    def read(filename):
        with open(filename) as f:
            header = f.readline()

        header_parse = re.search('# (.*) ([0-9]+) ([0-9]+) ([0-9]+)', header)

        if not header_parse:
            raise SyntaxError("File %s not correctly formatted" % filename)

        name = header_parse.group(1)
        num_of_vertices = int(header_parse.group(2))
        num_of_edges = int(header_parse.group(3))
        k = int(header_parse.group(4))

        graph = nx.read_edgelist(filename)
        if graph.number_of_nodes() != num_of_vertices or graph.number_of_edges() != num_of_edges:
            raise AssertionError("The input file stated %s nodes and %s edges, but %s nodes and %s edges are actually "
                                 "contained in the graph, please correct the input file." %
                                 (num_of_vertices, num_of_edges, graph.number_of_nodes(), graph.number_of_edges()))

        return GraphPartitioningTask(name, k, graph)
