import networkx as nx
import re
from app.data.task import GraphPartitioningTask


class Reader:

    @staticmethod
    def read(filename):
        with open(filename) as f:
            header = f.readline()

        header_parse = re.search('# (.*) [0-9]+ [0-9]+ ([0-9]+)', header)

        if not header_parse:
            raise SyntaxError("File %s not correctly formatted" % filename)

        name = header_parse.group(1)
        k = int(header_parse.group(2))
        graph = nx.read_edgelist(filename)
        return GraphPartitioningTask(name, k, graph)
