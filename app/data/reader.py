import os
import pickle
import re
import logging

import networkx as nx

from app.algorithms.spectral_clustering import compute_eigenvectors


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

        return {"name": name, "k": k, "graph": graph}

    @staticmethod
    def load_embedding(output_dir, task_params, max_offset, negative_offset, normalised=True, directed=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        num_eig_vec = task_params["k"] + max_offset - negative_offset
        if num_eig_vec < 1:
            num_eig_vec = 1
        eigvec_file = "%s_size_%d_%s.eigvec" % (task_params["name"], num_eig_vec, "norm" if normalised else "not_norm")
        eigvec_file = output_dir + os.sep + eigvec_file
        file_exists = os.path.exists(eigvec_file) and os.path.isfile(eigvec_file)

        if file_exists:
            logging.info("[Loading] eigen vectors from %s", eigvec_file)
            with open(eigvec_file, 'rb') as f:
                embedding = pickle.load(f)
        else:
            logging.info("[Computing] eigen vectors, eigen vector file not found.")
            embedding = compute_eigenvectors(task_params["graph"], task_params["k"] + max_offset - negative_offset,
                                             normalised=normalised, directed=directed)
            with open(eigvec_file, 'wb') as f:
                pickle.dump(embedding, f)
            logging.info("[Writing] eigen vectors to %s", eigvec_file)

        return embedding
