import os
import pickle
import re
import logging
import networkx as nx
from app.algorithms.spectral_clustering import compute_eigenvectors, compute_manifold_eigenvector


class Reader:

    @staticmethod
    def read(filename):
        """
        Loads the graph from file
        :param filename: path of the file to load
        :return: dictionary containing the name of graph loaded, the number of clusters and the graph itself
        """
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
    def load_embedding(output_dir, task_params, max_offset, negative_offset, normalised=True, manifold_method=False):
        """
        Static method that loads the embedding if stored, computes and stores it otherwise
        :param output_dir: path of the directory where to store or load the embedding from
        :param task_params: parameters needed to identify the file through its name or to store it with the right name
        :param max_offset: maximum offset applied to the number of eigenvectors (=k+max+offset-1+negative_offset)
        :param negative_offset: negative offset applied to the number of eigenvectors (=k+max+offset-1+negative_offset)
        :param normalised: flag defining whether the Laplacian matrix is/should be normalised or not
        :param manifold_method: flag defining whether the manifold method is/should be used or not
        :return: embedding whose columns are the eigenvectors
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        num_eig_vec = task_params["k"] + max_offset - negative_offset
        if num_eig_vec < 1:
            num_eig_vec = 1
        eigvec_file = "%s_size_%d_%s%s.eigvec" % (task_params["name"], num_eig_vec,
                                                  "norm" if normalised else "not_norm",
                                                  "_manifold" if manifold_method else "")
        eigvec_file = output_dir + os.sep + eigvec_file
        file_exists = os.path.exists(eigvec_file) and os.path.isfile(eigvec_file)

        if file_exists:
            logging.info("[Loading] eigen vectors from %s", eigvec_file)
            with open(eigvec_file, 'rb') as f:
                embedding = pickle.load(f)
        else:
            logging.info("[Computing] eigen vectors, eigen vector file not found.%s",
                         " Using manifold embedding." if manifold_method else "")
            compute_function = compute_eigenvectors if not manifold_method else compute_manifold_eigenvector
            embedding = compute_function(task_params["graph"], task_params["k"] + max_offset - negative_offset,
                                         normalised=normalised)
            with open(eigvec_file, 'wb') as f:
                pickle.dump(embedding, f)
            logging.info("[Writing] eigen vectors to %s", eigvec_file)

        return embedding
