"""
TBD Header
"""
import logging

import networkx as nx
import pygraphviz
from causalgraphicalmodels import CausalGraphicalModel

_logger = logging.getLogger(__name__)


class GraphUtil():
    """
    Graph utility class.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_dot_from_nxgraph(nx_graph):
        """
        Create a dot string from an nxgraph
        """
        return nx.nx_pydot.to_pydot(nx_graph)


    @staticmethod
    def get_nxgraph_from_adjacency_matrix(adjacency_matrix):
        """
        Create an nxgraph from an adjacency_matrix (numpy array).
        """
        nx_graph = nx.DiGraph(nx.from_numpy_matrix(adjacency_matrix))
        return nx_graph

    @staticmethod
    def get_nxgraph_from_dot(dot_str):
        """
        Create an nxgraph from a dot string.
        """
        nx_graph = nx.drawing.nx_agraph.from_agraph(pygraphviz.AGraph(dot_str))
        return nx_graph

    @staticmethod
    def get_digraph_from_dot(dot_str):
        """
        Loads a DiGraph from a dot string.
        """

        # load graph from dot data
        nx_graph = GraphUtil.get_nxgraph_from_dot(dot_str)
        nx_graph = nx.DiGraph(nx_graph)
        return nx_graph

    @staticmethod
    def get_causal_graph_from_dot(dot_str):
        """
        Create a CausalGraphicalModel from a dot string.
        """

        # load graph from dot data
        nx_graph = GraphUtil.get_nxgraph_from_dot(dot_str)

        # create a causal graph
        return GraphUtil.get_causal_graph_from_nxgraph(nx_graph)

    @staticmethod
    def get_causal_graph_from_nxgraph(nx_graph):
        """
        Create a CausalGraphicalModel from an nxgraph.
        """
        # create a causal graph
        causal_graph = CausalGraphicalModel(
            nodes=nx_graph.nodes(),
            edges=nx_graph.edges()
        )

        return causal_graph

    @staticmethod
    def get_causal_graph_with_latent_edges(nx_graph, latent_edges):
        """
        Create a CausalGraphicalModel from an nxgraph with unobserved latent edges.
        """
        # create a causal graph
        causal_graph = CausalGraphicalModel(
            nodes=nx_graph.nodes(),
            edges=nx_graph.edges(),
            latent_edges=latent_edges
        )

        return causal_graph

    @staticmethod
    def get_causal_graph_from_bif(bif_reader):
        """
        Create a CausalGraphicalModel from an bif file.
        """
        # create a causal graph
        causal_graph = CausalGraphicalModel(
            nodes=bif_reader.get_variables(),
            edges=bif_reader.get_edges()
        )

        return causal_graph
