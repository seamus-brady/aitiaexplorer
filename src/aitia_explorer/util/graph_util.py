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
    def get_nxgraph_from_dot(dot_str):
        """
        Create an nxgraph from a dot string.
        :param dot_str: dot string
        :return: CausalGraphicalModel
        """
        nx_graph = nx.drawing.nx_agraph.from_agraph(pygraphviz.AGraph(dot_str))
        return nx_graph

    @staticmethod
    def get_digraph_from_dot(dot_str):
        """
        Loads a DiGraph from a dor string.
        :param dot_str: dot string
        :return: DiGraph
        """

        # load graph from dot data
        nx_graph = GraphUtil.get_nxgraph_from_dot(dot_str)
        nx_graph = nx.DiGraph(nx_graph)
        return nx_graph

    @staticmethod
    def get_causal_graph_from_dot(dot_str):
        """
        Create a CausalGraphicalModel from a dot string.
        :param dot_str: dot string
        :return: CausalGraphicalModel
        """

        # load graph from dot data
        nx_graph = GraphUtil.get_nxgraph_from_dot(dot_str)

        # create a causal graph
        return GraphUtil.get_causal_graph_from_nxgraph(nx_graph)

    @staticmethod
    def get_causal_graph_from_nxgraph(nx_graph):
        """
        Create a CausalGraphicalModel from an nxgraph.
        :param dot_str: dot string
        :return: CausalGraphicalModel
        """
        # create a causal graph
        causal_graph = CausalGraphicalModel(
            nodes=nx_graph.nodes(),
            edges=nx_graph.edges()
        )

        return causal_graph
