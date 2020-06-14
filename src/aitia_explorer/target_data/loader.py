"""
TBD
"""
import logging
import os
import networkx as nx
import pandas as pd

from aitia_explorer.target_data.graphs.simulated_data_graph import SimulatedData1Graph

_logger = logging.getLogger(__name__)


class TargetData:
    """
    A class that provides known causal data for targetting in tests.
    """

    def __init__(self):
        pass

    @staticmethod
    def data_dir():
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        return data_dir

    @staticmethod
    def simulated_data_1_graph():
        graph = nx.Graph()
        graph.add_edges_from(SimulatedData1Graph.edges())
        graph.add_nodes_from(SimulatedData1Graph.nodes())
        graph = nx.DiGraph(graph)
        return graph

    @staticmethod
    def simulated_data_1():
        data_dir = os.path.join(TargetData.data_dir(), "simulated_data_1.txt")
        simulated_data = pd.read_table(data_dir, sep="\t")
        return simulated_data