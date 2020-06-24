import os

from tests.unit import TestAPI

from aitia_explorer.metrics.graph_metrics import GraphMetrics
from aitia_explorer.algorithm_runner import AlgorithmRunner
from aitia_explorer.util.graph_util import GraphUtil
from aitia_explorer.target_data.loader import TargetData


class Test_Metrics(TestAPI):
    """
    Tests for graph metrics.
    """
    pc_util = AlgorithmRunner()

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'resources/data')

    def tearDown(self):
        pass

    def test_retrieve_adjacency_matrix(self):
        dot_str = TargetData.simulated_data_1_graph()
        graph = GraphUtil.get_digraph_from_dot(dot_str)
        metrics = GraphMetrics()
        adj_matrix = metrics.retrieve_adjacency_matrix(graph)
        self.assertTrue(adj_matrix is not None, "No adjacency matrix created.")

    def test_precision_recall(self):
        # get the simulated data
        simulated_data = TargetData.simulated_data_1()
        dot_str = self.pc_util.algo_pc(simulated_data)
        pred_graph = GraphUtil.get_digraph_from_dot(dot_str)

        # get the known data
        dot_str = TargetData.simulated_data_1_graph()
        target_graph =  GraphUtil.get_digraph_from_dot(dot_str)
        metrics = GraphMetrics()
        prec_recall = metrics.precision_recall(target_graph, pred_graph)

        self.assertTrue(prec_recall[0] == 0.41250000000000003)

    def test_shd(self):
        # get the simulated data
        simulated_data = TargetData.simulated_data_1()
        dot_str = self.pc_util.algo_pc(simulated_data)
        pred_graph = GraphUtil.get_digraph_from_dot(dot_str)

        # get the known data
        dot_str = TargetData.simulated_data_1_graph()
        target_graph = GraphUtil.get_digraph_from_dot(dot_str)
        metrics = GraphMetrics()
        shd = metrics.SHD(target_graph, pred_graph)

        self.assertTrue(shd == 10)
