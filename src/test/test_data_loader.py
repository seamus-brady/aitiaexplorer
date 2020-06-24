import os

from tests.unit import TestAPI

from aitia_explorer.target_data.loader import TargetData
from aitia_explorer.util.graph_util import GraphUtil


class Test_Data_Loader(TestAPI):
    """
    Tests for scm and data generation.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_load_simulated_data_graph(self):
        simulated_data = TargetData.simulated_data_1()
        self.assertTrue(simulated_data is not None, "No simulated data loaded.")

        dot_str = TargetData.simulated_data_1_graph()
        self.assertTrue(dot_str is not None, "No simulated graph loaded.")

    def test_create_known_graph(self):
        dot_str = TargetData.simulated_data_1_graph()
        graph = GraphUtil.get_digraph_from_dot(dot_str)
        self.assertTrue(graph.edges() is not None, "No known simulated graph created.")
        self.assertTrue(graph.nodes() is not None, "No known simulated graph created.")

    def test_scm_generation(self):
        scm1 = TargetData.scm1()
        scm2 = TargetData.scm2()
        scm3 = TargetData.scm2()
        self.assertTrue(scm1 is not None)
        self.assertTrue(scm2 is not None)
        self.assertTrue(scm3 is not None)

    def test_random_scm(self):
        random_scm = TargetData.random_scm()
        self.assertTrue(random_scm is not None)