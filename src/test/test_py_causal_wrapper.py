import os

import pandas as pd
import pygraphviz
from networkx.drawing import nx_agraph
from pycausal import search as s
from pycausal.pycausal import pycausal
from tests.unit import TestAPI

from aitia_explorer.algorithm_runner import AlgorithmRunner


class Test_PyCausalWrapper(TestAPI):
    """
    Tests for the pycausal wrapper starting and stopping the Java VM.
    """
    pc_util = AlgorithmRunner()

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'resources/data')

    def tearDown(self):
        pass

    def test_multiple_algo_run(self):
        dot_str_list = []

        pc = pycausal()
        pc.start_vm()
        tetrad = s.tetradrunner()

        data_dir = os.path.join(self.data_dir, "sim_discrete_data_20vars_100cases.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str_list.append(self.pc_util.algo_bayes_est(df, pc))

        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str_list.append(self.pc_util.algo_fci(df, pc))

        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str_list.append(self.pc_util.algo_pc(df, pc))

        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str_list.append(self.pc_util.algo_fges_continuous(df, pc))

        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str_list.append(self.pc_util.algo_fges_discrete(df, pc))

        pc.stop_vm()

        self.assertTrue(len(dot_str_list) == 5)

    def test_run_all_algorithms(self):
        pc = pycausal()
        pc.start_vm()
        dot_str_list = []
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        for algo in self.pc_util.get_all_causal_algorithms():
            algo = algo[1] # just need the func
            dot_str_list.append(algo(df, pc))
        pc.stop_vm()
        self.assertTrue(len(dot_str_list) == 12)

    def test_dot_graph_load(self):
        # get the graph
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_fges_discrete(df)

        graph = nx_agraph.from_agraph(pygraphviz.AGraph(dot_str))
        self.assertTrue(graph is not None, "Nx did not load graph data.")


    def test_causal_graph_load(self):
        # get the graph
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_fges_discrete(df)

        causal_graph = self.pc_util.get_causal_graph_from_dot(dot_str)

        self.assertTrue(causal_graph is not None, "CausalGraphicalModel did not load graph data.")
