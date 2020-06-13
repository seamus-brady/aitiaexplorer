import os

import pandas as pd
import pygraphviz
from networkx.drawing import nx_agraph
from pycausal import search as s
from pycausal.pycausal import pycausal
from tests.unit import TestAPI

from aitia_explorer.py_causal_wrapper import PyCausalWrapper


class Test_PyCausalWrapper(TestAPI):
    """
    TBD
    """
    wrapper = PyCausalWrapper()

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'resources/data')

    def tearDown(self):
        pass

    ############### BayesEst ##################
    def test_algo_bayes_est(self):
        data_dir = os.path.join(self.data_dir, "sim_discrete_data_20vars_100cases.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_bayes_est(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### FCI ##################
    def test_fci_bayes_est(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_fci(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### PC ##################
    def test_pc(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_pc(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### FGES ##################
    def test_algo_fges_continuous(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_fges_continuous(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_fges_discrete(self):
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_fges_discrete(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_fges_mixed(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_fges_mixed(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### GFCI ##################
    def test_algo_gfci_continuous(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_gfci_continuous(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_gfci_discrete(self):
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_gfci_discrete(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_gfci_mixed(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_gfci_mixed(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### RFCI ##################
    def test_algo_rfci_continuous(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_rfci_continuous(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_rfci_discrete(self):
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_rfci_discrete(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_rfci_mixed(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_rfci_mixed(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_multiple_algo_run(self):
        dot_str_list = []

        pc = pycausal()
        pc.start_vm()
        tetrad = s.tetradrunner()

        data_dir = os.path.join(self.data_dir, "sim_discrete_data_20vars_100cases.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str_list.append(self.wrapper.algo_bayes_est(df, pc))

        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str_list.append(self.wrapper.algo_fci(df, pc))

        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str_list.append(self.wrapper.algo_pc(df, pc))

        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str_list.append(self.wrapper.algo_fges_continuous(df, pc))

        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str_list.append(self.wrapper.algo_fges_discrete(df, pc))

        pc.stop_vm()

        self.assertTrue(len(dot_str_list) == 5)

    def test_run_all_algorithms(self):
        pc = pycausal()
        pc.start_vm()
        dot_str_list = []
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        for algo in self.wrapper.get_all_algorithms():
            dot_str_list.append(algo(df, pc))
        pc.stop_vm()
        self.assertTrue(len(dot_str_list) == 12)

    def test_dot_graph_load(self):
        # get the graph
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_fges_discrete(df)

        graph = nx_agraph.from_agraph(pygraphviz.AGraph(dot_str))
        self.assertTrue(graph is not None, "Nx did not load graph data.")


    def test_causal_graph_load(self):
        # get the graph
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.wrapper.algo_fges_discrete(df)

        causal_graph = self.wrapper.get_causal_graph_from_dot(dot_str)

        self.assertTrue(causal_graph is not None, "CausalGraphicalModel did not load graph data.")
