import os

import pandas as pd
from tests.unit import TestAPI

from aitia_explorer.py_causal_wrapper import PyCausalUtil


class Test_Algorithms(TestAPI):
    """
    Tests for individual algorithms.
    """
    pc_util = PyCausalUtil()


    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'resources/data')

    def tearDown(self):
        pass

    ############### BayesEst ##################
    def test_algo_bayes_est(self):
        data_dir = os.path.join(self.data_dir, "sim_discrete_data_20vars_100cases.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_bayes_est(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### FCI ##################
    def test_fci(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_fci(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### PC ##################
    def test_pc(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_pc(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### FGES ##################
    def test_algo_fges_continuous(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_fges_continuous(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_fges_discrete(self):
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_fges_discrete(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_fges_mixed(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_fges_mixed(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### GFCI ##################
    def test_algo_gfci_continuous(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_gfci_continuous(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_gfci_discrete(self):
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_gfci_discrete(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_gfci_mixed(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_gfci_mixed(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### RFCI ##################
    def test_algo_rfci_continuous(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_rfci_continuous(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_rfci_discrete(self):
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_rfci_discrete(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_rfci_mixed(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = self.pc_util.algo_rfci_mixed(df)
        self.assertTrue(dot_str is not None, "No graph returned.")