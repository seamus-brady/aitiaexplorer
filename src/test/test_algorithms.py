import os

import pandas as pd
from tests.unit import TestAPI

from aitia_explorer.algorithm_runner import AlgorithmRunner


class Test_Algorithms(TestAPI):
    """
    Tests for individual causal discovery algorithms.
    Please note these tests have to be run individually as the Java VM wrapper in PyCausal
    does not like repeated runs! But hopefully they are still useful.
    """

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'resources/data')

    def tearDown(self):
        pass

    ############### NOTEARS ##################

    def test_algo_notears(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        adj_matrix = AlgorithmRunner.algo_notears(df)
        self.assertTrue(adj_matrix is not None, "No graph returned.")

    ############### BayesEst ##################
    def test_algo_bayes_est(self):
        data_dir = os.path.join(self.data_dir, "sim_discrete_data_20vars_100cases.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_bayes_est(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### FCI ##################
    def test_fci(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_fci(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### PC ##################
    def test_pc(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_pc(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### FGES ##################
    def test_algo_fges_continuous(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_fges_continuous(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_fges_discrete(self):
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_fges_discrete(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_fges_mixed(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_fges_mixed(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### GFCI ##################
    def test_algo_gfci_continuous(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_gfci_continuous(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_gfci_discrete(self):
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_gfci_discrete(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_gfci_mixed(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_gfci_mixed(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### RFCI ##################
    def test_algo_rfci_continuous(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_rfci_continuous(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_rfci_discrete(self):
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_rfci_discrete(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_rfci_mixed(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_rfci_mixed(df)
        self.assertTrue(dot_str is not None, "No graph returned.")