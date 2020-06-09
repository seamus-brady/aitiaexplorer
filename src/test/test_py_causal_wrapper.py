import os

from tests.unit import TestAPI
import pandas as pd
from app.py_causal_wrapper import PyCausalWrapper


class Test_PyCausalWrapper(TestAPI):
    """
    TBD
    """
    wrapper = PyCausalWrapper()

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'resources/data')

    def tearDown(self):
        pass

    def test_get_tetrad(self):
        self.wrapper.start_vm()
        tetrad = self.wrapper.get_tetrad()
        self.wrapper.stop_vm()
        self.assertTrue(tetrad is not None)


    def test_get_algos(self):
        self.wrapper.start_vm()
        causal_discovery_algos = []
        for algo in self.wrapper.get_causal_discovery_algos():
            causal_discovery_algos.append(algo)
        self.wrapper.stop_vm()
        self.assertTrue(len(causal_discovery_algos) == 25)


    def test_get_algos(self):
        self.wrapper.start_vm()
        self.wrapper.dump_all_algo_desc()
        self.wrapper.stop_vm()
        self.assertTrue(True)

    def test_algo_bayes_est(self):
        data_dir = os.path.join(self.data_dir, "sim_discrete_data_20vars_100cases.txt")
        df = pd.read_table(data_dir, sep="\t")
        self.wrapper.start_vm()
        dot_str = self.wrapper.algo_bayes_est(df)
        self.wrapper.stop_vm()
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_fges_continuous(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        self.wrapper.start_vm()
        dot_str = self.wrapper.algo_fges_continuous(df)
        self.wrapper.stop_vm()
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_fges_discrete(self):
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        self.wrapper.start_vm()
        dot_str = self.wrapper.algo_fges_discrete(df)
        self.wrapper.stop_vm()
        self.assertTrue(dot_str is not None, "No graph returned.")