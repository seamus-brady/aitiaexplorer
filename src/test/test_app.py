import os

import pandas as pd
from pycausal.pycausal import pycausal

from tests.unit import TestAPI

from aitia_explorer.app import App


class Test_App(TestAPI):
    """
    Tests for the aitia_explorer app.
    """

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'resources/data')

    def tearDown(self):
        pass

    def test_scm_load(self):
        aitia = App()
        scm1 = aitia.data.scm1()
        target_graph_str = str(scm1.cgm.draw())
        df = scm1.sample(1000)
        self.assertTrue(target_graph_str is not None)
        self.assertTrue(df is not None)

    def test_run_analysis_one_algo(self):
        pc = pycausal()
        pc.start_vm()
        aitia = App()
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = aitia.algo_runner.algo_pc(df, pc)
        analysis_results, summary = aitia._run_analysis(df,
                                                        target_graph_str=dot_str,

                                                        pc=pc)
        self.assertTrue(summary is not None)
