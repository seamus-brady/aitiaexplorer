import os

import pandas as pd
from pycausal.pycausal import pycausal
from tests.unit import TestAPI

from aitia_explorer.app import App
from aitia_explorer.feature_reduction.bayesian_gaussian_mixture_wrapper import BayesianGaussianMixtureWrapper
from aitia_explorer.feature_selection_runner import FeatureSelectionRunner
from aitia_explorer.target_data.loader import TargetData


class Test_App(TestAPI):
    """
    Tests for the aitia_explorer app.
    """
    bgmm = BayesianGaussianMixtureWrapper()

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

    def test_run_analysis(self):
        pc = pycausal()
        pc.start_vm()
        aitia = App()
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        # just need a test graph
        dot_str = aitia.algo_runner.algo_pc(df, pc)
        # algo list
        algorithm_list = []
        algorithm_list.append(aitia.algo_runner.PC)
        algorithm_list.append(aitia.algo_runner.FCI)
        analysis_results = aitia._run_analysis(df,
                                               algorithm_list=algorithm_list,
                                               target_graph_str=dot_str,
                                               pc=pc)
        self.assertTrue(analysis_results is not None)

    def test_all_returned_features(self):
        feature_set = set()
        hepart_data = TargetData.hepar2_100_data()
        feature_set.update(FeatureSelectionRunner.random_forest_feature_reduction(hepart_data, 10))
        feature_set.update(FeatureSelectionRunner.pfa_feature_reduction(hepart_data, 10))
        feature_set.update(FeatureSelectionRunner.linear_regression_feature_reduction(hepart_data, 10))
        feature_set.update(FeatureSelectionRunner.xgboost_feature_reduction(hepart_data, 10))
        feature_set.update(FeatureSelectionRunner.rfe_feature_reduction(hepart_data, 10))
        df_reduced = self.bgmm.get_reduced_dataframe(hepart_data, list(feature_set))
        self.assertTrue(df_reduced is not None)
