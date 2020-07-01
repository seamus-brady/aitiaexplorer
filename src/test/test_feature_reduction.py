from tests.unit import TestAPI

from aitia_explorer.feature_reduction.bayesian_gaussian_mixture_wrapper import BayesianGaussianMixtureWrapper
from aitia_explorer.feature_selection_runner import FeatureSelectionRunner
from aitia_explorer.target_data.loader import TargetData


class Test_Feature_Reduction(TestAPI):
    """
    Tests for feature reduction.
    """

    runner = FeatureSelectionRunner()

    bgmm = BayesianGaussianMixtureWrapper()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_randomforest_feature_reduction(self):
        hepart_data = TargetData.hepar2_100_data()
        self.assertTrue(hepart_data is not None, "No data loaded.")
        feature_list = FeatureSelectionRunner.random_forest_feature_reduction(hepart_data, 10)
        df_reduced = self.bgmm.get_reduced_dataframe(hepart_data, feature_list)
        self.assertTrue(df_reduced is not None)

    def test_pfa(self):
        hepart_data = TargetData.hepar2_100_data()
        self.assertTrue(hepart_data is not None, "No data loaded.")
        feature_indices = FeatureSelectionRunner.pfa_feature_reduction(hepart_data, 10)
        df_reduced = self.bgmm.get_reduced_dataframe(hepart_data, feature_indices)
        self.assertTrue(df_reduced is not None)

    def test_linear_regression(self):
        hepart_data = TargetData.hepar2_100_data()
        self.assertTrue(hepart_data is not None, "No data loaded.")
        feature_indices = FeatureSelectionRunner.linear_regression_feature_reduction(hepart_data, 10)
        df_reduced = self.bgmm.get_reduced_dataframe(hepart_data, feature_indices)
        self.assertTrue(df_reduced is not None)

    def test_xgboost(self):
        hepart_data = TargetData.hepar2_100_data()
        self.assertTrue(hepart_data is not None, "No data loaded.")
        feature_indices = FeatureSelectionRunner.xgboost_feature_reduction(hepart_data, 10)
        df_reduced = self.bgmm.get_reduced_dataframe(hepart_data, feature_indices)
        self.assertTrue(df_reduced is not None)

    def test_rfe(self):
        hepart_data = TargetData.hepar2_100_data()
        self.assertTrue(hepart_data is not None, "No data loaded.")
        feature_indices = FeatureSelectionRunner.rfe_feature_reduction(hepart_data, 10)
        df_reduced = self.bgmm.get_reduced_dataframe(hepart_data, feature_indices)
        self.assertTrue(df_reduced is not None)


