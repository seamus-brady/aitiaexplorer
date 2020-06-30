

from tests.unit import TestAPI

from aitia_explorer.feature_reduction.bayesian_gaussian_mixture_wrapper import BayesianGaussianMixtureWrapper
from aitia_explorer.feature_reduction.linear_regression_feature_reduction import LinearRegressionFeatureReduction
from aitia_explorer.feature_reduction.pfa_feature_reduction import PrincipalFeatureAnalysis
from aitia_explorer.feature_reduction.randomforest_feature_reduction import RandomForestFeatureReduction
from aitia_explorer.feature_reduction.xgboost_feature_reduction import XGBoostFeatureReduction
from aitia_explorer.target_data.loader import TargetData


class Test_Feature_Reduction(TestAPI):
    """
    Tests for feature reduction.
    """

    bgmm = BayesianGaussianMixtureWrapper()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_randomforest_feature_reduction(self):
        hepart_data = TargetData.hepar2_100_data()
        self.assertTrue(hepart_data is not None, "No data loaded.")

        feature_reducer = RandomForestFeatureReduction()
        feature_list = feature_reducer.get_feature_list(hepart_data, 10)
        df_reduced = self.bgmm.get_reduced_dataframe(hepart_data, feature_list)
        self.assertTrue(df_reduced is not None)

    def test_pfa(self):
        hepart_data = TargetData.hepar2_100_data()
        self.assertTrue(hepart_data is not None, "No data loaded.")

        pfa = PrincipalFeatureAnalysis()
        feature_indices = pfa.get_feature_list(hepart_data, 10)
        df_reduced = self.bgmm.get_reduced_dataframe(hepart_data, feature_indices)
        self.assertTrue(df_reduced is not None)

    def test_linear_regression(self):
        hepart_data = TargetData.hepar2_100_data()
        self.assertTrue(hepart_data is not None, "No data loaded.")

        lrfr = LinearRegressionFeatureReduction()
        feature_indices = lrfr.get_feature_list(hepart_data, 10)
        df_reduced = self.bgmm.get_reduced_dataframe(hepart_data, feature_indices)
        self.assertTrue(df_reduced is not None)

    def test_xgboost(self):
        hepart_data = TargetData.hepar2_100_data()
        self.assertTrue(hepart_data is not None, "No data loaded.")

        xgb = XGBoostFeatureReduction()
        feature_indices = xgb.get_feature_list(hepart_data, 10)
        df_reduced = self.bgmm.get_reduced_dataframe(hepart_data, feature_indices)
        self.assertTrue(df_reduced is not None)