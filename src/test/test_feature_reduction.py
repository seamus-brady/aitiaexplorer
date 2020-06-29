

from tests.unit import TestAPI

from aitia_explorer.feature_reduction.feature_reduction import UnsupervisedFeatureReduction
from aitia_explorer.target_data.loader import TargetData


class Test_Feature_Reduction(TestAPI):
    """
    Tests for feature reduction.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_feature_reduction(self):
        hepart_data = TargetData.hepar2_100_data()
        self.assertTrue(hepart_data is not None, "No data loaded.")

        feature_reducer = UnsupervisedFeatureReduction()
        feature_list = feature_reducer.get_reduced_feature_list(hepart_data, 10)
        df_reduced = feature_reducer.get_reduced_dataframe(hepart_data, feature_list)
        self.assertTrue(df_reduced is not None)