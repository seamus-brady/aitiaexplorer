#
# This file is part of AitiaExplorer and is released under the FreeBSD License.
#
# Copyright (c) 2020, Seamus Brady <seamus@corvideon.ie>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.#
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

    def test_sgdclassifier(self):
        hepart_data = TargetData.hepar2_100_data()
        self.assertTrue(hepart_data is not None, "No data loaded.")
        feature_indices = FeatureSelectionRunner.sgdclassifier_feature_reduction(hepart_data, 10)
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


