"""
TBD Header
"""
import logging

from aitia_explorer.algorithms.bayes_est_algorithm import BayesEstAlgorithm
from aitia_explorer.algorithms.fci_algorithm import FCIAlgorithm
from aitia_explorer.algorithms.fges_algorithm import FGESAlgorithm
from aitia_explorer.algorithms.gfci_algorithm import GFCIAlgorithm
from aitia_explorer.algorithms.notears_algorithm import NOTEARSAlgorithm
from aitia_explorer.algorithms.pc_algorithm import PCAlgorithm
from aitia_explorer.algorithms.rfci_algorithm import RFCIAlgorithm
from aitia_explorer.feature_reduction.linear_regression_feature_reduction import LinearRegressionFeatureReduction
from aitia_explorer.feature_reduction.pfa_feature_reduction import PrincipalFeatureAnalysis
from aitia_explorer.feature_reduction.randomforest_feature_reduction import RandomForestFeatureReduction
from aitia_explorer.feature_reduction.recursive_feature_elimination import RecursiveFeatureElimination
from aitia_explorer.feature_reduction.xgboost_feature_reduction import XGBoostFeatureReduction

_logger = logging.getLogger(__name__)


class FeatureSelectionRunner:
    """
    Class that runs feature selection algorithms.
    """

    def __init__(self):
        pass
        # feature selection constants
        self.LINEAR_REGRESSION = ('LINEAR_REGRESSION',
                                  FeatureSelectionRunner.linear_regression_feature_reduction)
        self.PRINCIPAL_FEATURE_ANALYSIS = ('PRINCIPAL_FEATURE_ANALYSIS',
                                           FeatureSelectionRunner.pfa_feature_reduction)
        self.RANDOM_FOREST = ('RANDOM_FOREST',
                              FeatureSelectionRunner.random_forest_feature_reduction)
        self.RECURSIVE_FEATURE_ELIMINATION = ('RECURSIVE_FEATURE_ELIMINATION',
                                              FeatureSelectionRunner.rfe_feature_reduction)
        self.XGBOOST = ('XGBOOST',
                        FeatureSelectionRunner.xgboost_feature_reduction)

    # -------------------------------------------------------------------------------------------------
    #                   The methods below run the causal discovery algorithms
    # -------------------------------------------------------------------------------------------------

    ############### LinearRegressionFeatureReduction ##################
    @staticmethod
    def linear_regression_feature_reduction(df, n_features=10):
        return LinearRegressionFeatureReduction.get_feature_list(df, n_features=n_features)

    ############### PrincipalFeatureAnalysis ##################
    @staticmethod
    def pfa_feature_reduction(df, n_features=10):
        return PrincipalFeatureAnalysis.get_feature_list(df, n_features=n_features)

    ############### RandomForestFeatureReduction ##################
    @staticmethod
    def random_forest_feature_reduction(df, n_features=10):
        return RandomForestFeatureReduction.get_feature_list(df, n_features=n_features)

    ############### RecursiveFeatureElimination ##################
    @staticmethod
    def rfe_feature_reduction(df, n_features=10):
        return RecursiveFeatureElimination.get_feature_list(df, n_features=n_features)

    ############### XGBoostFeatureReduction ##################
    @staticmethod
    def xgboost_feature_reduction(df, n_features=10):
        return XGBoostFeatureReduction.get_feature_list(df, n_features=n_features)