"""
TBD Header
"""
import logging

import numpy as np
from sklearn.linear_model import LinearRegression

from aitia_explorer.feature_reduction.bayesian_gaussian_mixture_wrapper import BayesianGaussianMixtureWrapper

_logger = logging.getLogger(__name__)


class LinearRegressionFeatureReduction(object):
    """
    A class that allows a number of features to be selected.
    This uses Unsupervised Learning in the form of LinearRegression.
    """

    bgmm = BayesianGaussianMixtureWrapper()

    def __init__(self, ):
        pass

    def get_feature_list(self, incoming_df, n_features=10):
        """
        Returns a reduced list of features.
        :param incoming_df:
        :param n_features:
        :return:
        """

        # define the model
        model = LinearRegression()

        # get ths synthetic data
        x, y = self.bgmm.get_synthetic_training_data(incoming_df)

        # fit the model
        model.fit(x, y)

        # get importance
        coefs = model.coef_

        # sort the feature indexes and return
        features = np.argsort(coefs)[::-1]

        return features[:n_features]
