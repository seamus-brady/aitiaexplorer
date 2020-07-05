"""
TBD Header
"""
import logging

import numpy as np
from sklearn.linear_model import SGDClassifier

from aitia_explorer.feature_reduction.bayesian_gaussian_mixture_wrapper import BayesianGaussianMixtureWrapper

_logger = logging.getLogger(__name__)


class SGDClassifierFeatureReduction(object):
    """
    A class that allows a number of features to be selected.
    This uses Unsupervised Learning in the form of LinearRegression.
    """

    bgmm = BayesianGaussianMixtureWrapper()

    def __init__(self, ):
        pass

    @staticmethod
    def get_feature_list(incoming_df, n_features=None):
        """
        Returns a reduced list of features.
        :param incoming_df:
        :param n_features:
        :return:
        """

        # define the model
        model = SGDClassifier()

        # get ths synthetic data
        x, y = SGDClassifierFeatureReduction.bgmm.get_synthetic_training_data(incoming_df)

        # fit the model
        model.fit(x, y)

        # get importance
        coefs = model.coef_

        # sort the feature indexes and return
        features = np.argsort(coefs)[::-1]

        # flatten nested list
        features = features[0]

        if n_features is None:
            # set to the number of columns in the df
            n_features = len(list(incoming_df))

        return list(features[:n_features])
