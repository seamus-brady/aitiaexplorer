"""
TBD Header
"""
import logging

import numpy as np
from xgboost import XGBClassifier

from aitia_explorer.feature_reduction.bayesian_gaussian_mixture_wrapper import BayesianGaussianMixtureWrapper

_logger = logging.getLogger(__name__)


class XGBoostFeatureReduction:
    """
    A class that allows a number of features to be selected.
    This used Unsupervised Learning in the form of BayesianGaussianMixture and
    Unsupervised RandomForestClassifier.
    """

    bgmm = BayesianGaussianMixtureWrapper()

    def __init__(self):
        pass

    @staticmethod
    def get_feature_list(incoming_df, n_features=None):
        """
        Uses an Unsupervised XGBClassifier with a sample generated data that is
        marked as synthetic, allowing the XGBClassifier to learn the data features.
        A list of features is returned sorted by importance.
        :param incoming_df:
        :param n_features:
        :param treatment:
        :param outcome:
        :return:
        """

        if n_features is None:
            # set to the number of columns in the df
            n_features = len(list(incoming_df))

        x, y = XGBoostFeatureReduction.bgmm.get_synthetic_training_data(incoming_df)

        # Create an unsupervised random forest classifier
        clf = XGBClassifier(n_samples=1000, n_features=n_features, random_state=42, verbosity=0)

        # Train the classifier
        clf.fit(x, y)

        # sort the feature indexes and return
        features =  np.argsort(clf.feature_importances_)[::-1]

        return list(features[:n_features])