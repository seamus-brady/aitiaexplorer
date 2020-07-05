"""
TBD Header
"""
import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from aitia_explorer.feature_reduction.bayesian_gaussian_mixture_wrapper import BayesianGaussianMixtureWrapper

_logger = logging.getLogger(__name__)


class RandomForestFeatureReduction:
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
        Uses an Unsupervised RandomForestClassifier with a sample generated data that is
        marked as synthetic, allowing the RandomForestClassifier to learn the data features.
        A list of features is returned sorted by importance.
        :param incoming_df:
        :param n_features:
        :param treatment:
        :param outcome:
        :return:
        """

        x, y = RandomForestFeatureReduction.bgmm.get_synthetic_training_data(incoming_df)

        # Create an unsupervised random forest classifier
        clf = RandomForestClassifier(n_estimators=10000, random_state=42, n_jobs=-1)

        # Train the classifier
        clf.fit(x, y)

        # sort the feature indexes and return
        features = np.argsort(clf.feature_importances_)[::-1]

        if n_features is None:
            # set to the number of columns in the df
            n_features = len(list(incoming_df))

        return features[:n_features]