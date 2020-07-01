"""
TBD Header
"""
import logging

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFE

from aitia_explorer.feature_reduction.bayesian_gaussian_mixture_wrapper import BayesianGaussianMixtureWrapper

_logger = logging.getLogger(__name__)


class RecursiveFeatureElimination:
    """
    Feature ranking with recursive feature elimination.
    """

    bgmm = BayesianGaussianMixtureWrapper()

    def __init__(self):
        pass

    @staticmethod
    def get_feature_list(incoming_df, n_features=5):
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

        x, y = RecursiveFeatureElimination.bgmm.get_synthetic_training_data(incoming_df)

        # Create an unsupervised random forest classifier
        rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=n_features)

        # Train the classifier
        rfe.fit(x, y)

        # sort the feature indexes and return
        features = []

        for i in range(x.shape[1]):
            # see if column has been marked true or false
            if rfe.support_[i]:
                features.append(i)

        return features
