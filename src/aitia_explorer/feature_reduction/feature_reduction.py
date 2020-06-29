"""
TBD Header
"""
import logging

import pandas as pd
from sklearn import mixture
from sklearn.ensemble import RandomForestClassifier

_logger = logging.getLogger(__name__)


class UnsupervisedFeatureReduction:
    """
    A class that allows a number of features to be selected.
    This used Unsupervised Learning in the form of BayesianGaussianMixture and
    Unsupervised RandomForestClassifier.
    """

    def __init__(self):
        pass

    def get_reduced_dataframe(self, incoming_df, requested_features, sample_with_gmm=False):
        """
        Returns a df with the requested features only.
        :param incoming_df: df
        :param requested_features: list of features
        :param sample_with_gmm: boolean, should sample with BayesianGaussianMixture?
        :return: df
        """
        df_reduced = incoming_df[requested_features]
        if sample_with_gmm:
            return self.get_gmm_sample_data(df_reduced,
                                            list(df_reduced),
                                            len(df_reduced.index))
        return df_reduced

    def get_reduced_feature_list(self, incoming_df, number_features=5, treatment=None, outcome=None):
        """
        Uses an Unsupervised RandomForestClassifier with a sample generated data that is
        marked as synthetic, allowing the RandomForestClassifier to learn the data features.
        A list of features is returned sorted by importance.
        :param incoming_df:
        :param number_features:
        :param treatment:
        :param outcome:
        :return:
        """
        # number of records in df
        number_records = len(incoming_df.index)

        # get sample data from the unsupervised BayesianGaussianMixture
        df_bgmm = self.get_gmm_sample_data(incoming_df, list(incoming_df), number_records)

        # set the class on the samples
        df_bgmm['original_data'] = 0

        # add the class to the incoming df
        incoming_df['original_data'] = 1

        # concatinate the two dataframes
        df_combined = incoming_df.append(df_bgmm, ignore_index=True)

        # shuffle the data
        df_combined = df_combined.sample(frac=1)

        # get the X and y
        X = df_combined.drop(['original_data'], axis=1).values
        y = df_combined['original_data'].values
        y = y.ravel()

        # get the feature labels
        feat_labels = list(incoming_df)

        # Create an unsupervised random forest classifier
        clf = RandomForestClassifier(n_estimators=10000, random_state=42, n_jobs=-1)

        # Train the classifier
        clf.fit(X, y)

        # now get the features
        feature_dict = {}

        # get the name and gini importance of each feature
        for feature in zip(feat_labels, clf.feature_importances_):
            feature_dict[feature[0]] = feature[1]

        # sort the features
        features_sorted = {k: v for k, v in sorted(feature_dict.items(), key=lambda item: item[1])}
        features_sorted = list(features_sorted.keys())
        features_sorted.reverse()

        # return the number of feature requested
        features_final = features_sorted[:number_features]

        # append the requested treatment and outcome
        if treatment is not None:
            features_final.append(treatment)

        if outcome is not None:
            features_final.append(outcome)

        return features_final

    def get_gmm_sample_data(self, incoming_df, column_list, sample_size):
        """
        Unsupervised Learning in the form of BayesianGaussianMixture to create sample data.
        :param incoming_df: df
        :param column_list: list
        :param sample_size: int
        :return: df
        """
        gmm = mixture.BayesianGaussianMixture(n_components=2,
                                              covariance_type="full",
                                              n_init=100,
                                              random_state=42).fit(incoming_df)
        clustered_data = gmm.sample(sample_size)
        clustered_df = pd.DataFrame(clustered_data[0], columns=column_list)
        return clustered_df
