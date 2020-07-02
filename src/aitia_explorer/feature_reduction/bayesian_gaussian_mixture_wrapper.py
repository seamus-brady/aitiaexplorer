"""
TBD Header
"""
import logging

import pandas as pd
from sklearn import mixture

_logger = logging.getLogger(__name__)


class BayesianGaussianMixtureWrapper(object):
    """
    A class that wraps BayesianGaussianMixture.
    """

    def __init__(self, ):
        pass

    def get_gmm_sample_data(self, incoming_df, column_list, sample_size):
        """
        Unsupervised Learning in the form of BayesianGaussianMixture to create sample data.
        """
        gmm = mixture.BayesianGaussianMixture(n_components=2,
                                              covariance_type="full",
                                              n_init=100,
                                              random_state=42).fit(incoming_df)
        clustered_data = gmm.sample(sample_size)
        clustered_df = pd.DataFrame(clustered_data[0], columns=column_list)
        return clustered_df

    def get_synthetic_training_data(self, incoming_df):
        """
        Creates synthetic training data by sampling from a BayesianGaussianMixture supplied distribution.
        Synthetic data is then labelled differently from the original data.
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
        x = df_combined.drop(['original_data'], axis=1).values
        y = df_combined['original_data'].values
        y = y.ravel()

        return x, y

    def get_reduced_dataframe(self, incoming_df, feature_indices, sample_with_gmm=False):
        """
        Returns a df with the requested features only.
        """
        requested_features = [list(incoming_df)[i] for i in feature_indices]
        df_reduced = incoming_df[requested_features]
        if sample_with_gmm:
            return self.get_gmm_sample_data(df_reduced,
                                            list(df_reduced),
                                            len(df_reduced.index))
        return df_reduced, requested_features
