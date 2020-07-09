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

        # add the class to a copy of incoming df, stops weird errors due to changed dataframes
        working_df = incoming_df.copy(deep=True)
        working_df['original_data'] = 1

        # concatinate the two dataframes
        df_combined = working_df.append(df_bgmm, ignore_index=True)

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
