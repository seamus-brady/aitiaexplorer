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