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
