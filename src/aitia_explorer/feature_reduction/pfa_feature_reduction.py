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
from collections import defaultdict

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from aitia_explorer.feature_reduction.bayesian_gaussian_mixture_wrapper import BayesianGaussianMixtureWrapper

_logger = logging.getLogger(__name__)


class PrincipalFeatureAnalysis(object):
    """
    A class that allows a number of features to be selected.
    This use Unsupervised Learning in the form of Principal Feature Analysis.
    See https://stats.stackexchange.com/questions/108743/methods-in-r-or-python-to-perform-feature-selection-in-unsupervised-learning/203978#203978
    and the paper at http://venom.cs.utsa.edu/dmz/techrep/2007/CS-TR-2007-011.pdf
    """

    def __init__(self, ):
        pass

    @staticmethod
    def get_feature_list(incoming_df, n_features=None, q=None):
        """
        Returns a reduced list of features.
        :param incoming_df:
        :param n_features:
        :param q:
        :return:
        """

        if n_features is None:
            # set to the number of columns in the df
            n_features = len(list(incoming_df))

        if not q:
            q = incoming_df.shape[1]

        sc = StandardScaler()
        incoming_df = sc.fit_transform(incoming_df)

        pca = PCA(n_components=q).fit(incoming_df)
        a_q = pca.components_.T

        kmeans = KMeans(n_clusters=n_features).fit(a_q)
        clusters = kmeans.predict(a_q)
        cluster_centers = kmeans.cluster_centers_

        dists = defaultdict(list)
        for i, c in enumerate(clusters):
            dist = euclidean_distances([a_q[i, :]], [cluster_centers[c, :]])[0][0]
            dists[c].append((i, dist))

        # sort the feature indexes and return
        return [sorted(f, key=lambda p: p[1])[0][0] for f in dists.values()]


