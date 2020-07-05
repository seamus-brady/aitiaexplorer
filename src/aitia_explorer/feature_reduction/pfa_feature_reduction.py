"""
TBD Header
"""
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


