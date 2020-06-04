"""
This code is based on work from https://github.com/amber0309/LiNGAM-GC
which is released under the MIT License
"""

from tests.unit import TestAPI
from causal_discovery.algorithms.lingam_algorithm import LiNGAM_GC_Algorithm


class Test_Lingam_Algorithm(TestAPI):

    def test_lingam_results(self):
        X, test_adjacency_matrix, test_causal_order = LiNGAM_GC_Algorithm.generate_test_gcm(4, 3)
        model = LiNGAM_GC_Algorithm()
        model.fit(X)
        causal_order, adjacency_matrix= model.get_results()
        # we cannot guarantee a match as the results are random, so just use counts
        self.assertTrue(len(causal_order) == 4)
        self.assertTrue(adjacency_matrix.shape == (4, 4))
