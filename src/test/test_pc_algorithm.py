"""
This code is based on work from https://github.com/keiichishima/pcalg/
which is released under the BSD License
"""

from tests.unit import TestAPI

import numpy as np
import networkx as nx

from causal_discovery.algorithms.pg_algorithm import PC_Algorithm
from causal_discovery.independence_tests.g_square_tests import GSquareTest

TEST_SET_SIZE = 2000


class Test_PC_Algorithm(TestAPI):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_me(self):
        pc_algo = PC_Algorithm()
        dm = np.array(bin_data).reshape((5000, 5))
        (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_bin,
                                         data_matrix=dm,
                                         alpha=0.01)
        g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
        g_answer = nx.DiGraph()
        g_answer.add_nodes_from([0, 1, 2, 3, 4])
        g_answer.add_edges_from([(0, 1), (2, 3), (3, 2), (3, 1),
                                 (2, 4), (4, 2), (4, 1)])
        print('Edges are:', g.edges(), end='')
        if nx.is_isomorphic(g, g_answer):
            print(' => GOOD')
        else:
            print(' => WRONG')
            print('True edges should be:', g_answer.edges())

        dm = np.array(dis_data).reshape((10000, 5))
        (g, sep_set) = estimate_skeleton(indep_test_func=ci_test_dis,
                                         data_matrix=dm,
                                         alpha=0.01,
                                         levels=[3, 2, 3, 4, 2])
        g = estimate_cpdag(skel_graph=g, sep_set=sep_set)
        g_answer = nx.DiGraph()
        g_answer.add_nodes_from([0, 1, 2, 3, 4])
        g_answer.add_edges_from([(0, 2), (1, 2), (1, 3), (4, 3)])
        print('Edges are:', g.edges(), end='')
        if nx.is_isomorphic(g, g_answer):
            print(' => GOOD')
        else:
            print(' => WRONG')
            print('True edges should be:', g_answer.edges())
