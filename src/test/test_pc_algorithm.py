"""
This code is based on work from https://github.com/keiichishima/pcalg/
which is released under the BSD License
"""

import networkx as nx
import numpy as np
from tests.unit import TestAPI

import test.resources.bin_data as test_data
from causal_discovery.algorithms.pc_algorithm import PC_Algorithm
from causal_discovery.independence_tests.g_square_tests import GSquareTest

TEST_SET_SIZE = 2000


class Test_PC_Algorithm(TestAPI):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_binary_data(self):
        test_passed = True
        pc_algo = PC_Algorithm()
        data_matrix = np.array(test_data.binary_test_data).reshape((5000, 5))
        (g, sep_set) = pc_algo.estimate_skeleton(
            indep_test_func=GSquareTest.test_binary,
            data_matrix=data_matrix,
            alpha=0.01)
        g = pc_algo.estimate_cpdag(skel_graph=g, sep_set=sep_set)
        g_answer = nx.DiGraph()
        g_answer.add_nodes_from([0, 1, 2, 3, 4])
        g_answer.add_edges_from([(0, 1), (2, 3), (3, 2), (3, 1),
                                 (2, 4), (4, 2), (4, 1)])
        print('Edges are:', g.edges(), end='')
        if not nx.is_isomorphic(g, g_answer):
            print('Failed! True edges should be:', g_answer.edges())
            test_passed = False
        self.assertTrue(test_passed)

    def test_discrete_data(self):
        test_passed = True
        pc_algo = PC_Algorithm()
        data_matrix = np.array(test_data.discrete_test_data).reshape((10000, 5))
        (g, sep_set) = pc_algo.estimate_skeleton(indep_test_func=GSquareTest.test_discrete,
                                                 data_matrix=data_matrix,
                                                 alpha=0.01,
                                                 levels=[3, 2, 3, 4, 2])
        g = pc_algo.estimate_cpdag(skel_graph=g, sep_set=sep_set)
        g_answer = nx.DiGraph()
        g_answer.add_nodes_from([0, 1, 2, 3, 4])
        g_answer.add_edges_from([(0, 2), (1, 2), (1, 3), (4, 3)])
        print('Edges are:', g.edges(), end='')
        if not nx.is_isomorphic(g, g_answer):
            print('Failed! True edges should be:', g_answer.edges())
            test_passed = False
        self.assertTrue(test_passed)
