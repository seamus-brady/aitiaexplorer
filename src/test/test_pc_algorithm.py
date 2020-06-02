"""
This code is based on work from https://github.com/keiichishima/pcalg/
which is released under the BSD License
"""

from tests.unit import TestAPI
import itertools

import numpy.random
import pandas as pd
import networkx as nx

from causal_discovery.algorithms.ic_algorithm import IC_Algorithm
from causal_discovery.independence_tests.robust_regression_test import RobustRegressionTest

TEST_SET_SIZE = 2000


class Test_PC_Algorithm(TestAPI):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def