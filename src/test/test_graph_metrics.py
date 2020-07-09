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
import os

from tests.unit import TestAPI

from aitia_explorer.metrics.graph_metrics import GraphMetrics
from aitia_explorer.algorithm_runner import AlgorithmRunner
from aitia_explorer.util.graph_util import GraphUtil
from aitia_explorer.target_data.loader import TargetData


class Test_Metrics(TestAPI):
    """
    Tests for graph metrics.
    """
    pc_util = AlgorithmRunner()

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'resources/data')

    def tearDown(self):
        pass

    def test_retrieve_adjacency_matrix(self):
        dot_str = TargetData.simulated_data_1_graph()
        graph = GraphUtil.get_digraph_from_dot(dot_str)
        metrics = GraphMetrics()
        adj_matrix = metrics.retrieve_adjacency_matrix(graph)
        self.assertTrue(adj_matrix is not None, "No adjacency matrix created.")

    def test_precision_recall(self):
        # get the simulated data
        simulated_data = TargetData.simulated_data_1()
        dot_str = self.pc_util.algo_pc(simulated_data)
        pred_graph = GraphUtil.get_digraph_from_dot(dot_str)

        # get the known data
        dot_str = TargetData.simulated_data_1_graph()
        target_graph =  GraphUtil.get_digraph_from_dot(dot_str)
        metrics = GraphMetrics()
        prec_recall = metrics.precision_recall(target_graph, pred_graph)

        self.assertTrue(prec_recall[0] == 0.41250000000000003)

    def test_shd(self):
        # get the simulated data
        simulated_data = TargetData.simulated_data_1()
        dot_str = self.pc_util.algo_pc(simulated_data)
        pred_graph = GraphUtil.get_digraph_from_dot(dot_str)

        # get the known data
        dot_str = TargetData.simulated_data_1_graph()
        target_graph = GraphUtil.get_digraph_from_dot(dot_str)
        metrics = GraphMetrics()
        shd = metrics.SHD(target_graph, pred_graph)

        self.assertTrue(shd == 10)
