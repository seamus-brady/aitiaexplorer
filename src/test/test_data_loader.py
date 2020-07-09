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
from tests.unit import TestAPI

from aitia_explorer.target_data.loader import TargetData
from aitia_explorer.util.graph_util import GraphUtil


class Test_Data_Loader(TestAPI):
    """
    Tests for scm and data generation.
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_load_simulated_data_graph(self):
        simulated_data = TargetData.simulated_data_1()
        self.assertTrue(simulated_data is not None, "No simulated data loaded.")

        dot_str = TargetData.simulated_data_1_graph()
        self.assertTrue(dot_str is not None, "No simulated graph loaded.")

    def test_create_known_graph(self):
        dot_str = TargetData.simulated_data_1_graph()
        graph = GraphUtil.get_digraph_from_dot(dot_str)
        self.assertTrue(graph.edges() is not None, "No known simulated graph created.")
        self.assertTrue(graph.nodes() is not None, "No known simulated graph created.")

    def test_scm_generation(self):
        scm1 = TargetData.scm1()
        scm2 = TargetData.scm2()
        scm3 = TargetData.scm2()
        self.assertTrue(scm1 is not None)
        self.assertTrue(scm2 is not None)
        self.assertTrue(scm3 is not None)

    def test_random_scm(self):
        random_scm = TargetData.random_scm1()
        self.assertTrue(random_scm is not None)

    def test_hepar2_10k_data(self):
        hepar2_10k_data = TargetData.hepar2_10k_data()
        self.assertTrue(hepar2_10k_data is not None)

    def test_hepar2_100_data(self):
        hepar2_100_data = TargetData.hepar2_100_data()
        self.assertTrue(hepar2_100_data is not None)

    def test_hepar2_graph(self):
        hepar2_dot = TargetData.hepar2_graph()
        self.assertTrue(hepar2_dot is not None)

