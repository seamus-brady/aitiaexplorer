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

import pandas as pd
from tests.unit import TestAPI

from aitia_explorer.algorithm_runner import AlgorithmRunner
from aitia_explorer.target_data.loader import TargetData


class Test_Algorithms(TestAPI):
    """
    Tests for individual causal discovery causal_algorithms.
    Please note these tests have to be run individually as the Java VM wrapper in PyCausal
    does not like repeated runs! But hopefully they are still useful.
    """

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'resources/data')

    def tearDown(self):
        pass


    ############### BayesEst ##################
    def test_algo_bayes_est(self):
        data_dir = os.path.join(self.data_dir, "sim_discrete_data_20vars_100cases.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_bayes_est(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### FCI ##################
    def test_fci(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_fci(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### PC ##################
    def test_pc(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_pc(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### FGES ##################
    def test_algo_fges_continuous(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_fges_continuous(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_fges_discrete(self):
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_fges_discrete(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_fges_mixed(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_fges_mixed(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### GFCI ##################
    def test_algo_gfci_continuous(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_gfci_continuous(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_gfci_discrete(self):
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_gfci_discrete(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_gfci_mixed(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_gfci_mixed(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### RFCI ##################
    def test_algo_rfci_continuous(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_rfci_continuous(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_rfci_discrete(self):
        data_dir = os.path.join(self.data_dir, "audiology.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_rfci_discrete(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    def test_algo_rfci_mixed(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_rfci_mixed(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### Hill Climbing ##################
    def test_algo_hill_climbing(self):
        data_dir = os.path.join(self.data_dir, "charity.txt")
        df = pd.read_table(data_dir, sep="\t")
        dot_str = AlgorithmRunner.algo_hill_climber(df)
        self.assertTrue(dot_str is not None, "No graph returned.")

    ############### MIIC ##################
    def test_algo_miic(self):
        df = TargetData.hepar2_100_data()
        df = df.drop(['fat', 'surgery', 'gallstones'], axis=1)
        latent_edges = AlgorithmRunner.algo_miic(df)
        self.assertTrue(len(latent_edges)==7, "No latent edges returned.")