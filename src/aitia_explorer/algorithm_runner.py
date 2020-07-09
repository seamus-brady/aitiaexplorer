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

from aitia_explorer.causal_algorithms.bayes_est_algorithm import BayesEstAlgorithm
from aitia_explorer.causal_algorithms.fci_algorithm import FCIAlgorithm
from aitia_explorer.causal_algorithms.fges_algorithm import FGESAlgorithm
from aitia_explorer.causal_algorithms.gfci_algorithm import GFCIAlgorithm
from aitia_explorer.causal_algorithms.hill_climbing_algorithm import HillClimbingAlgorithm
from aitia_explorer.causal_algorithms.miic_algorithm import MIICAlgorithm
from aitia_explorer.causal_algorithms.notears_algorithm import NOTEARSAlgorithm
from aitia_explorer.causal_algorithms.pc_algorithm import PCAlgorithm
from aitia_explorer.causal_algorithms.rfci_algorithm import RFCIAlgorithm

_logger = logging.getLogger(__name__)


class AlgorithmRunner:
    """
    Class that runs causal discovery causal_algorithms.
    """

    def __init__(self):
        # causal discovery algorithm constants
        self.BAYES_EST = ('BayesEst', AlgorithmRunner.algo_bayes_est)
        self.FCI = ('FCI', AlgorithmRunner.algo_fci)
        self.PC = ('PC', AlgorithmRunner.algo_pc)
        self.FGES_continuous = ('FGES-continuous', AlgorithmRunner.algo_fges_continuous)
        self.FGES_discrete = ('FGES-discrete', AlgorithmRunner.algo_fges_discrete)
        self.FGES_mixed_data = ('FGES-mixed-data', AlgorithmRunner.algo_fges_mixed)
        self.GFCI_continuous = ('GFCI-continuous', AlgorithmRunner.algo_gfci_continuous)
        self.GFCI_discrete = ('GFCI-discrete', AlgorithmRunner.algo_gfci_discrete)
        self.GFCI_mixed_data = ('GFCI-mixed-data', AlgorithmRunner.algo_gfci_mixed)
        self.RFCI_continuous = ('RFCI-continuous', AlgorithmRunner.algo_rfci_continuous)
        self.RFCI_discrete = ('RFCI-discrete', AlgorithmRunner.algo_rfci_discrete)
        self.RFCI_mixed_data = ('RFCI-mixed-data', AlgorithmRunner.algo_rfci_mixed)

        # NOTEARS is included for convenience, but will not return a labelled causal graph
        # which makes it less useful for causal exploration.
        # This returns an adjacency matrix instead.
        self.NOTEARS = ('NOTEARS', AlgorithmRunner.algo_notears)

        # this greedy hill climbing algorithm is provided to approimate missing
        # target graphs so we can provide some heuristics to the user
        self.HILL_CLIMBING = ('HILL_CLIMBING', AlgorithmRunner.algo_hill_climber)

        # the MIIC algorithm is used to return a list of unobserved latent edges in a dataframe
        self.MIIC = ('MIIC', AlgorithmRunner.algo_miic)

    def get_all_causal_algorithms(self):
        return [self.BAYES_EST,
                self.FCI,
                self.PC,
                self.FGES_continuous,
                self.FGES_discrete,
                self.FGES_mixed_data,
                self.GFCI_continuous,
                self.GFCI_discrete,
                self.GFCI_mixed_data,
                self.RFCI_continuous,
                self.RFCI_discrete,
                self.RFCI_mixed_data,
                self.NOTEARS,
                self.HILL_CLIMBING,
                self.MIIC
                ]

    # -------------------------------------------------------------------------------------------------
    #                   The methods below run the causal discovery causal_algorithms
    # -------------------------------------------------------------------------------------------------

    ############### BayesEst ##################
    @staticmethod
    def algo_notears(df, pc=None):
        return NOTEARSAlgorithm.run(df, pc)

    ############### BayesEst ##################
    @staticmethod
    def algo_bayes_est(df, pc=None):
        return BayesEstAlgorithm.run(df, pc)

    ############### FCI ##################
    @staticmethod
    def algo_fci(df, pc=None):
        return FCIAlgorithm.run(df, pc)

    ############### PC ##################
    @staticmethod
    def algo_pc(df, pc=None):
        return PCAlgorithm.run(df, pc)

    ############### FGES ##################
    @staticmethod
    def algo_fges_continuous(df, pc=None):
        return FGESAlgorithm.run_continuous(df, pc)

    @staticmethod
    def algo_fges_discrete(df, pc=None):
        return FGESAlgorithm.run_discrete(df, pc)

    @staticmethod
    def algo_fges_mixed(df, pc=None):
        return FGESAlgorithm.run_mixed(df, pc)

    ############### GFCI ##################
    @staticmethod
    def algo_gfci_continuous(df, pc=None):
        return GFCIAlgorithm.run_continuous(df, pc)

    @staticmethod
    def algo_gfci_discrete(df, pc=None):
        return GFCIAlgorithm.run_discrete(df, pc)

    @staticmethod
    def algo_gfci_mixed(df, pc=None):
        return GFCIAlgorithm.run_mixed(df, pc)

    ############### RFCI ##################
    @staticmethod
    def algo_rfci_continuous(df, pc=None):
        return RFCIAlgorithm.run_continuous(df, pc)

    @staticmethod
    def algo_rfci_discrete(df, pc=None):
        return RFCIAlgorithm.run_discrete(df, pc)

    @staticmethod
    def algo_rfci_mixed(df, pc=None):
        return RFCIAlgorithm.run_mixed(df, pc)

    ############### Hill Climber ##################
    @staticmethod
    def algo_hill_climber(df, pc=None):
        return HillClimbingAlgorithm.run(df, pc)

    ############### MIIC ##################
    @staticmethod
    def algo_miic(df, pc=None):
        return MIICAlgorithm.run(df, pc)
