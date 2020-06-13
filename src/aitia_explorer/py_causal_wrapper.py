"""
TBD Header
"""
import logging

import pygraphviz
from causalgraphicalmodels import CausalGraphicalModel
from networkx.drawing import nx_agraph
from pycausal import search as s
from pycausal.pycausal import pycausal

from aitia_explorer.algorithms.bayes_est_algorithm import BayesEstAlgorithm
from aitia_explorer.algorithms.fci_algorithm import FCIAlgorithm
from aitia_explorer.algorithms.fges_algorithm import FGESAlgorithm
from aitia_explorer.algorithms.gfci_algorithm import GFCIAlgorithm
from aitia_explorer.algorithms.pc_algorithm import PCAlgorithm
from aitia_explorer.algorithms.rfci_algorithm import RFCIAlgorithm

_logger = logging.getLogger(__name__)


class PyCausalWrapper():
    """
    TBD
    """

    def __init__(self):
        pass

    def get_causal_discovery_algos(self):
        """
        TBD
        :return:
        """
        # start java vm and get algo runner
        pc = pycausal()
        pc.start_vm()
        tetrad = s.tetradrunner()

        causal_discovery_algos = []
        algo_dict = tetrad.algos
        for k, v in algo_dict.items():
            causal_discovery_algos.append(k)

        # stop java vm
        pc.stop_vm()
        return causal_discovery_algos

    def get_causal_graph_from_dot(self, dot_str):
        """
        TBD
        :param dot_str: dot string
        :return: CausalGraphicalModel
        """

        # load graph from dot data
        nx_graph = nx_agraph.from_agraph(pygraphviz.AGraph(dot_str))

        # create a causal graph
        causal_graph = CausalGraphicalModel(
            nodes=nx_graph.nodes(),
            edges=nx_graph.edges()
        )

        return causal_graph

    # -------------------------------------------------------------------------------------------------
    #                   The methods below run the causal discovery algorithms
    # -------------------------------------------------------------------------------------------------

    ############### BayesEst ##################
    def algo_bayes_est(self, df, pc=None):
        return BayesEstAlgorithm.run(df, pc)

    ############### FCI ##################
    def algo_fci(self, df, pc=None):
        return FCIAlgorithm.run(df, pc)

    ############### PC ##################
    def algo_pc(self, df, pc=None):
        return PCAlgorithm.run(df, pc)

    ############### FGES ##################
    def algo_fges_continuous(self, df, pc=None):
        return FGESAlgorithm.run_continuous(df, pc)

    def algo_fges_discrete(self, df, pc=None):
        return FGESAlgorithm.run_discrete(df, pc)

    def algo_fges_mixed(self, df, pc=None):
        return FGESAlgorithm.run_mixed(df, pc)

    ############### GFCI ##################
    def algo_gfci_continuous(self, df, pc=None):
        return GFCIAlgorithm.run_continuous(df, pc)

    def algo_gfci_discrete(self, df, pc=None):
        return GFCIAlgorithm.run_discrete(df, pc)

    def algo_gfci_mixed(self, df, pc=None):
        return GFCIAlgorithm.run_mixed(df, pc)

    ############### RFCI ##################
    def algo_rfci_continuous(self, df, pc=None):
        return RFCIAlgorithm.run_continuous(df, pc)

    def algo_rfci_discrete(self, df, pc=None):
        return RFCIAlgorithm.run_discrete(df, pc)

    def algo_rfci_mixed(self, df, pc=None):
        return RFCIAlgorithm.run_mixed(df, pc)