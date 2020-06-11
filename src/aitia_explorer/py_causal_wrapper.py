"""
TBD Header
"""
import logging

import pygraphviz
from causalgraphicalmodels import CausalGraphicalModel
from networkx.drawing import nx_agraph
from pycausal import search as s
from pycausal.pycausal import pycausal
from aitia_explorer.causal_data_type import CausalDataType

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
    #                   The methods below return causal discovery algorithms....
    # -------------------------------------------------------------------------------------------------

    def algo_bayes_est(self, df, pc=None):
        """
        bayesEst is the revised Greedy Equivalence Search (GES) algorithm developed
        by Joseph D. Ramsey, Director of Research Computing, Department of Philosophy,
        Carnegie Mellon University, Pittsburgh, PA.
        :param df: dataframe
        :return: dot graph string
        """
        dot_str = None
        try:
            # start java vm and get algo runner
            if pc is None:
                pc = pycausal()
                pc.start_vm()

            bayes_est = s.bayesEst(df, depth=-1, alpha=0.05, verbose=True)
            graph = bayes_est.getTetradGraph()
            dot_str = pc.tetradGraphToDot(graph)

            # stop java vm
            if pc is None:
                pc.stop_vm()
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str


    def algo_fges_continuous(self, df, causal_data_type: CausalDataType, pc=None):
        """
        FGES is an optimized and parallelized version of an algorithm developed by Meek [Meek, 1997]
        called the Greedy Equivalence Search (GES). The algorithm was further developed and studied
        by Chickering [Chickering, 2002]. GES is a Bayesian algorithm that heuristically searches
        the space of CBNs and returns the model with highest Bayesian score it finds. In particular,
        GES starts its search with the empty graph. It then performs a forward stepping search in
        which edges are added between nodes in order to increase the Bayesian score. This process
        continues until no single edge addition increases the score. Finally, it performs a backward
        stepping search that removes edges until no single edge removal can increase the score.
        The algorithms requires a decomposable score—that is, a score that for the entire DAG model
        is a sum of logged scores of each variables given its parents in the model. The algorithms
        can take all continuous data (using the SEM BIC score), all discrete data
        (using the BDeu score) or a mixture of continuous and discrete data
        (using the Conditional Gaussian score); these are all decomposable scores.
        :param df: dataframe
        :return: dot graph string
        """
        single_run = False
        dot_str = None
        try:
            # start java vm and get algo runner
            if pc is None:
                pc = pycausal()
                pc.start_vm()
                single_run = True

            tetrad = s.tetradrunner()
            tetrad.run(algoId='fges', dfs=df, scoreId='sem-bic', dataType=causal_data_type.CONTINUOUS_DATA,
                       maxDegree=-1, faithfulnessAssumed=True, verbose=True)
            graph = tetrad.getTetradGraph()
            dot_str = pc.tetradGraphToDot(graph)

            # shutdown java vm
            if single_run:
                pc.stop_vm()
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str

    def algo_fges_discrete(self, df, pc=None):
        """
        FGES is an optimized and parallelized version of an algorithm developed by Meek [Meek, 1997]
        called the Greedy Equivalence Search (GES). The algorithm was further developed and studied
        by Chickering [Chickering, 2002]. GES is a Bayesian algorithm that heuristically searches
        the space of CBNs and returns the model with highest Bayesian score it finds. In particular,
        GES starts its search with the empty graph. It then performs a forward stepping search in
        which edges are added between nodes in order to increase the Bayesian score. This process
        continues until no single edge addition increases the score. Finally, it performs a backward
        stepping search that removes edges until no single edge removal can increase the score.
        The algorithms requires a decomposable score—that is, a score that for the entire DAG model
        is a sum of logged scores of each variables given its parents in the model. The algorithms
        can take all continuous data (using the SEM BIC score), all discrete data
        (using the BDeu score) or a mixture of continuous and discrete data
        (using the Conditional Gaussian score); these are all decomposable scores.
        :param df: dataframe
        :return: dot graph string
        """
        single_run = False
        dot_str = None
        try:
            # start java vm and get algo runner
            if pc is None:
                pc = pycausal()
                pc.start_vm()
                single_run = True

            tetrad = s.tetradrunner()
            tetrad.run(algoId='fges', dfs=df, scoreId='cg-bic-score', dataType='discrete',
                       maxDegree=3, faithfulnessAssumed=True, symmetricFirstStep=True, verbose=True)
            graph = tetrad.getTetradGraph()
            dot_str = pc.tetradGraphToDot(graph)

            # shutdown java vm
            if single_run:
                pc.stop_vm()
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str


    def algo_fges_mixed(self, df, pc=None):
        """
        FGES is an optimized and parallelized version of an algorithm developed by Meek [Meek, 1997]
        called the Greedy Equivalence Search (GES). The algorithm was further developed and studied
        by Chickering [Chickering, 2002]. GES is a Bayesian algorithm that heuristically searches
        the space of CBNs and returns the model with highest Bayesian score it finds. In particular,
        GES starts its search with the empty graph. It then performs a forward stepping search in
        which edges are added between nodes in order to increase the Bayesian score. This process
        continues until no single edge addition increases the score. Finally, it performs a backward
        stepping search that removes edges until no single edge removal can increase the score.
        The algorithms requires a decomposable score—that is, a score that for the entire DAG model
        is a sum of logged scores of each variables given its parents in the model. The algorithms
        can take all continuous data (using the SEM BIC score), all discrete data
        (using the BDeu score) or a mixture of continuous and discrete data
        (using the Conditional Gaussian score); these are all decomposable scores.
        :param df: dataframe
        :return: dot graph string
        """
        single_run = False
        dot_str = None
        try:
            # start java vm and get algo runner
            if pc is None:
                pc = pycausal()
                pc.start_vm()
                single_run = True

            tetrad = s.tetradrunner()
            tetrad.run(algoId='fges', dfs=df, scoreId='cg-bic-score', dataType='mixed',
                       maxDegree=3, faithfulnessAssumed=True, symmetricFirstStep=True, verbose=True)
            graph = tetrad.getTetradGraph()
            dot_str = pc.tetradGraphToDot(graph)

            # shutdown java vm
            if single_run:
                pc.stop_vm()
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str