"""
TBD Header
"""
import logging

from pycausal import search as s
from pycausal.pycausal import pycausal

from aitia_explorer.algorithms.algorithm_constants import AlgorithmConstants

_logger = logging.getLogger(__name__)


class FGESAlgorithm():
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
    """

    def __init__(self):
        pass

    @staticmethod
    def run_continuous(df, pc=None):
        """
        Run the algorithm against a continuous dataframe to return a dot format causal graph.
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
            tetrad.run(algoId='fges',
                       dfs=df,
                       maxDegree=-1,
                       faithfulnessAssumed=True,
                       verbose=AlgorithmConstants.VERBOSE)
            graph = tetrad.getTetradGraph()
            dot_str = pc.tetradGraphToDot(graph)

            # shutdown java vm
            if single_run:
                pc.stop_vm()
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str

    @staticmethod
    def run_discrete(df, pc=None):
        """
        Run the algorithm against a discrete dataframe to return a dot format causal graph.
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
            tetrad.run(algoId='fges',
                       dfs=df,
                       scoreId='bdeu-score',
                       dataType=AlgorithmConstants.DISCRETE,
                       maxDegree=3,
                       faithfulnessAssumed=True,
                       symmetricFirstStep=True,
                       verbose=AlgorithmConstants.VERBOSE)
            graph = tetrad.getTetradGraph()
            dot_str = pc.tetradGraphToDot(graph)

            # shutdown java vm
            if single_run:
                pc.stop_vm()
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str

    @staticmethod
    def run_mixed(df, pc=None):
        """
        Run the algorithm against a mixed dataframe to return a dot format causal graph.
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
            tetrad.run(algoId='fges',
                       dfs=df,
                       scoreId='cg-bic-score',
                       dataType=AlgorithmConstants.MIXED,
                       numCategoriesToDiscretize=4,
                       maxDegree=3,
                       faithfulnessAssumed=True,
                       symmetricFirstStep=True,
                       verbose=AlgorithmConstants.VERBOSE)
            graph = tetrad.getTetradGraph()
            dot_str = pc.tetradGraphToDot(graph)

            # shutdown java vm
            if single_run:
                pc.stop_vm()
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str
