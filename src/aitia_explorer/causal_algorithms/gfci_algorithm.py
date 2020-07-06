"""
TBD Header
"""
import logging

from pycausal import search as s
from pycausal.pycausal import pycausal

from aitia_explorer.causal_algorithms.algorithm_constants import AlgorithmConstants

_logger = logging.getLogger(__name__)


class GFCIAlgorithm():
    """
    GFCI is a combination of the FGES [CCD-FGES, 2016] algorithm and the FCI algorithm [Spirtes, 1993]
    that improves upon the accuracy and efficiency of FCI. In order to understand the basic methodology
    of GFCI, it is necessary to understand some basic facts about the FGES and FCI causal_algorithms.
    The FGES algorithm is used to improve the accuracy of both the adjacency phase and the orientation
    phase of FCI by providing a more accurate initial graph that contains a subset of both the
    non-adjacencies and orientations of the final output of FCI. The initial set of nonadjacencies
    given by FGES is augmented by FCI performing a set of conditional independence tests that lead
    to the removal of some further adjacencies whenever a conditioning set is found that makes two
    adjacent variables independent. After the adjacency phase of FCI, some of the orientations of F
    GES are then used to provide an initial orientation of the undirected graph that is then
    augmented by the orientation phase of FCI to provide additional orientations.
    """


    def __init__(self):
        pass

    @staticmethod
    def run_continuous(df, pc=None):
        """
        Run the algorithm against a continuous dataframe to return a dot format causal graph.
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
            tetrad.run(algoId='gfci',
                       dfs=df,
                       testId='fisher-z-test',
                       scoreId='sem-bic',
                       maxDegree=-1,
                       maxPathLength=-1,
                       completeRuleSetUsed=False,
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
            tetrad.run(algoId='gfci',
                       dfs=df,
                       testId='bdeu-test',
                       scoreId='bdeu-score',
                       dataType=AlgorithmConstants.DISCRETE,
                       maxDegree=3,
                       maxPathLength=-1,
                       completeRuleSetUsed=False,
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
            tetrad.run(algoId='gfci',
                       dfs=df,
                       testId='cg-lr-test',
                       scoreId='cg-bic-score',
                       dataType=AlgorithmConstants.MIXED,
                       numCategoriesToDiscretize=4,
                       maxDegree=3,
                       maxPathLength=-1,
                       completeRuleSetUsed=False,
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
