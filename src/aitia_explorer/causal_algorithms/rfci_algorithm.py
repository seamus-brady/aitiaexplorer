
import logging

from pycausal import search as s
from pycausal.pycausal import pycausal

from aitia_explorer.causal_algorithms.algorithm_constants import AlgorithmConstants

_logger = logging.getLogger(__name__)


class RFCIAlgorithm():
    """
    A modification of the FCI algorithm in which some expensive steps are finessed and the
    output is somewhat differently interpreted. In most cases this runs faster than FCI
    (which can be slow in some steps) and is almost as informative. See Colombo et al., 2012.
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
            tetrad.run(algoId='rfci',
                       dfs=df,
                       testId='fisher-z-test',
                       depth=-1,
                       maxPathLength=-1,
                       completeRuleSetUsed=False,
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
            tetrad.run(algoId='rfci',
                       dfs=df,
                       testId='chi-square-test',
                       dataType=AlgorithmConstants.DISCRETE,
                       depth=3,
                       maxPathLength=-1,
                       completeRuleSetUsed=True,
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
            tetrad.run(algoId='rfci',
                       dfs=df,
                       testId='cg-lr-test',
                       dataType=AlgorithmConstants.MIXED,
                       numCategoriesToDiscretize=4,
                       depth=-1,
                       maxPathLength=-1,
                       discretize=False,
                       completeRuleSetUsed=False,
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
