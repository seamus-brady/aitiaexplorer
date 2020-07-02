"""
TBD Header
"""
import logging
from pycausal.pycausal import pycausal
from pycausal import search as s

from aitia_explorer.causal_algorithms.algorithm_constants import AlgorithmConstants

_logger = logging.getLogger(__name__)


class BayesEstAlgorithm():
    """
    bayesEst is the revised Greedy Equivalence Search (GES) algorithm developed
    by Joseph D. Ramsey, Director of Research Computing, Department of Philosophy,
    Carnegie Mellon University, Pittsburgh, PA.
    """

    def __init__(self):
        pass

    @staticmethod
    def run(df, pc=None):
        """
        Run the algorithm against the dataframe to return a dot format causal graph.
        :param df: dataframe
        :return: dot graph string
        """
        dot_str = None
        try:
            # start java vm and get algo runner
            if pc is None:
                pc = pycausal()
                pc.start_vm()

            bayes_est = s.bayesEst(df, depth=-1, alpha=0.05, verbose=AlgorithmConstants.VERBOSE)
            graph = bayes_est.getTetradGraph()
            dot_str = pc.tetradGraphToDot(graph)

            # stop java vm
            if pc is None:
                pc.stop_vm()
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str
