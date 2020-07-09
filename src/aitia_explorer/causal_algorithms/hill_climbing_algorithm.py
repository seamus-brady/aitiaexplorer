
import logging
import pyAgrum as gum
import tempfile

from aitia_explorer.causal_algorithms.algorithm_constants import AlgorithmConstants
from aitia_explorer.util.graph_util import GraphUtil

_logger = logging.getLogger(__name__)


class HillClimbingAlgorithm():
    """
    Uses a greedy hill climbing algorithm to approximate the underlying bayes net. It won't be perfect but
    it will be good enough to provide a heuristic when the known graph is missing or absent.
    """

    def __init__(self):
        pass

    @staticmethod
    def run(df, pc=None):
        """
        Run the algorithm against the dataframe to return a dot string.
        """
        dot_str = None
        try:
            fp = tempfile.NamedTemporaryFile(suffix='.csv')
            df.to_csv(fp.name, encoding='utf-8', index=False)
            learner = gum.BNLearner(fp.name)
            learner.useGreedyHillClimbing()
            bn = learner.learnBN()
            return bn.toDot()
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str
