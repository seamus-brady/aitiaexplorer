"""
TBD Header
"""
import logging

from pycausal import search as s
from pycausal.pycausal import pycausal

_logger = logging.getLogger(__name__)


class FCIAlgorithm():
    """
    The FCI algorithm is a constraint-based algorithm that takes as input
    sample data and optional background knowledge and in the large sample
    limit outputs an equivalence class of CBNs that (including those with
    hidden confounders) that entail the set of conditional independence
    relations judged to hold in the population. It is limited to several
    thousand variables, and on realistic sample sizes it is inaccurate
    in both adjacencies and orientations. FCI has two phases:
    an adjacency phase and an orientation phase. The adjacency phase
    of the algorithm starts with a complete undirected graph and
    then performs a sequence of conditional independence tests that
    lead to the removal of an edge between any two adjacent variables
    that are judged to be independent, conditional on some subset of the
    observed variables; any conditioning set that leads to the removal of
    an adjacency is stored. After the adjacency phase, the resulting
    undirected graph has the correct set of adjacencies, but all of the
    edges are unoriented. FCI then enters an orientation phase that uses
    the stored conditioning sets that led to the removal of adjacencies
    to orient as many of the edges as possible. See [Spirtes, 1993].
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
        single_run = False
        dot_str = None
        try:
            # start java vm and get algo runner
            if pc is None:
                pc = pycausal()
                pc.start_vm()
                single_run = True

            tetrad = s.tetradrunner()
            tetrad.run(algoId='fci',
                       dfs=df,
                       testId='fisher-z-test',
                       depth=-1,
                       maxPathLength=-1,
                       completeRuleSetUsed=False,
                       verbose=True)
            graph = tetrad.getTetradGraph()
            dot_str = pc.tetradGraphToDot(graph)

            # shutdown java vm
            if single_run:
                pc.stop_vm()
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str
