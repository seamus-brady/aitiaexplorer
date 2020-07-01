"""
TBD Header
"""
import logging
import networkx as nx
import aitia_explorer.algorithms.notears as notears

from aitia_explorer.algorithms.algorithm_constants import AlgorithmConstants

_logger = logging.getLogger(__name__)


class NOTEARSAlgorithm():
    """
    Based on https://github.com/jmoss20/notears
    a python package implementing "DAGs with NO TEARS: Smooth Optimization for Structure Learning",
    Xun Zheng, Bryon Aragam, Pradeem Ravikumar and Eric P. Xing (March 2018, arXiv:1803.01422)
    """

    def __init__(self):
        pass

    @staticmethod
    def run(df, pc=None):
        """
        Run the algorithm against the dataframe to return a dot format causal graph.
        :param pc: Not used, kept for api consistency
        :param df: dataframe
        :return: dot graph string
        """
        dot_str = None
        try:
            output_dict = notears.run(notears.notears_standard,
                                      df,
                                      notears.loss.least_squares_loss,
                                      notears.loss.least_squares_loss_grad,
                                      e=1e-8,
                                      verbose=AlgorithmConstants.VERBOSE)

            # an acyclic graph can be recovered by removing the lowest weighted edges
            # (in magnitude) until the remaining graph is acyclic
            acyclic_W = notears.utils.threshold_output(output_dict['W'])

            # convert to nx graph
            nx_graph =  nx.DiGraph(nx.from_numpy_matrix(acyclic_W))
            return nx.nx_pydot.to_pydot(nx_graph)
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str
