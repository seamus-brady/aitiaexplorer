"""
TBD Header
"""
import logging
import os
import pandas as pd
from pycausal.pycausal import pycausal
from aitia_explorer.py_causal_wrapper import PyCausalUtil
from aitia_explorer.util.graph_util import GraphUtil

_logger = logging.getLogger(__name__)


class App():
    """
    The main AitiaExplorer app entry point.
    """
    pc_util = PyCausalUtil()

    def __init__(self):
        pass

    def run_analysis(self, df, pc=None):
        """
        Runs an analysis on the supplied dataframe.
        This can take a PyCausalWrapper if multiple runs are being done.
        :param df: dataframe
        :param pc: pycausal wrapper
        :return: list of results
        """
        analysis_results = []
        pc_supplied = True

        # get py-causal if needed
        if pc is None:
            pc_supplied = False
            pc = pycausal()
            pc.start_vm()

        for algo in self.pc_util.get_all_algorithms():
            # dict to store run result
            single_result = dict()
            single_result['algo_name'] = algo[0]

            # discover the graph using algo
            algo_func = algo[1]
            dot_str = algo_func(df, pc)

            # store the dot graph
            single_result['dot_str'] = dot_str

            # get the causal graph
            if dot_str is not None:
                causal_graph = GraphUtil.get_causal_graph_from_dot(dot_str)
                single_result['causal_graph'] = causal_graph
                single_result['num_ind_rel'] = len(causal_graph.get_all_independence_relationships())
            else:
                single_result['causal_graph'] = None

            analysis_results.append(single_result)

        # shutdown the java vm if needed
        if not pc_supplied:
            pc.stop_vm()

        # filter the results
        analysis_results_filtered = analysis_results.copy()
        for result in analysis_results:
            if result['causal_graph'] is None:
                analysis_results_filtered.remove(result)

        return analysis_results_filtered
