"""
TBD Header
"""
import logging

import networkx as nx
import pandas as pd
from pycausal.pycausal import pycausal

from aitia_explorer.metrics.graph_metrics import GraphMetrics
from aitia_explorer.algorithm_runner import AlgorithmRunner
from aitia_explorer.target_data.loader import TargetData
from aitia_explorer.util.graph_util import GraphUtil

_logger = logging.getLogger(__name__)


class App():
    """
    The main AitiaExplorer app entry point.
    """
    algo_runner = AlgorithmRunner()
    graph_metrics = GraphMetrics()
    graph_util = GraphUtil()
    data = TargetData()

    def __init__(self):
        self.vm_running = False

    def run_analysis(self, df, algorithm_list=None, target_graph_str=None, pc=None):
        """
        Runs an analysis on the supplied dataframe.
        This can take a PyCausalWrapper if multiple runs are being done.
        :param target_graph:
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

        algo_list = algorithm_list
        if algo_list is None:
            algo_list = self.algo_runner.get_all_algorithms()

        for algo in algo_list:
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
                causal_graph = self.graph_util.get_causal_graph_from_dot(dot_str)
                single_result['causal_graph'] = causal_graph
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

        df_results = self.get_result_dataframe(analysis_results_filtered, target_graph_str)

        return analysis_results_filtered, df_results

    def get_result_dataframe(self, analysis_results_filtered, target_graph_str):
        """
        Provides a dataframe with the analysis results.
        :param analysis_results_filtered:
        :param target_graph_str:
        :return:
        """
        df_results = pd.DataFrame(
            columns=('Algorithm',
                     'AURC',
                     'SHD'))

        target_nxgraph = None
        if target_graph_str is not None:
            target_nxgraph = self.graph_util.get_nxgraph_from_dot(target_graph_str)

        for result in analysis_results_filtered:
            if result['dot_str'] is not None and result['causal_graph'] is not None:
                pred_graph = self.graph_util.get_nxgraph_from_dot(result['dot_str'])
                if target_nxgraph is not None:
                    prec_recall = self.graph_metrics.precision_recall(target_nxgraph, pred_graph)[0]
                    shd = self.graph_metrics.SHD(target_nxgraph, pred_graph)
                else:
                    prec_recall = 0
                    shd = 0
                new_row = {'Algorithm': result['algo_name'],
                           'AURC': prec_recall,
                           'SHD': shd
                           }
                df_results = df_results.append(new_row, ignore_index=True)
        return df_results
