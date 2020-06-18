"""
TBD Header
"""
import logging

import networkx as nx
import pandas as pd
from pycausal.pycausal import pycausal

from aitia_explorer.metrics.graph_metrics import GraphMetrics
from aitia_explorer.py_causal_wrapper import PyCausalUtil
from aitia_explorer.util.graph_util import GraphUtil

_logger = logging.getLogger(__name__)


class App():
    """
    The main AitiaExplorer app entry point.
    """
    pc_util = PyCausalUtil()

    def __init__(self):
        self.vm_running = False

    def run_analysis(self, df, target_graph_str=None, pc=None):
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
                     'Ind Relations Diff',
                     'Isomorphic to Target?',
                     'AURC',
                     'SHD'))
        metrics = GraphMetrics()
        target_nxgraph = None
        if target_graph_str is not None:
            causal_graph_target = GraphUtil.get_causal_graph_from_dot(target_graph_str)
            target_ind_rels = len(causal_graph_target.get_all_independence_relationships())
            target_nxgraph = GraphUtil.get_nxgraph_from_dot(target_graph_str)
        else:
            target_ind_rels = 0
        for result in analysis_results_filtered:
            if result['dot_str'] is not None and result['causal_graph'] is not None:
                pred_graph = GraphUtil.get_nxgraph_from_dot(result['dot_str'])
                if target_nxgraph is not None:
                    prec_recall = metrics.precision_recall(target_nxgraph, pred_graph)[0]
                    shd = metrics.SHD(target_nxgraph, pred_graph)
                    isomorphic = nx.is_isomorphic(target_nxgraph, pred_graph)
                    ind_rel_diff = target_ind_rels - result['num_ind_rel']
                else:
                    ind_rel_diff = result['num_ind_rel']
                    prec_recall = 0
                    shd = 0
                    isomorphic = 'NA'
                new_row = {'Algorithm': result['algo_name'],
                           'Ind Relations Diff': ind_rel_diff,
                           'Isomorphic to Target?': isomorphic,
                           'AURC': prec_recall,
                           'SHD': shd
                           }
                df_results = df_results.append(new_row, ignore_index=True)
        return df_results
