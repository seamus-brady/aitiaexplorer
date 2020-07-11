#
# This file is part of AitiaExplorer and is released under the FreeBSD License.
#
# Copyright (c) 2020, Seamus Brady <seamus@corvideon.ie>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.#

import logging

import pandas as pd
from pycausal.pycausal import pycausal

from aitia_explorer.algorithm_runner import AlgorithmRunner
from aitia_explorer.entities.analysis_results import AnalysisResults
from aitia_explorer.entities.single_analysis_result import SingleAnalysisResult
from aitia_explorer.feature_reduction.bayesian_gaussian_mixture_wrapper import BayesianGaussianMixtureWrapper
from aitia_explorer.feature_selection_runner import FeatureSelectionRunner
from aitia_explorer.metrics.graph_metrics import GraphMetrics
from aitia_explorer.target_data.loader import TargetData
from aitia_explorer.util.graph_util import GraphUtil

_logger = logging.getLogger(__name__)


class App:
    """
    The main AitiaExplorer app entry point.
    """

    algo_runner = AlgorithmRunner()
    feature_selection = FeatureSelectionRunner()
    graph_metrics = GraphMetrics()
    graph_util = GraphUtil()
    data = TargetData()
    bgmm = BayesianGaussianMixtureWrapper()

    def __init__(self):
        self.vm_running = False

    def run_analysis_with_high_low(self,
                                   incoming_df,
                                   target_graph_str=None,
                                   feature_high=None,
                                   feature_low=None,
                                   feature_selection_list=None,
                                   algorithm_list=None,
                                   pc=None,
                                   verbose=True
                                   ):
        """
        Runs the entire analysis with feature selection and causal discovery but
        also records the SHD for each run and returns the result that minimises SHD.
        """
        if feature_high is None:
            # just default to number of features in dataframe
            feature_high = len(list(incoming_df))

        if feature_high is None:
            # just default to 1
            feature_low = 1

        if target_graph_str is None:
            # no target graph has been supplied, so let's create an approximation
            # using the hill climbing algorithm
            if verbose:
                print("No target graph has been supplied.")
                print("The system will generate an approximate target graph using the greedy hill climbing algorithm.")
            target_graph_str = self.algo_runner.algo_hill_climber(incoming_df)

        # to store all the results
        results_dict = dict()
        all_results_df = pd.DataFrame()

        # this is just so we can pass it back
        returned_target_graph = None

        for i in range(feature_high, feature_low, -1):

            if verbose:
                print("-----------------------------------------------")
                print("Starting analysis with {0} features...".format(i))

            # get current run results
            result_obj, results_df, returned_target_graph = self.run_analysis(
                         incoming_df,
                         target_graph_str=target_graph_str,
                         n_features=i,
                         feature_selection_list=feature_selection_list,
                         algorithm_list=algorithm_list,
                         pc=pc,
                         verbose=verbose)

            results_dict[i] = (result_obj, results_df)
            all_results_df = all_results_df.append(results_df,  ignore_index=True)

            if verbose:
                print("Completed analysis with {0} features...".format(i))

        # now we need to figure out the lowest SHD
        shd_results_dict = dict()

        # get the minimum shd for each run
        for k, v in results_dict.items():
            # dict holds a tuple, second value is df
            results_df = v[1]
            minimum_shd = results_df['SHD'].min()
            shd_results_dict[k] = minimum_shd

        # sort the shd
        shd_tuple_list = sorted(shd_results_dict.items(), key=lambda x: x[1])

        # first results are the best, first value in tuple is feature no / index
        i = shd_tuple_list[0][0]

        if verbose:
            print("All done!")
            print("The results with the lowest SHD have been returned.")

        # return the results from the results dict
        # --> results_obj, result_df, target_graph, all_results_df
        return results_dict[i][0], results_dict[i][1], returned_target_graph, all_results_df

    def run_analysis(self,
                     incoming_df,
                     target_graph_str=None,
                     n_features=None,
                     feature_selection_list=None,
                     algorithm_list=None,
                     pc=None,
                     verbose=True):
        """
        Runs the entire analysis with feature selection and causal discovery.
        """

        if n_features is None:
            # just default to number of features in dataframe
            n_features = len(list(incoming_df))

        if target_graph_str is None:
            # no target graph has been supplied, so let's create an approximation
            # using the hill climbing algorithm
            if verbose:
                print("No target graph has been supplied.")
                print("The system will generate an approximate target graph using the greedy hill climbing algorithm.")
            target_graph_str = self.algo_runner.algo_hill_climber(incoming_df)

        feature_selection_list = self._get_feature_selection_algorithms(feature_selection_list)

        amalgamated_analysis_results = []

        for feature_selection in feature_selection_list:
            # get the actual function
            feature_func = feature_selection[1]

            # get the feature list from the function
            features = feature_func(incoming_df, n_features)

            if verbose:
                print("Running causal discovery on features selected by {0}".format(feature_selection[0]))

            # get the reduced dataframe
            df_reduced, requested_features = self.get_reduced_dataframe(incoming_df, features)

            # check to see if this reduced dataframe has introduced unobserved latent edges
            latent_edges = []
            latent_edges.extend(self.algo_runner.algo_miic(df_reduced))

            if verbose:
                print("There are {0} latent edges in the reduced dataset".format(len(latent_edges)))

            analysis_results = self._run_causal_algorithms(df_reduced,
                                                           feature_selection_method=feature_selection[0],
                                                           requested_features=requested_features,
                                                           n_features=n_features,
                                                           target_graph_str=target_graph_str,
                                                           algorithm_list=algorithm_list,
                                                           latent_edges=latent_edges,
                                                           pc=pc,
                                                           verbose=verbose)

            if verbose:
                print("Completed causal discovery on features selected by {0}".format(feature_selection[0]))

            amalgamated_analysis_results.append(analysis_results)

        if verbose:
            print("Completed analysis.")

        # we need to flatten all the results
        amalgamated_list_of_dicts = []
        final_results = []
        for results in amalgamated_analysis_results:
            for result in results.results:
                # append as dict for the dataframe output
                amalgamated_list_of_dicts.append(result.asdict())
                # flatten the results
                final_results.append(result)

        # generate the target graph for the user
        target_graph = self.graph_util.get_causal_graph_from_dot(target_graph_str)

        return final_results, pd.DataFrame(amalgamated_list_of_dicts), target_graph

    def run_causal_discovery(self, df, target_graph_str, algorithm_list, pc):
        """
        Runs the causal discovery.
        """
        analysis_results = self._run_causal_algorithms(df,
                                                       target_graph_str=target_graph_str,
                                                       algorithm_list=algorithm_list,
                                                       pc=pc)
        return analysis_results, analysis_results.to_dataframe()

    def get_reduced_dataframe(self, incoming_df, feature_indices, sample_with_gmm=False):
        """
        A wrapper call for the BayesianGaussianMixtureWrapper :)
        """
        bgmm = BayesianGaussianMixtureWrapper()
        return bgmm.get_reduced_dataframe(incoming_df, feature_indices, sample_with_gmm)

    def _run_causal_algorithms(self,
                               incoming_df,
                               requested_features=None,
                               feature_selection_method=None,
                               n_features=None,
                               algorithm_list=None,
                               target_graph_str=None,
                               latent_edges=[],
                               pc=None,
                               verbose=True):
        """
        Runs an analysis on the supplied dataframe.
        This can take a PyCausalWrapper if multiple runs are being done.
        """
        analysis_results = AnalysisResults()
        pc_supplied = True

        # get py-causal if needed
        if pc is None:
            pc_supplied = False
            pc = pycausal()
            pc.start_vm()

        algo_list = self._get_causal_algorithms(algorithm_list)

        for algo in algo_list:
            # dict to store run result
            analysis_result = SingleAnalysisResult()
            analysis_result.feature_selection_method = feature_selection_method
            analysis_result.feature_list = requested_features
            analysis_result.num_features_requested = n_features
            analysis_result.causal_algorithm = algo[0]
            analysis_result.latent_edges = latent_edges

            if verbose:
                print("Running causal discovery using {0}".format(algo[0]))

            # get the graph from the algo
            algo_fn = algo[1]
            dot_str = self._discover_graph(algo_fn, incoming_df, pc)

            # store the dot graph
            analysis_result.dot_format_string = dot_str

            # convert the causal graph
            if dot_str is not None:
                causal_graph = self.graph_util.get_causal_graph_from_dot(dot_str)
                analysis_result.causal_graph = causal_graph
                nx_graph = self.graph_util.get_digraph_from_dot(dot_str)
                analysis_result.causal_graph_with_latent_edges = \
                    self.graph_util.get_causal_graph_with_latent_edges(nx_graph, latent_edges)

            analysis_results.results.append(analysis_result)

        # shutdown the java vm if needed
        if not pc_supplied:
            pc.stop_vm()

        # filter the results
        analysis_results_filtered = self._filter_empty_results(analysis_results)

        # add the causal metrics
        updated_analysis_results = self._add_causal_metrics(analysis_results_filtered, target_graph_str)

        return updated_analysis_results

    def _discover_graph(self, algo_fn, df, pc):
        """
        Siscover the graph using the supplied algorithm function.
        """
        dot_str = algo_fn(df, pc)
        return dot_str

    def _filter_empty_results(self, incoming_results):
        filtered_results = AnalysisResults()
        for result in incoming_results.results:
            if result.causal_graph is not None:
                filtered_results.results.append(result)
        return filtered_results

    def _get_feature_selection_algorithms(self, feature_selection_list):
        """
        Gets the list of feature selection algorithms to run.
        """
        algo_list = feature_selection_list
        if algo_list is None:
            algo_list = self.feature_selection.get_all_feature_selection_algorithms()
        return algo_list

    def _get_causal_algorithms(self, algorithm_list):
        """
        Gets the list of causal algorithms to run.
        """
        algo_list = algorithm_list
        if algo_list is None:
            algo_list = self.algo_runner.get_all_causal_algorithms()
        return algo_list

    def _add_causal_metrics(self, incoming_analysis_results, target_graph_str):
        """
        Provides the causal analysis results.
        """
        return_analysis_results = AnalysisResults()
        target_nxgraph = None
        if target_graph_str is not None:
            target_nxgraph = self.graph_util.get_nxgraph_from_dot(target_graph_str)

        for result in incoming_analysis_results.results:
            if result.dot_format_string is not None \
                    and result.causal_graph is not None:
                pred_graph = self.graph_util.get_nxgraph_from_dot(result.dot_format_string)
                if target_nxgraph is not None:
                    prec_recall = self.graph_metrics.precision_recall(target_nxgraph, pred_graph)[0]
                    shd = self.graph_metrics.SHD(target_nxgraph, pred_graph)
                else:
                    prec_recall = 0
                    shd = 0
                result.AUPRC = prec_recall
                result.SHD = shd
            return_analysis_results.results.append(result)
        return return_analysis_results
