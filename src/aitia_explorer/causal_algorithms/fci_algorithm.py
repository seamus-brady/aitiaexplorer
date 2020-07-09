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

from pycausal import search as s
from pycausal.pycausal import pycausal

from aitia_explorer.causal_algorithms.algorithm_constants import AlgorithmConstants

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
