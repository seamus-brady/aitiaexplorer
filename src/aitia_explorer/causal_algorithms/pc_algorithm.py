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


class PCAlgorithm():
    """
    PC algorithm (Spirtes and Glymour, Social Science Computer Review, 1991) is a
    pattern search which assumes that the underlying causal structure of the input
    data is acyclic, and that no two variables are caused by the same latent
    (unmeasured) variable. In addition, it is assumed that the input data set is
    either entirely continuous or entirely discrete; if the data set is continuous,
    it is assumed that the causal relation between any two variables is linear,
    and that the distribution of each variable is Normal. Finally, the sample
    should ideally be i.i.d.. Simulations show that PC and several of the other
    causal_algorithms described here often succeed when these assumptions, needed to
    prove their correctness, do not strictly hold. The PC algorithm will sometimes
    output double headed edges. In the large sample limit, double headed edges
    in the output indicate that the adjacent variables have an unrecorded common
    cause, but PC tends to produce false positive double headed edges on small samples.
    The PC algorithm is correct whenever decision procedures for independence and
    conditional independence are available. The procedure conducts a sequence of
    independence and conditional independence tests, and efficiently builds a
    pattern from the results of those tests. As implemented in TETRAD, PC is
    intended for multinomial and approximately Normal distributions with
    i.i.d. data. The tests have an alpha value for rejecting the null hypothesis,
    which is always a hypothesis of independence or conditional independence.
    For continuous variables, PC uses tests of zero correlation or zero partial
    correlation for independence or conditional independence respectively.
    For discrete or categorical variables, PC uses either a chi square or a
    g square test of independence or conditional independence
    (see Causation, Prediction, and Search for details on tests).
    In either case, the tests require an alpha value for rejecting the
    null hypothesis, which can be adjusted by the user. The procedures make
    no adjustment for multiple testing.
    (For PC, CPC, JPC, JCPC, FCI, all testing searches.)
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
            tetrad.run(algoId='pc-all',
                       dfs=df,
                       testId='fisher-z-test',
                       fasRule=2,
                       depth=2,
                       conflictRule=1,
                       concurrentFAS=True,
                       useMaxPOrientationHeuristic=True,
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
