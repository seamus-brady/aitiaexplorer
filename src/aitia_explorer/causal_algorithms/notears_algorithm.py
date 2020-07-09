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
import aitia_explorer.causal_algorithms.notears as notears

from aitia_explorer.causal_algorithms.algorithm_constants import AlgorithmConstants

_logger = logging.getLogger(__name__)


class NOTEARSAlgorithm():
    """
    Based on https://github.com/jmoss20/notears
    a python package implementing "DAGs with NO TEARS: Smooth Optimization for Structure Learning",
    Xun Zheng, Bryon Aragam, Pradeem Ravikumar and Eric P. Xing (March 2018, arXiv:1803.01422)

    Unfortunately this implementation contains no support for labelled nodes which makes it less useful
    for drawing causal graphs. This returns an adjacency matrix instead.
    """

    def __init__(self):
        pass

    @staticmethod
    def run(df, pc=None):
        """
        Run the algorithm against the dataframe to return a dot format causal graph.
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
            return acyclic_W
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str
