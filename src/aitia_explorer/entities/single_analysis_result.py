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

_logger = logging.getLogger(__name__)


class SingleAnalysisResult():
    """
    A class for handling results from one causal analysis run.
    """

    def __init__(self):
        self.causal_algorithm = None
        self.feature_selection_method = None
        self.feature_list = None
        self.num_features_requested = None
        self.dot_format_string = None
        self.causal_graph = None
        self.causal_graph_with_latent_edges = None
        self.latent_edges = []
        self.AUPRC = None
        self.SHD = None


    def asdict(self):
        """
        For displaying in notebooks...
        :return:
        """
        return {'No. of Features Req.': self.num_features_requested,
                'Causal Algorithm': self.causal_algorithm,
                'Feature Selection Method': self.feature_selection_method,
                # leaving this out as it contain too much information
                #'feature list': self.feature_list,
                'AUPRC': self.AUPRC,
                'SHD': self.SHD
                }