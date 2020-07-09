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

import networkx as nx
import pygraphviz
from causalgraphicalmodels import CausalGraphicalModel

_logger = logging.getLogger(__name__)


class GraphUtil():
    """
    Graph utility class.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_dot_from_nxgraph(nx_graph):
        """
        Create a dot string from an nxgraph
        """
        return nx.nx_pydot.to_pydot(nx_graph)


    @staticmethod
    def get_nxgraph_from_adjacency_matrix(adjacency_matrix):
        """
        Create an nxgraph from an adjacency_matrix (numpy array).
        """
        nx_graph = nx.DiGraph(nx.from_numpy_matrix(adjacency_matrix))
        return nx_graph

    @staticmethod
    def get_nxgraph_from_dot(dot_str):
        """
        Create an nxgraph from a dot string.
        """
        nx_graph = nx.drawing.nx_agraph.from_agraph(pygraphviz.AGraph(dot_str))
        return nx_graph

    @staticmethod
    def get_digraph_from_dot(dot_str):
        """
        Loads a DiGraph from a dot string.
        """

        # load graph from dot data
        nx_graph = GraphUtil.get_nxgraph_from_dot(dot_str)
        nx_graph = nx.DiGraph(nx_graph)
        return nx_graph

    @staticmethod
    def get_causal_graph_from_dot(dot_str):
        """
        Create a CausalGraphicalModel from a dot string.
        """

        # load graph from dot data
        nx_graph = GraphUtil.get_nxgraph_from_dot(dot_str)

        # create a causal graph
        return GraphUtil.get_causal_graph_from_nxgraph(nx_graph)

    @staticmethod
    def get_causal_graph_from_nxgraph(nx_graph):
        """
        Create a CausalGraphicalModel from an nxgraph.
        """
        # create a causal graph
        causal_graph = CausalGraphicalModel(
            nodes=nx_graph.nodes(),
            edges=nx_graph.edges()
        )

        return causal_graph

    @staticmethod
    def get_causal_graph_with_latent_edges(nx_graph, incoming_latent_edges):
        """
        Create a CausalGraphicalModel from an nxgraph with unobserved latent edges.
        """

        # only add edges with where one of the nodes exist
        latent_edges = []
        for l in incoming_latent_edges:
            node1 = l[0]
            node2 = l[1]
            if node1 in nx_graph.nodes() or node2 in nx_graph.nodes():
                latent_edges.append(l)
                
        # create a causal graph
        causal_graph = CausalGraphicalModel(
            nodes=nx_graph.nodes(),
            edges=nx_graph.edges(),
            latent_edges=latent_edges
        )

        return causal_graph

    @staticmethod
    def get_causal_graph_from_bif(bif_reader):
        """
        Create a CausalGraphicalModel from an bif file.
        """
        # create a causal graph
        causal_graph = CausalGraphicalModel(
            nodes=bif_reader.get_variables(),
            edges=bif_reader.get_edges()
        )

        return causal_graph
