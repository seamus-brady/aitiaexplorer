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
"""
TBD
"""
import logging
import os
import random

import numpy as np
import pandas as pd
from pgmpy.readwrite import BIFReader
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from causalgraphicalmodels import StructuralCausalModel
from causalgraphicalmodels.csm import logistic_model, linear_model

from aitia_explorer.util.graph_util import GraphUtil

_logger = logging.getLogger(__name__)


class TargetData:
    """
    A class that provides known causal data for targeting in tests.
    """

    def __init__(self):
        pass

    @staticmethod
    def data_dir():
        """
        Get the data dir.
        :return: dir.
        """
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        return data_dir

    @staticmethod
    def graphs_dir():
        """
        Get the graph dir.
        :return: dir
        """
        graphs_dir = os.path.join(os.path.dirname(__file__), 'graphs')
        return graphs_dir

    @staticmethod
    def simulated_data_1_graph():
        """
        Simulated graph produced by Tetrad.
        See http://www.phil.cmu.edu/tetrad/
        :return: graph string in dot format
        """
        graph_path = os.path.join(TargetData.graphs_dir(), "simulated_data_graph_1.dot")
        with open(graph_path, 'r') as dot_file:
            simulated_data_graph = dot_file.read()
        return simulated_data_graph

    @staticmethod
    def hepar2_graph():
        """
        HEPAR II graph.
        Loads this from a BIF file.
        See hepar2_graph for the associated causal graph.
        See https://www.ccd.pitt.edu/wiki/index.php/Data_Repository
        :return: graph string in dot format
        """
        graph_path = os.path.join(TargetData.graphs_dir(), "hepar2.bif")
        reader = BIFReader(graph_path)
        csm = GraphUtil.get_causal_graph_from_bif(reader)
        return str(csm.draw())

    @staticmethod
    def simulated_data_1():
        """
        Simulated data produced by Tetrad.
        See http://www.phil.cmu.edu/tetrad/
        :return: dataframe
        """
        data_dir = os.path.join(TargetData.data_dir(), "simulated_data_1.txt")
        simulated_data = pd.read_table(data_dir, sep="\t")
        return simulated_data

    @staticmethod
    def audiology_data():
        """
        Simulated data provided with Tetrad.
        See http://www.phil.cmu.edu/tetrad/
        :return: dataframe
        """
        data_dir = os.path.join(TargetData.data_dir(), "audiology.txt")
        return pd.read_table(data_dir, sep="\t")

    @staticmethod
    def charity_data():
        """
        Simulated data provided with Tetrad.
        See http://www.phil.cmu.edu/tetrad/
        :return: dataframe
        """
        data_dir = os.path.join(TargetData.data_dir(), "charity.txt")
        return pd.read_table(data_dir, sep="\t")

    @staticmethod
    def lucas0_data():
        """
        LUCAS  (LUng CAncer Simple set) contain toy data generated artificially
        by causal Bayesian networks with binary variables.
        See http://www.causality.inf.ethz.ch/data/LUCAS.html
        :return:
        """
        data_dir = os.path.join(TargetData.data_dir(), "lucas0_train.csv")
        return pd.read_csv(data_dir)

    @staticmethod
    def lucas2_data():
        """
        LUCAS  (LUng CAncer Simple set) contain toy data generated artificially
        by causal Bayesian networks with binary variables.
        See http://www.causality.inf.ethz.ch/data/LUCAS.html
        :return:
        """
        data_dir = os.path.join(TargetData.data_dir(), "lucas0_train.csv")
        return pd.read_csv(data_dir)

    @staticmethod
    def hepar2_10k_data():
        """
        HEPAR II dataset - full 10k dataset.
        See hepar2_graph for the associated causal graph.
        See https://www.ccd.pitt.edu/wiki/index.php/Data_Repository
        :return: scaled dataframe
        """
        # load the data
        data_dir = os.path.join(TargetData.data_dir(), "HEPARTWO10k.csv")
        df = pd.read_csv(data_dir)

        # Create a label (category) encoder object
        le = preprocessing.LabelEncoder()
        df_encoded = df.apply(le.fit_transform)

        # scale the data
        df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_encoded), columns=list(df))

        return df_scaled

    @staticmethod
    def hepar2_100_data():
        """
        HEPAR II dataset - first 100 rows from dataset.
        See hepar2_graph for the associated causal graph.
        See https://www.ccd.pitt.edu/wiki/index.php/Data_Repository
        :return: scaled dataframe
        """
        # load the data
        data_dir = os.path.join(TargetData.data_dir(), "HEPARTWO100.csv")
        df = pd.read_csv(data_dir)

        # Create a label (category) encoder object
        le = preprocessing.LabelEncoder()
        df_encoded = df.apply(le.fit_transform)

        # scale the data
        df_scaled = pd.DataFrame(StandardScaler().fit_transform(df_encoded),  columns=list(df))

        return df_scaled

    @staticmethod
    def scm1():
        """
        Returns a StructuralCausalModel for sampling.
        See https://github.com/ijmbarr/causalgraphicalmodels/blob/master/causalgraphicalmodels/examples.py
        :return:
        """
        return StructuralCausalModel({
            'a': lambda n_samples: np.random.normal(size=n_samples),
            'b': lambda n_samples: np.random.normal(size=n_samples),
            'x': logistic_model(['a', 'b'], [-1, 1]),
            'c': linear_model(['x'], [1]),
            'y': linear_model(['c', 'e'], [3, -1]),
            'd': linear_model(['b'], [-1]),
            'e': linear_model(['d'], [2]),
            'f': linear_model(['y'], [0.7]),
            'h': linear_model(['y', 'a'], [1.3, 2.1])
        })

    @staticmethod
    def scm2():
        """
        Returns a StructuralCausalModel for sampling.
        See https://github.com/ijmbarr/causalgraphicalmodels/blob/master/causalgraphicalmodels/examples.py
        :return: StructuralCausalModel
        """
        return StructuralCausalModel({
            'a': lambda n_samples: np.random.normal(size=n_samples),
            'b': lambda n_samples: np.random.normal(size=n_samples),
            'c': logistic_model(['a', 'b'], [-1, 1]),
            'd': logistic_model(['c', 'b'], [-1, 1]),
            'e': lambda n_samples: np.random.normal(size=n_samples),
            'f': lambda n_samples: np.random.normal(size=n_samples),
            'g': linear_model(['f'], [0.7]),
            'h': linear_model(['f'], [0.9]),
        })

    @staticmethod
    def scm3():
        """
        Returns a StructuralCausalModel for sampling.
        See https://github.com/ijmbarr/causalgraphicalmodels/blob/master/causalgraphicalmodels/examples.py
        :return:
        """
        return StructuralCausalModel({
            'a': lambda n_samples: np.random.normal(size=n_samples),
            'b': lambda n_samples: np.random.normal(size=n_samples),
            'c': lambda n_samples: np.random.normal(size=n_samples),
            'd': lambda n_samples: np.random.normal(size=n_samples),
            'e': lambda n_samples: np.random.normal(size=n_samples),
            'f': logistic_model(['c', 'b'], [-1, 1]),
            'g': linear_model(['f', 'd'], [0.5, 1.1]),
            'h': linear_model(['g', 'd'], [0.5, 1.1]),
            'i': logistic_model(['a', 'b'], [-1, 1]),
            'j': logistic_model(['c', 'b'], [-1, 1]),
            'k': lambda n_samples: np.random.normal(size=n_samples),
            'l': lambda n_samples: np.random.normal(size=n_samples),
            'm': linear_model(['k', 'l'], [0.8, 1.2]),
            'n': logistic_model(['m', 'b'], [-1, 1]),
        })

    @staticmethod
    def random_scm1():
        """
        Generates a random SCM.
        See https://github.com/ijmbarr/causalgraphicalmodels/blob/master/causalgraphicalmodels/examples.py
        :return: StructuralCausalModel
        """
        return StructuralCausalModel({
            'X1': linear_model(['X20', 'X8'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X2': lambda n_samples: np.random.normal(size=n_samples),
            'X3': lambda n_samples: np.random.normal(size=n_samples),
            'X4': linear_model(['X14'], [random.uniform(0, 1)]),
            'X5': lambda n_samples: np.random.normal(size=n_samples),
            'X6': lambda n_samples: np.random.normal(size=n_samples),
            'X7': linear_model(['X12', 'X9'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X8': lambda n_samples: np.random.normal(size=n_samples),
            'X9': lambda n_samples: np.random.normal(size=n_samples),
            'X10': lambda n_samples: np.random.normal(size=n_samples),
            'X11': linear_model(['X1', 'X5'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X12': linear_model(['X5', 'X6'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X13': linear_model(['X9'], [random.uniform(0, 1)]),
            'X14': linear_model(['X8'], [random.uniform(0, 1)]),
            'X15': linear_model(['X14'], [random.uniform(0, 1)]),
            'X16': linear_model(['X15', 'X2'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X17': linear_model(['X10', 'X11'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X18': linear_model(['X12', 'X2'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X19': linear_model(['X3'], [random.uniform(0, 1)]),
            'X20': linear_model(['X19', 'X6'], [random.uniform(0, 1), random.uniform(0, 1)])
        })

    @staticmethod
    def random_scm2():
        """
        Generates a random SCM.
        See https://github.com/ijmbarr/causalgraphicalmodels/blob/master/causalgraphicalmodels/examples.py
        :return: StructuralCausalModel
        """
        return StructuralCausalModel({
            'X1': lambda n_samples: np.random.normal(size=n_samples),
            'X2': lambda n_samples: np.random.normal(size=n_samples),
            'X3': linear_model(['X24'], [random.uniform(0, 1)]),
            'X4': lambda n_samples: np.random.normal(size=n_samples),
            'X5': lambda n_samples: np.random.normal(size=n_samples),
            'X6': lambda n_samples: np.random.normal(size=n_samples),
            'X7': lambda n_samples: np.random.normal(size=n_samples),
            'X8': lambda n_samples: np.random.normal(size=n_samples),
            'X9': lambda n_samples: np.random.normal(size=n_samples),
            'X10': lambda n_samples: np.random.normal(size=n_samples),
            'X11': linear_model(['X1', 'X5'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X12': logistic_model(['X5', 'X6'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X13': linear_model(['X9','X1'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X14': linear_model(['X8'], [random.uniform(0, 1)]),
            'X15': linear_model(['X14', 'X4', 'X27'],
                                [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]),
            'X16': linear_model(['X15', 'X2'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X17': linear_model(['X10', 'X11'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X18': linear_model(['X12', 'X2'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X19': linear_model(['X3'], [random.uniform(0, 1)]),
            'X20': linear_model(['X19', 'X6'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X21': linear_model(['X8'], [random.uniform(0, 1)]),
            'X22': lambda n_samples: np.random.normal(size=n_samples),
            'X23': linear_model(['X18'], [random.uniform(0, 1)]),
            'X24': linear_model(['X21', 'X22', 'X25'],
                                [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]),
            'X25': lambda n_samples: np.random.normal(size=n_samples),
            'X26': linear_model(['X14', 'X18'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X27': lambda n_samples: np.random.normal(size=n_samples),
            'X28': linear_model(['X7', 'X26'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X29': linear_model(['X30', 'X2'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X30': lambda n_samples: np.random.normal(size=n_samples),
        })


    @staticmethod
    def random_scm_treatment_outcome():
        """
        Generates a random SCM with treatment (X12) and outcome (X18).
        See https://github.com/ijmbarr/causalgraphicalmodels/blob/master/causalgraphicalmodels/examples.py
        :return: StructuralCausalModel
        """
        return StructuralCausalModel({
            'X1': lambda n_samples: np.random.normal(size=n_samples),
            'X2': lambda n_samples: np.random.normal(size=n_samples),
            'X3': linear_model(['X24'], [random.uniform(0, 1)]),
            'X4': lambda n_samples: np.random.normal(size=n_samples),
            'X5': lambda n_samples: np.random.normal(size=n_samples),
            'X6': lambda n_samples: np.random.normal(size=n_samples),
            'X7': lambda n_samples: np.random.normal(size=n_samples),
            'X8': lambda n_samples: np.random.normal(size=n_samples),
            'X9': lambda n_samples: np.random.normal(size=n_samples),
            'X10': lambda n_samples: np.random.normal(size=n_samples),
            'X11': linear_model(['X1', 'X5'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X12': logistic_model(['X5', 'X6'], [0, 1]),
            'X13': linear_model(['X9','X1'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X14': linear_model(['X8'], [random.uniform(0, 1)]),
            'X15': linear_model(['X14', 'X4', 'X27'],
                                [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]),
            'X16': linear_model(['X15', 'X2'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X17': linear_model(['X10', 'X11'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X18': logistic_model(['X12', 'X2'], [0, 1]),
            'X19': linear_model(['X3'], [random.uniform(0, 1)]),
            'X20': linear_model(['X19', 'X6'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X21': linear_model(['X8'], [random.uniform(0, 1)]),
            'X22': lambda n_samples: np.random.normal(size=n_samples),
            'X23': linear_model(['X18'], [random.uniform(0, 1)]),
            'X24': linear_model(['X21', 'X22', 'X25'],
                                [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]),
            'X25': lambda n_samples: np.random.normal(size=n_samples),
            'X26': linear_model(['X14', 'X18'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X27': lambda n_samples: np.random.normal(size=n_samples),
            'X28': linear_model(['X7', 'X26'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X29': linear_model(['X30', 'X2'], [random.uniform(0, 1), random.uniform(0, 1)]),
            'X30': lambda n_samples: np.random.normal(size=n_samples),
        })