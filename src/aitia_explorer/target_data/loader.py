"""
TBD
"""
import logging
import os
import pandas as pd
import numpy as np
from causalgraphicalmodels import StructuralCausalModel
from causalgraphicalmodels.csm import logistic_model, linear_model

_logger = logging.getLogger(__name__)


class TargetData:
    """
    A class that provides known causal data for targeting in tests.
    """

    def __init__(self):
        pass

    @staticmethod
    def data_dir():
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        return data_dir

    @staticmethod
    def graphs_dir():
        graphs_dir = os.path.join(os.path.dirname(__file__), 'graphs')
        return graphs_dir

    @staticmethod
    def simulated_data_1_graph():
        graph_path = os.path.join(TargetData.graphs_dir(), "simulated_data_graph_1.dot")
        with open(graph_path, 'r') as dot_file:
            simulated_data_graph = dot_file.read()
        return simulated_data_graph

    @staticmethod
    def simulated_data_1():
        data_dir = os.path.join(TargetData.data_dir(), "simulated_data_1.txt")
        simulated_data = pd.read_table(data_dir, sep="\t")
        return simulated_data

    @staticmethod
    def audiology_data():
        data_dir = os.path.join(TargetData.data_dir(), "audiology.txt")
        return pd.read_table(data_dir, sep="\t")

    @staticmethod
    def charity_data():
        data_dir = os.path.join(TargetData.data_dir(), "charity.txt")
        return pd.read_table(data_dir, sep="\t")

    @staticmethod
    def lucas0_data():
        """
        See http://www.causality.inf.ethz.ch/data/LUCAS.html
        :return:
        """
        data_dir = os.path.join(TargetData.data_dir(), "lucas0_train.csv")
        return pd.read_csv(data_dir)

    @staticmethod
    def lucas2_data():
        """
        See http://www.causality.inf.ethz.ch/data/LUCAS.html
        :return:
        """
        data_dir = os.path.join(TargetData.data_dir(), "lucas0_train.csv")
        return pd.read_csv(data_dir)

    @staticmethod
    def scm1():
        """
        Returns a StructuralCausalModel for sampling
        See https://github.com/ijmbarr/causalgraphicalmodels/blob/master/causalgraphicalmodels/examples.py
        :return:
        """
        return StructuralCausalModel({
            "a": lambda n_samples: np.random.normal(size=n_samples),
            "b": lambda n_samples: np.random.normal(size=n_samples),
            "x": logistic_model(["a", "b"], [-1, 1]),
            "c": linear_model(["x"], [1]),
            "y": linear_model(["c", "e"], [3, -1]),
            "d": linear_model(["b"], [-1]),
            "e": linear_model(["d"], [2]),
            "f": linear_model(["y"], [0.7]),
            "h": linear_model(["y", "a"], [1.3, 2.1])
        })