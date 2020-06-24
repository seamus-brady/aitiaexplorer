"""
TBD
"""
import logging
import os
import pandas as pd
import numpy as np
import random
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
        Returns a StructuralCausalModel for sampling
        See https://github.com/ijmbarr/causalgraphicalmodels/blob/master/causalgraphicalmodels/examples.py
        :return:
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
        Returns a StructuralCausalModel for sampling
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
    def virus_scm():
        """
        Returns a StructuralCausalModel for sampling
        See https://github.com/ijmbarr/causalgraphicalmodels/blob/master/causalgraphicalmodels/examples.py
        :return:
        """
        return StructuralCausalModel({
            'age': lambda n_samples: np.random.normal(size=n_samples),
            'bmi': lambda n_samples: np.random.normal(size=n_samples),
            'gene_1': lambda n_samples: np.random.normal(size=n_samples),
            'gene_2': lambda n_samples: np.random.normal(size=n_samples),
            'gene_3': lambda n_samples: np.random.normal(size=n_samples),
            'gene_4': linear_model(['age', 'gene_2'], [-1, 0.8]),
            'gene_5': linear_model(['gene_4', 'gene_1'], [0.5, 1.5]),
            'gene_6': linear_model(['gene_1', 'gene_5'], [0.5, 1.1]),
            'gene_7': linear_model(['gene_6', 'gene_5'], [-1, 0.75]),
            'blood_pressure': lambda n_samples: np.random.normal(size=n_samples),
            'smoking': lambda n_samples: np.random.normal(size=n_samples),
            'heart_health': linear_model(['blood_pressure', 'age'], [0.8, 1.2]),
            'white_cell_count': linear_model(['gene_1', 'bmi'], [-1, 1]),
            'serious_viral_illness': linear_model(['gene_3', 'heart_health', 'white_cell_count'], [-1, 0.2, 1]),
        })

    @staticmethod
    def random_scm(upper=20):
        """
        Generates a random SCM.
        :return:
        """
        # create a set of feature vars
        features = []
        for n in np.arange(0, upper):
            features.append('X' + str(n))

        # now set a value for each feature
        scm_contents = dict()
        normal_feature_num = upper - random.randint(1, upper)

        # normally distributed features
        for n in np.arange(0, normal_feature_num):
            scm_contents[features.pop()] = lambda n_samples: np.random.normal(size=n_samples)

        # conditional features
        edges = []
        for n in np.arange(0, upper - normal_feature_num):
            feature = features.pop()
            choice = random.randint(0, 1)
            if choice == 0:
                f0 = TargetData.get_random_int(n, n1=None, upper=(upper - normal_feature_num))
                suggested_edge = feature + 'X' + str(f0)
                if suggested_edge not in edges:
                    scm_contents[feature] = linear_model(['X' + str(f0)],
                                                                [random.uniform(0, 1)])
                    edges.append(suggested_edge)
            if choice == 1:
                f0 = TargetData.get_random_int(n, n1=None, upper=(upper - normal_feature_num))
                f1 = TargetData.get_random_int(n, f0, upper=(upper - normal_feature_num))
                suggested_edge = feature + 'X' + str(f0) + 'X' + str(f1)
                if suggested_edge not in edges:
                    scm_contents[feature] = linear_model(['X' + str(f0), 'X' + str(f1)],
                                                                [random.uniform(0, 1), random.uniform(0, 1)])
                    edges.append(suggested_edge)
        return StructuralCausalModel(scm_contents)

    @staticmethod
    def get_random_int(n0, n1=None, upper=None):
        """
        Gets any random int, except for the n0, n1 passed in.
        :param n:
        :param upper:
        :return:
        """
        random_int = None
        if n1 is None:
            while random_int is None or random_int == n0:
                random_int = random.randint(0, upper)
        else:
            while random_int is None or random_int == n0 or random_int == n1:
                random_int = random.randint(0, upper)
        return random_int
