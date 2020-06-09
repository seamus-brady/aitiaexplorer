"""
TBD Header
"""

from pycausal.pycausal import pycausal
from app.causal_discovery_algo_enum import CausalDiscoveryAlgos
from pycausal import search

class PyCausalWrapper():
    """
    TBD
    """

    def __init__(self):
        """
        TBD
        :return:
        """
        self.pc = pycausal()

    def start_vm(self):
        """
        TBD
        :return:
        """
        self.pc.start_vm()


    def stop_vm(self):
        """
        TBD
        :return:
        """
        self.pc.stop_vm()

    def get_tetrad(self):
        """
        TBD
        :return:
        """
        return search.tetradrunner()

    def get_causal_discovery_algos(self):
        """
        TBD
        :return:
        """
        causal_discovery_algos = []
        algo_dict = self.get_tetrad().algos
        for k, v in algo_dict.items():
            causal_discovery_algos.append(k)
        return causal_discovery_algos
