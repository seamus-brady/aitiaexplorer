from tests.unit import TestAPI
import time
from app.py_causal_wrapper import PyCausalWrapper


class Test_PyCausalWrapper(TestAPI):
    """
    TBD
    """
    wrapper = PyCausalWrapper()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_tetrad(self):
        self.wrapper.start_vm()
        tetrad = self.wrapper.get_tetrad()
        self.wrapper.stop_vm()
        self.assertTrue(tetrad is not None)


    def test_get_algos(self):
        self.wrapper.start_vm()
        causal_discovery_algos = []
        for algo in self.wrapper.get_causal_discovery_algos():
            causal_discovery_algos.append(algo)
        self.wrapper.stop_vm()
        self.assertTrue(len(causal_discovery_algos) == 25)
