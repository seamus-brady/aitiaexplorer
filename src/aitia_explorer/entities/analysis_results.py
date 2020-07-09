
import logging
import pandas as pd

_logger = logging.getLogger(__name__)


class AnalysisResults():
    """
    A class for handling multiple results from a causal analysis run.
    """

    def __init__(self, results=[]):
        self.results = []

    def to_dataframe(self):
        output_results = []
        for result in self.results:
            output_results.append(result.asdict())
        return pd.DataFrame(output_results)