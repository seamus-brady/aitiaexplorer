"""
TBD Header
"""
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
        self.dot_format_string = None
        self.causal_graph = None
        self.AUPR = None
        self.SHD = None


    def asdict(self):
        """
        For displaying in notebooks...
        :return:
        """
        return {'causal algorithm': self.causal_algorithm,
                'feature selection method': self.feature_selection_method,
                'feature list': self.feature_list,
                'AUPR': self.AUPR,
                'SHD': self.SHD
                }