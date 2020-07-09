
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
        self.AUPR = None
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
                'AUPR': self.AUPR,
                'SHD': self.SHD
                }