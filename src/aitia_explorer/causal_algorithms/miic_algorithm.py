
import logging
import pyAgrum as gum
import tempfile

from aitia_explorer.causal_algorithms.algorithm_constants import AlgorithmConstants
from aitia_explorer.util.graph_util import GraphUtil

_logger = logging.getLogger(__name__)


class MIICAlgorithm():
    """
    Returns a list of unobserved latent edges from a dataframe.
    MIIC (Multivariate Information based Inductive Causation) combines constraint-based
    and information-theoretic approaches to disentangle direct from indirect effects
    amongst correlated variables, including cause-effect relationships and the effect of unobserved latent causes..

    See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5685645/
    """

    def __init__(self):
        pass

    @staticmethod
    def run(df, pc=None):
        """
        Run the algorithm against the dataframe and gets a list of unobserved latent edges.
        """
        try:
            fp = tempfile.NamedTemporaryFile(suffix='.csv')
            df.to_csv(fp.name, encoding='utf-8', index=False)
            learner = gum.BNLearner(fp.name)
            learner.useMIIC()
            bn = learner.learnBN()
            latent_edges = []
            latent_edges.extend([(bn.variable(i).name(),
                                  bn.variable(j).name()) for (i,j) in learner.latentVariables()])
            return latent_edges
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return None
