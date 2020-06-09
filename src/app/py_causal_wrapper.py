"""
TBD Header
"""
import logging

from pycausal import prior as p
from pycausal import search as s
from pycausal.pycausal import pycausal

_logger = logging.getLogger(__name__)


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
        return s.tetradrunner()

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

    def dump_all_algo_desc(self):
        """
        TBD
        :return:
        """
        for algo in self.get_causal_discovery_algos():
            print("------------------------------------------------------------------------")
            print("Algorithm ID: ", algo)
            print("Description: ", algo)
            self.get_tetrad().getAlgorithmDescription(algo)

    # -------------------------------------------------------------------------------------------------
    #                   The methods below return causal discovery algorithms....
    # -------------------------------------------------------------------------------------------------

    def algo_bayes_est(self, df):
        """
        bayesEst is the revised Greedy Equivalence Search (GES) algorithm developed
        by Joseph D. Ramsey, Director of Research Computing, Department of Philosophy,
        Carnegie Mellon University, Pittsburgh, PA.
        :param df: dataframe
        :return: dot graph string
        """
        dot_str = None
        try:
            bayes_est = s.bayesEst(df, depth=-1, alpha=0.05, verbose=True)
            if bayes_est.getTetradGraph() is not None:
                dot_str = self.pc.tetradGraphToDot(bayes_est.getTetradGraph())
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str

    def algo_fges_continuous(self, df):
        """
        FGES is an optimized and parallelized version of an algorithm developed by Meek [Meek, 1997]
        called the Greedy Equivalence Search (GES). The algorithm was further developed and studied
        by Chickering [Chickering, 2002]. GES is a Bayesian algorithm that heuristically searches
        the space of CBNs and returns the model with highest Bayesian score it finds. In particular,
        GES starts its search with the empty graph. It then performs a forward stepping search in
        which edges are added between nodes in order to increase the Bayesian score. This process
        continues until no single edge addition increases the score. Finally, it performs a backward
        stepping search that removes edges until no single edge removal can increase the score.
        The algorithms requires a decomposable score—that is, a score that for the entire DAG model
        is a sum of logged scores of each variables given its parents in the model. The algorithms
        can take all continuous data (using the SEM BIC score), all discrete data
        (using the BDeu score) or a mixture of continuous and discrete data
        (using the Conditional Gaussian score); these are all decomposable scores.
        :param df: dataframe
        :return: dot graph string
        """
        dot_str = None
        # pass in empty prior
        forbid = []
        require = []
        tempForbid = p.ForbiddenWithin([])
        temporal = [tempForbid]
        try:
            prior = p.knowledge(forbiddirect=forbid, requiredirect=require, addtemporal=temporal)
            self.get_tetrad().run(algoId='fges', dfs=df, scoreId='sem-bic', dataType='continuous',
                                  maxDegree=-1, faithfulnessAssumed=True, verbose=True)
            if self.get_tetrad().getTetradGraph() is not None:
                dot_str = self.pc.tetradGraphToDot(self.get_tetrad().getTetradGraph())
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str

    def algo_fges_discrete(self, df):
        """
        FGES is an optimized and parallelized version of an algorithm developed by Meek [Meek, 1997]
        called the Greedy Equivalence Search (GES). The algorithm was further developed and studied
        by Chickering [Chickering, 2002]. GES is a Bayesian algorithm that heuristically searches
        the space of CBNs and returns the model with highest Bayesian score it finds. In particular,
        GES starts its search with the empty graph. It then performs a forward stepping search in
        which edges are added between nodes in order to increase the Bayesian score. This process
        continues until no single edge addition increases the score. Finally, it performs a backward
        stepping search that removes edges until no single edge removal can increase the score.
        The algorithms requires a decomposable score—that is, a score that for the entire DAG model
        is a sum of logged scores of each variables given its parents in the model. The algorithms
        can take all continuous data (using the SEM BIC score), all discrete data
        (using the BDeu score) or a mixture of continuous and discrete data
        (using the Conditional Gaussian score); these are all decomposable scores.
        :param df: dataframe
        :return: dot graph string
        """
        dot_str = None
        # pass in empty prior
        forbid = []
        require = []
        tempForbid = p.ForbiddenWithin([])
        temporal = [tempForbid]
        try:
            prior = p.knowledge(forbiddirect=forbid, requiredirect=require, addtemporal=temporal)
            self.get_tetrad().run(algoId='fges', dfs=df, scoreId='bdeu-score', priorKnowledge=prior,
                                  dataType='discrete', maxDegree=3, faithfulnessAssumed=True, verbose=True,
                                  numberResampling=5, resamplingEnsemble=1, addOriginalDataset=True)
            if self.get_tetrad().getTetradGraph() is not None:
                dot_str = self.pc.tetradGraphToDot(self.get_tetrad().getTetradGraph())
        except Exception as e:
            _logger.error(str(e))
            print(str(e))
        return dot_str
