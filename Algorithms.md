# Causal Discovery Algorithms in AitiaExplorer

## Overview

- The causal discovery algorithms below are used in AitiaExplorer.
- Most of these algorithms are made available in the [py-causal](https://github.com/bd2kccd/py-causal) project which is used internally in AitiaExplorer.
- They are implemented in Java as part of the [Tetrad](http://www.phil.cmu.edu/tetrad/) software and are mainly ported over from the original R versions.
 - These algorithms are then exposed to Python (using py-causal) via a [Java-Python interop framework](https://github.com/CellProfiler/python-javabridge/).
- Other causal algorithms are listed also.


## py-causal Algorithm List

- Each algorithm is listed alongside the metadata that is available from within py-causal.
- The Algorithm ID is the name as listed in py-causal / Tetrad.
- In some cases a link has been provided to the original R implementation documentation or other useful
- For more information on the algorithm and the parameters, please see the example Jupyter Notebooks available in py-causal.

### bayesEst

Algorithm ID: bayesEst

*bayesEst is the revised Greedy Equivalence Search (GES) algorithm developed by Joseph D. Ramsey, Director of Research Computing, Department of Philosophy, Carnegie Mellon University, Pittsburgh, PA.*

https://rdrr.io/github/bd2kccd/r-causal/man/bayesEst.html
 
### PC Algorithm

Algorithm ID:  pc-all

_PC algorithm (Spirtes and Glymour, Social Science Computer Review, 1991) is a pattern search which assumes that the underlying causal structure of the input data is acyclic, and that no two variables are caused by the same latent (unmeasured) variable. In addition, it is assumed that the input data set is either entirely continuous or entirely discrete; if the data set is continuous, it is assumed that the causal relation between any two variables is linear, and that the distribution of each variable is Normal. Finally, the sample should ideally be i.i.d.. Simulations show that PC and several of the other algorithms described here often succeed when these assumptions, needed to prove their correctness, do not strictly hold. The PC algorithm will sometimes output double headed edges. In the large sample limit, double headed edges in the output indicate that the adjacent variables have an unrecorded common cause, but PC tends to produce false positive double headed edges on small samples._

_The PC algorithm is correct whenever decision procedures for independence and conditional independence are available. The procedure conducts a sequence of independence and conditional independence tests, and efficiently builds a pattern from the results of those tests. As implemented in TETRAD, PC is intended for multinomial and approximately Normal distributions with i.i.d. data. The tests have an alpha value for rejecting the null hypothesis, which is always a hypothesis of independence or conditional independence. For continuous variables, PC uses tests of zero correlation or zero partial correlation for independence or conditional independence respectively. For discrete or categorical variables, PC uses either a chi square or a g square test of independence or conditional independence (see Causation, Prediction, and Search for details on tests). In either case, the tests require an alpha value for rejecting the null hypothesis, which can be adjusted by the user. The procedures make no adjustment for multiple testing. (For PC, CPC, JPC, JCPC, FCI, all testing searches)._

https://www.rdocumentation.org/packages/pcalg/versions/2.6-10/topics/pc
 
### FCI Algorithm

Algorithm ID:  fci

*The FCI algorithm is a constraint-based algorithm that takes as input sample data and optional background knowledge and in the large sample limit outputs an equivalence class of CBNs that (including those with hidden confounders) that entail the set of conditional independence relations judged to hold in the population. It is limited to several thousand variables, and on realistic sample sizes it is inaccurate in both adjacencies and orientations. FCI has two phases: an adjacency phase and an orientation phase. The adjacency phase of the algorithm starts with a complete undirected graph and then performs a sequence of conditional independence tests that lead to the removal of an edge between any two adjacent variables that are judged to be independent, conditional on some subset of the observed variables; any conditioning set that leads to the removal of an adjacency is stored. After the adjacency phase, the resulting undirected graph has the correct set of adjacencies, but all of the edges are unoriented. FCI then enters an orientation phase that uses the stored conditioning sets that led to the removal of adjacencies to orient as many of the edges as possible. See [Spirtes, 1993].*

https://www.rdocumentation.org/packages/pcalg/versions/2.6-10/topics/fci

### FGES Algorithm

Algorithm ID:  fges

*FGES is an optimized and parallelized version of an algorithm developed by Meek [Meek, 1997] called the Greedy Equivalence Search (GES). The algorithm was further developed and studied by Chickering [Chickering, 2002]. GES is a Bayesian algorithm that heuristically searches the space of CBNs and returns the model with highest Bayesian score it finds. In particular, GES starts its search with the empty graph. It then performs a forward stepping search in which edges are added between nodes in order to increase the Bayesian score. This process continues until no single edge addition increases the score. Finally, it performs a backward stepping search that removes edges until no single edge removal can increase the score. More information is available here and here. The reference is Ramsey et al., 2017.*

*The algorithms requires a decomposable score—that is, a score that for the entire DAG model is a sum of logged scores of each variables given its parents in the model. The algorithms can take all continuous data (using the SEM BIC score), all discrete data (using the BDeu score) or a mixture of continuous and discrete data (using the Conditional Gaussian score); these are all decomposable scores.*

https://www.ccd.pitt.edu//wiki/index.php?title=Fast_Greedy_Equivalence_Search_(FGES)_Algorithm_for_Continuous_Variables

### GFCI Algorithm

Algorithm ID:  gfci

*GFCI is a combination of the FGES [CCD-FGES, 2016] algorithm and the FCI algorithm [Spirtes, 1993] that improves upon the accuracy and efficiency of FCI. In order to understand the basic methodology of GFCI, it is necessary to understand some basic facts about the FGES and FCI algorithms. The FGES algorithm is used to improve the accuracy of both the adjacency phase and the orientation phase of FCI by providing a more accurate initial graph that contains a subset of both the non-adjacencies and orientations of the final output of FCI. The initial set of nonadjacencies given by FGES is augmented by FCI performing a set of conditional independence tests that lead to the removal of some further adjacencies whenever a conditioning set is found that makes two adjacent variables independent. After the adjacency phase of FCI, some of the orientations of FGES are then used to provide an initial orientation of the undirected graph that is then augmented by the orientation phase of FCI to provide additional orientations. A verbose description of GFCI can be found here (discrete variables) and here (continuous variables).*

https://www.ccd.pitt.edu/wiki/index.php/Greedy_Fast_Causal_Inference_(GFCI)_Algorithm_for_Continuous_Variables

### RFCI Algorithm

Algorithm ID:  rfci

*A modification of the FCI algorithm in which some expensive steps are finessed and the output is somewhat differently interpreted. In most cases this runs faster than FCI (which can be slow in some steps) and is almost as informative. See Colombo et al., 2012.*

https://www.rdocumentation.org/packages/pcalg/versions/2.6-10/topics/rfci

## Other Causal Algorithms

### NOTEARS
 
The NOTEARS algorithm is made available for use but is not run automatically by AitiaExplorer at the moment.
This algorithm returns an unlabelled adjacency matrix rather than a causal graph, making less useful for display even though it is very efficient:

 - NOTEARS: https://github.com/jmoss20/notears 
 
 _Python package implementing "DAGs with NO TEARS: Smooth Optimization for Structure Learning", Xun Zheng, Bryon Aragam, Pradeem Ravikumar and Eric P. Xing (March 2018, arXiv:1803.01422)_
 
## Related Algorithms
 
The following two algorithms are exposed by the [pyAgrum](https://agrum.gitlab.io/pages/pyagrum.html) project.

The first algorithm is used in AitiaExplorer for creating approximate causal graphs:

 - Greedy Hill Climbing Algorithm: https://webia.lip6.fr/~phw//aGrUM/docs/last/notebooks/13-learningClassifier.ipynb.html
 
 The second algorithm is used for finding unobserved latent variables:
 
 - MIIC: https://webia.lip6.fr/~phw//aGrUM/docs/last/notebooks/14-LearningAndEssentialGraphs.ipynb.html