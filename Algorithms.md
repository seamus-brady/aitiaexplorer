# Causal Discovery Algorithms in AitiaExplorer

## Overview

- The causal discovery algorithms below are used in AitiaExplorer.
- These are made available in the [py-causal](https://github.com/bd2kccd/py-causal) project which is used internally in AitiaExplorer.
- They are implemented in Java and reached via a Java-Python interop framework (https://github.com/CellProfiler/python-javabridge/).
- Unfortunately there is a not a huge amount of documentation available, so in some cases a link has been provided to the original R implementation documentation.


## Causal Algorithm List

 - bayesEst: https://rdrr.io/github/bd2kccd/r-causal/man/bayesEst.html
 
### PC Algorithm

Algorithm ID:  pc-all

_PC algorithm (Spirtes and Glymour, Social Science Computer Review, 1991) is a pattern search which assumes that the underlying causal structure of the input data is acyclic, and that no two variables are caused by the same latent (unmeasured) variable. In addition, it is assumed that the input data set is either entirely continuous or entirely discrete; if the data set is continuous, it is assumed that the causal relation between any two variables is linear, and that the distribution of each variable is Normal. Finally, the sample should ideally be i.i.d.. Simulations show that PC and several of the other algorithms described here often succeed when these assumptions, needed to prove their correctness, do not strictly hold. The PC algorithm will sometimes output double headed edges. In the large sample limit, double headed edges in the output indicate that the adjacent variables have an unrecorded common cause, but PC tends to produce false positive double headed edges on small samples.

The PC algorithm is correct whenever decision procedures for independence and conditional independence are available. The procedure conducts a sequence of independence and conditional independence tests, and efficiently builds a pattern from the results of those tests. As implemented in TETRAD, PC is intended for multinomial and approximately Normal distributions with i.i.d. data. The tests have an alpha value for rejecting the null hypothesis, which is always a hypothesis of independence or conditional independence. For continuous variables, PC uses tests of zero correlation or zero partial correlation for independence or conditional independence respectively. For discrete or categorical variables, PC uses either a chi square or a g square test of independence or conditional independence (see Causation, Prediction, and Search for details on tests). In either case, the tests require an alpha value for rejecting the null hypothesis, which can be adjusted by the user. The procedures make no adjustment for multiple testing. (For PC, CPC, JPC, JCPC, FCI, all testing searches.)_

https://www.rdocumentation.org/packages/pcalg/versions/2.6-10/topics/pc
 
### FCI Algorithm


https://www.rdocumentation.org/packages/pcalg/versions/2.6-10/topics/fci

### FGES Algorithm

https://www.ccd.pitt.edu//wiki/index.php?title=Fast_Greedy_Equivalence_Search_(FGES)_Algorithm_for_Continuous_Variables

### GFCI Algorithm

https://www.ccd.pitt.edu/wiki/index.php/Greedy_Fast_Causal_Inference_(GFCI)_Algorithm_for_Continuous_Variables

### RFCI Algorithm

https://www.rdocumentation.org/packages/pcalg/versions/2.6-10/topics/rfci
 
The NOTEARS algorithm is made available for use but is not run automatically by AitiaExplorer at the moment.
This algorithm returns an unlabelled adjacency matrix rather than a causal graph, making less useful for display even though it is very efficient:

 - NOTEARS: https://github.com/jmoss20/notears 
 
 _Python package implementing "DAGs with NO TEARS: Smooth Optimization for Structure Learning", Xun Zheng, Bryon Aragam, Pradeem Ravikumar and Eric P. Xing (March 2018, arXiv:1803.01422)_
 
## Related Algorithm List
 
The following two algorithms are exposed by the [pyAgrum](https://agrum.gitlab.io/pages/pyagrum.html) project.

The first algorithm is used in AitiaExplorer for creating approximate causal graphs:

 - Greedy Hill Climbing Algorithm: https://webia.lip6.fr/~phw//aGrUM/docs/last/notebooks/13-learningClassifier.ipynb.html
 
 The second algorithm is used for finding unobserved latent variables:
 
 - MIIC: https://webia.lip6.fr/~phw//aGrUM/docs/last/notebooks/14-LearningAndEssentialGraphs.ipynb.html