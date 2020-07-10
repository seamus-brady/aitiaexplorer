## Causal Discovery Algorithms Available in AitiaExplorer

The causal discovery algorithms below are used in AitiaExplorer as available in the py-causal project which are implemented in Java and reached via a Java-Python interop framework (https://github.com/CellProfiler/python-javabridge/).

Unfortunately there is a not a huge amount of documentation available, so in some cases a link has been provided to the original R implementation documentation.

 - bayesEst: https://rdrr.io/github/bd2kccd/r-causal/man/bayesEst.html
 - PC: https://www.rdocumentation.org/packages/pcalg/versions/2.6-10/topics/pc
 - FCI: https://www.rdocumentation.org/packages/pcalg/versions/2.6-10/topics/fci
 - FGES: https://www.ccd.pitt.edu//wiki/index.php?title=Fast_Greedy_Equivalence_Search_(FGES)_Algorithm_for_Continuous_Variables
 - GFCI: https://www.ccd.pitt.edu/wiki/index.php/Greedy_Fast_Causal_Inference_(GFCI)_Algorithm_for_Continuous_Variables
 - RFCI: https://www.rdocumentation.org/packages/pcalg/versions/2.6-10/topics/rfci
 
The following algorithm is made available for use but is not run automatically by AitiaExplorer at the moment:

 - NOTEARS: https://github.com/jmoss20/notears 
 
The following two algorithms are exposed by the pyAgrum project and are used in AitiaExplorer for creating approximate causal graphs and finding unobserved latent variables, respectively:

 - Greedy Hill Climbing Algorithm: https://webia.lip6.fr/~phw//aGrUM/docs/last/notebooks/13-learningClassifier.ipynb.html
 - MIIC: https://webia.lip6.fr/~phw//aGrUM/docs/last/notebooks/14-LearningAndEssentialGraphs.ipynb.html