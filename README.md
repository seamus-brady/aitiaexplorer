# AitaExplorer


The word _aitia_ comes from the Ancient Greek word _αιτία_ used in early Philosophy to mean _cause_.

 - AitiaExplorer is an experimental** causal discovery exploration tool.
 - It allows a user to explore selections of features in a dataset for causal connections.

## Overview

- AitiaExplorer takes in a random dataset in the form a Pandas Dataframe.
- The important features of the dataset can be extracted automatically using a variety of unsupervised learning methods. 
- AitiaExplorer will automatically discovery causal graphs based on subsets of features using a selection of causal discovery algorithms (see below).
- The best performing causal graph will be returned for further analysis.
- AitiaExplorer will work with data where a causal model is known but can also automatically generate an approximate causal graph where no graph is known.

## Usage

For examples of how AitiaExplorer can be used, please see the Jupyter Notebooks in the _notebooks_ folder.

## Requirements

You will need 
- Python 3.7+
- pip 

To view the Jupyter Notebook examples, you will also need Anaconda installed.

## Setup

Clone this repository:
```
git clone git@github.com:corvideon/aitiaexplorer.git
```
Change to the directory contain AitiaExplorer:

```
cd aitiaexplorer
```

Install the python dependencies using pip:
```
pip install -r /path/to/requirements.txt
```

## Causal Libraries Used

Internally, AitiaExplorer uses multiple libraries that support causal discovery.

The main libraries used are:

- py-causal: https://github.com/bd2kccd/py-causal
- CausalGraphicalModels: https://github.com/ijmbarr/causalgraphicalmodels
- pyAgrum: https://agrum.gitlab.io/pages/pyagrum.html

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
 
## License

```
AitiaExplorer is released under the FreeBSD License.

Copyright (c) 2020, Seamus Brady <seamus@corvideon.ie>

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```



