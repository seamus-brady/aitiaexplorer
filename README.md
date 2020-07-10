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

Please see [here](Algorithms.md) for a list of the causal discovery algorithms available in AitiaExplorer.
 
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



