# AitaExplorer

Causal discovery and causal inference tool.


## Setup Notes

### Requirements

Install the requirements:
```
pip install cdt
pip install dowhy
```


### Gotchas

CausalDiscoveryToolbox needs an old sklearn version

`pip install scikit-learn==0.21.0`

Install the latest version of R and the following packages for full CDT support:

```
pcalg
kpcalg
bnlearn
sparsebn
SID
CAM
D2C
RCIT
```

The R package `graph` or `RBGL` may not install, so try using the method at https://stackoverflow.com/questions/47073836/r-how-to-install-package-graph
to install them directly from http://bioconductor.org/

## References

https://github.com/FenTechSolutions/CausalDiscoveryToolbox
- Causal Discovery Toolbox: Uncover causal relationships in Python
- Diviyan Kalainathan, Olivier Goudet
- https://arxiv.org/abs/1903.02278

