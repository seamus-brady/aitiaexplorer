"""
This code is based on work from https://github.com/akelleh/causality
which is released under the MIT License
"""

import statsmodels.api as sm

DEFAULT_BINS = 2


class RobustRegressionTest():

    def __init__(self, y, x, z, data, alpha):
        self.regression = sm.RLM(data[y], data[x + z])
        self.result = self.regression.fit()
        self.coefficient = self.result.params[x][0]
        confidence_interval = self.result.conf_int(alpha=alpha / 2.)
        self.upper = confidence_interval[1][x][0]
        self.lower = confidence_interval[0][x][0]

    def independent(self):
        if self.coefficient > 0.:
            if self.lower > 0.:
                return False
            else:
                return True
        else:
            if self.upper < 0.:
                return False
            else:
                return True
