"""
Implementation of LiNGAM-GC proposed in [1]

Shoubo Hu (shoubo.sub [at] gmail.com)
2020-05-17

[1] Chen, Zhitang, and Laiwan Chan. "Causality in linear nongaussian
    acyclic models in the presence of latent gaussian confounders."
    Neural Computation 25.6 (2013): 1605-1641.

This code is based on work from https://github.com/amber0309/LiNGAM-GC
which is released under the MIT License
"""

from copy import deepcopy

import numpy as np
from scipy.stats import kurtosis
from sklearn.linear_model import LassoLarsIC, LinearRegression
from sklearn.utils import check_array


class LiNGAM_GC_Algorithm(object):
    def __init__(self):
        self._causal_order = None  # list storing the causal order

    def fit(self, X):
        """
        fit LiNGAM-GC on X and estimate the adjacency matrix

        Input:
            X 		 the matrix of observed data
                     (n_instance, n_var) numpy array
        """
        self.data = check_array(X)
        n_var = self.data.shape[1]

        U = np.arange(n_var)
        K = []
        X_ = np.copy(X)
        for _ in range(0, n_var):
            cu_i = self._search_exogenous_x(X_, U)
            for i in U:
                if i != cu_i:
                    X_[:, i] = self._residual(X_[:, i], X_[:, cu_i])
            K.append(cu_i)
            U = U[U != cu_i]

        self._causal_order = K
        self._estimate_adjacency_matrix(X)

    def get_results(self):
        if self._causal_order == None:
            print('Error: model not yet fitted on data!')
            return self
        print('Estimated causal order')
        print(self._causal_order)
        print('Estimated graph structure')
        print(self.adjacency_matrix_)
        return self._causal_order, self.adjacency_matrix_

    def _estimate_adjacency_matrix(self, X):
        """
        estimate adjacency matrix according to the causal order.

        Input:
            X 		 the matrix of observed data
                     (n_instance, n_var) numpy array

        Output:
            self 	 the object itself
        """
        B = np.zeros([X.shape[1], X.shape[1]], dtype='float64')
        for i in range(1, len(self._causal_order)):
            coef = self._predict_adaptive_lasso(
                X, self._causal_order[:i], self._causal_order[i])
            B[self._causal_order[i], self._causal_order[:i]] = coef
        self.adjacency_matrix_ = B
        return self

    def _predict_adaptive_lasso(self, X, predictors, target, gamma=1.0):
        """
        predict with Adaptive Lasso.

        Input:
            X 				 training instances
                             (n_instances, n_var) numpy array
            predictors 		 indices of predictor variables
                             (n_predictors, ) list()
            target 			 index of target variable
                             int

        Output:
            coef 			 Coefficients of predictor variable
                             (n_predictors,) numpy array
        """
        lr = LinearRegression()
        lr.fit(X[:, predictors], X[:, target])
        weight = np.power(np.abs(lr.coef_), gamma)
        reg = LassoLarsIC(criterion='bic')
        reg.fit(X[:, predictors] * weight, X[:, target])
        return reg.coef_ * weight

    def _search_exogenous_x(self, X, U):
        """
        find the exogenous variable in the remaining ones

        Input:
            X 		 the matrix of observed data
                     (n_instance, n_var) numpy array
            U 		 list of remaining variables to be ordered
                     list()

        Output:
            i 		 the index of the exogenous variable
                     int
        """
        if len(U) == 1:
            return U[0]

        M_list = []
        for i in U:
            M = 0.0
            xi_hat = self._standardize(X[:, i])
            for j in U:
                if i != j:
                    xj_hat = self._standardize(X[:, j])
                    R_xi_xj = self._causal_measure(xi_hat, xj_hat)
                    M += np.min([0, R_xi_xj]) ** 2
            M_list.append(-1.0 * M)
        return U[np.argmax(M_list)]

    def _residual(self, xi, xj):
        return xi - (self._cross_cumulant_4th(xj, xi) / self._cross_cumulant_4th(xj, xj)) * xj

    def _cross_cumulant_4th(self, x, y):
        return np.mean(x ** 3 * y) - 3 * np.mean(x * y) * np.mean(x ** 2)

    def _causal_measure(self, x, y):
        """
        compute the cumulant-based measure in LiNGAM-GC

        Input:
            x, y 		the vector of standardized data of x and y
                        (n_instance,) numpy array

        Output:
            R 			the cumulant-based measure in LiNGAM-GC
                        float
        """
        C_xy = self._cross_cumulant_4th(x, y)
        C_yx = self._cross_cumulant_4th(y, x)
        R = C_xy ** 2 - C_yx ** 2
        return R

    def _standardize(self, x):
        """
        standardize the data to have unit kurtosis

        Input:
            x 			the vector of data
                        (n_instance, ) numpy array

        Output:
            x_hat 		the standardized data
                        (n_instance, ) numpy array
        """
        kurts = kurtosis(x)  # calculate Fisher kurtosis
        k_x = np.abs(kurts) ** (1. / 4)  # the quantity for standardization (k_x in [1])
        x_hat = x / k_x  # the standardized data
        return x_hat

    # methods to generate test data

    @staticmethod
    def generate_test_data_given_model(b, s, c, n_samples=10000, random_state=0):
        """
        Generate artificial data based on the given model.
        INPUT
          b 		 Strictly lower triangular coefficient matrix, (n_vars, n_vars) numpy array
                     NOTE: Each row of `b` corresponds to each variable, i.e., X = BX.
          s 		 Scales of disturbance variables, (n_vars, ) numpy array
          c 		 Means of observed variables, (n_vars,) numpy array
        OUTPUT
          xs 		 matrix of all observations, (n_samples, n_vars) numpy array
          b_struc 	 permuted graph structure, (n_vars, n_vars) numpy array
                         NOTE: 1 and 0 denotes the existence and non-existence of an edge, respectively
        """

        rng = np.random.RandomState(random_state)
        n_vars = b.shape[0]

        # Check args
        assert (b.shape == (n_vars, n_vars))
        assert (s.shape == (n_vars,))
        assert (np.sum(np.abs(np.diag(b))) == 0)
        np.allclose(b, np.tril(b))

        # Nonlinearity exponent, selected to lie in [0.5, 0.8] or [1.2, 2.0].
        # (<1 gives subgaussian, >1 gives supergaussian)
        # q = rng.rand(n_vars) * 1.1 + 0.5
        # ixs = np.where(q > 0.8)
        # q[ixs] = q[ixs] + 0.4

        # Generates disturbance variables
        ss = rng.randn(n_samples, n_vars)
        # ss = np.sign(ss) * (np.abs(ss)**q)

        # Normalizes the disturbance variables to have the appropriate scales
        ss = ss / np.std(ss, axis=0) * s
        # Generate the data one component at a time
        xs = np.zeros((n_samples, n_vars))
        for i in range(n_vars):
            # NOTE: columns of xs and ss correspond to rows of b
            xs[:, i] = ss[:, i] + xs.dot(b[i, :]) + c[i]

        # Permute variables
        p = rng.permutation(n_vars)
        xs[:, :] = xs[:, p]
        b_ = deepcopy(b)
        c_ = deepcopy(c)
        b_[:, :] = b_[p, :]
        b_[:, :] = b_[:, p]
        c_[:] = c[p]

        b_struc = deepcopy(b_)
        b_struc[b_struc != 0] = 1
        b_struc = b_struc.astype(int)

        p_ = list(p)
        k = [p_.index(i) for i in range(n_vars)]

        return xs, b_struc, k

    @staticmethod
    def generate_test_gcm(n_vars, n_edges, stds=1, biases=0):
        B = np.zeros((n_vars, n_vars), dtype=float)
        max_nedge = n_vars * (n_vars - 1) // 2

        trilB_vec = np.array([0] * (max_nedge - n_edges) + [1] * n_edges)
        np.random.shuffle(trilB_vec)
        ridx, cidx = np.tril_indices(n=n_vars, k=-1, m=n_vars)
        B[ridx, cidx] = trilB_vec

        stds_vec = np.ones((n_vars,), dtype=float) * stds
        bias_vec = np.ones((n_vars,), dtype=float) * biases

        X_, B_, k_ = LiNGAM_GC_Algorithm.generate_test_data_given_model(B, stds_vec, bias_vec)

        return X_, B_, k_
