#
# This file is part of AitiaExplorer and is released under the FreeBSD License.
#
# Copyright (c) 2020, Seamus Brady <seamus@corvideon.ie>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.#
import numpy as np
import scipy
import scipy.optimize


def run(variant, data, loss, loss_grad, **kwargs):
    return variant(data, loss, loss_grad, **kwargs)


def notears_standard(data, loss, loss_grad, c=0.25, r=10.0, e=1e-8, rnd_W_init=False, output_all_progress=False,
                     verbose=False):
    """
    Runs NOTEARS algorithm.
    
    Args:
        data (np.array): n x d data matrix with n samples, d variables
        c (float): minimum rate of progress, c \in (0,1)
        r (float): penalty growth rate, r > 1
        e (float): optimation accuracy, e > 0 (acyclicity stopping criteria)
        loss (function): loss function
        loss_grad (function): gradient of the loss function
        rnd_W_init (bool): initialize W to std. normal random matrix, rather than zero matrix
        output_all_progress (bool): return all intermediate values of W, rather than just the final value
        verbose (bool): print optimization information
    Returns:
        dict: { 'h': acyclicity of output, 
                'loss': loss of output, 
                'W': resulting optimized adjacency matrix}
    """
    n = np.shape(data)[0]
    d = np.shape(data)[1]

    data = np.array(data).astype(dtype=np.float64)
    cov = np.cov(data.T)

    if rnd_W_init:
        W = np.random.randn(d, d)
    else:
        W = np.zeros([d, d])  # initial guess
    W = W.astype(dtype=np.float64)
    a = 0.0  # initial guess
    p = 1.0  # initial penalty

    if output_all_progress:
        ret = []

    def h(W):
        # tr exp(W ◦ W) − d 
        return np.trace(scipy.linalg.expm(np.multiply(W, W))) - d

    def h_grad(W):
        # ∇h(W) = [exp(W ◦ W)]^T ◦ 2W
        return np.multiply(np.transpose(scipy.linalg.expm(np.multiply(W, W))), 2.0 * W)

    def L(W, p, a):
        W = np.reshape(W, [d, d]).astype(dtype=np.float64)
        return loss(W, data, cov, d, n) + (p / 2.0) * (h(W) ** 2) + a * (h(W))

    def L_grad(W, p, a):
        W = np.reshape(W, [d, d]).astype(dtype=np.float64)
        return np.reshape(loss_grad(W, data, cov, d, n) + h_grad(W) * (a + (p * h(W))), [d ** 2]).astype(
            dtype=np.float64)

    def get_W_star(p, W, a):
        return scipy.optimize.minimize(L, W, args=(p, a), jac=L_grad, method='L-BFGS-B', options={'disp': False})

    while True:
        W_star = np.reshape(get_W_star(p, W, a)['x'], [d, d]).astype(dtype=np.float64)
        h_W_star = h(W_star)
        if h(W) != 0.0:
            while h_W_star >= max(c * h(W), e):
                p = r * p
                W_star = np.reshape(get_W_star(p, W, a)['x'], [d, d]).astype(dtype=np.float64)
                h_W_star = h(W_star)
                if verbose:
                    print("Increasing p:\t p = {: .2e}\n\t\t h_W_star = {}".format(p, h_W_star))
        if output_all_progress:
            ret.append({'h': h_W_star, 'loss': loss(W_star, data, cov, d, n), 'a': a, 'W': W_star})
        if h_W_star < e:
            if verbose:
                print(
                    "Done:\t\t h = {}\n\t\t loss = {}\nt\t\t a = {}".format(h_W_star, loss(W_star, data, cov, d, n), a))
            if output_all_progress:
                return ret
            return {'h': h_W_star, 'loss': loss(W_star, data, cov, d, n), 'W': W_star}
        if verbose:
            print("Progress:\t h = {}\n\t\t loss = {}\n\t\t a = {}".format(h_W_star, loss(W_star, data, cov, d, n), a))
        a = a + p * h_W_star
        W = W_star
