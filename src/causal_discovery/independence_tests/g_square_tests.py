"""
This code is based on work from https://github.com/keiichishima/gsq/tree/master/gsq
which is released under the GPL v2 License
"""

import logging
from math import pow

import numpy as np
from scipy.stats import chi2

_logger = logging.getLogger(__name__)


class GSquareTest():
    """
    Static class for G square independence test used by PC algorithm.
    """

    def __init__(self):
       pass

    @staticmethod
    def test_binary(data, x, y, s, **kwargs):
        return GSquareTest._g_square_binary(data, x, y, s)

    @staticmethod
    def test_discrete(data, x, y, s, **kwargs):
        levels = []
        if 'levels' in kwargs:
            levels = kwargs['levels']
        else:
            import numpy as np
            levels = np.amax(data, axis=0) + 1
        return GSquareTest._g_square_discrete(data, x, y, s, levels)

    @staticmethod
    def _g_square_binary(data, x, y, s):
        """
        G square test for a binary data.
        Args:
            data: the data matrix to be used (as a numpy.ndarray).
            x: the first node (as an integer).
            y: the second node (as an integer).
            s: the set of neighbouring nodes of x and y (as a set()).
        Returns:
            p_val: the p-value of conditional independence.
        """

        def _calculate_tlog(x, y, s, dof, dm):
            nijk = np.zeros((2, 2, dof))
            s_size = len(s)
            z = []
            for z_index in range(s_size):
                z.append(s.pop())
                pass
            for row_index in range(0, dm.shape[0]):
                i = dm[row_index, x]
                j = dm[row_index, y]
                k = []
                k_index = 0
                for z_index in range(s_size):
                    k_index += dm[row_index, z[z_index]] * int(pow(2, z_index))
                    pass
                nijk[i, j, k_index] += 1
                pass
            nik = np.ndarray((2, dof))
            njk = np.ndarray((2, dof))
            for k_index in range(dof):
                nik[:, k_index] = nijk[:, :, k_index].sum(axis=1)
                njk[:, k_index] = nijk[:, :, k_index].sum(axis=0)
                pass
            nk = njk.sum(axis=0)
            tlog = np.zeros((2, 2, dof))
            tlog.fill(np.nan)
            for k in range(dof):
                tx = np.array([nik[:, k]]).T
                ty = np.array([njk[:, k]])
                tdijk = tx.dot(ty)
                tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
                pass
            return (nijk, tlog)

        _logger.debug('Edge %d -- %d with subset: %s' % (x, y, s))
        row_size = data.shape[0]
        s_size = len(s)
        dof = int(pow(2, s_size))
        row_size_required = 10 * dof
        if row_size < row_size_required:
            _logger.warning('Not enough samples. %s is too small. Need %s.'
                            % (str(row_size), str(row_size_required)))
            return 1
        nijk = None
        if s_size < 6:
            if s_size == 0:
                nijk = np.zeros((2, 2))
                for row_index in range(0, data.shape[0]):
                    i = data[row_index, x]
                    j = data[row_index, y]
                    nijk[i, j] += 1
                    pass
                tx = np.array([nijk.sum(axis=1)]).T
                ty = np.array([nijk.sum(axis=0)])
                tdij = tx.dot(ty)
                tlog = nijk * row_size / tdij
                pass
            if s_size > 0:
                nijk, tlog = _calculate_tlog(x, y, s, dof, data)
                pass
            pass
        else:
            nijk = np.zeros((2, 2, 1))
            i = data[0, x]
            j = data[0, y]
            k = []
            for z in s:
                k.append(data[:, z])
                pass
            k = np.array(k).T
            parents_count = 1
            parents_val = np.array([k[0, :]])
            nijk[i, j, parents_count - 1] = 1
            for it_sample in range(1, row_size):
                is_new = True
                i = data[it_sample, x]
                j = data[it_sample, y]
                tcomp = parents_val[:parents_count, :] == k[it_sample, :]
                for it_parents in range(parents_count):
                    if np.all(tcomp[it_parents, :]):
                        nijk[i, j, it_parents] += 1
                        is_new = False
                        break
                    pass
                if is_new is True:
                    parents_count += 1
                    parents_val = np.r_[parents_val, [k[it_sample, :]]]
                    nnijk = np.zeros((2, 2, parents_count))
                    for p in range(parents_count - 1):
                        nnijk[:, :, p] = nijk[:, :, p]
                    nnijk[i, j, parents_count - 1] = 1
                    nijk = nnijk
                    pass
                pass
            nik = np.ndarray((2, parents_count))
            njk = np.ndarray((2, parents_count))
            for k_index in range(parents_count):
                nik[:, k_index] = nijk[:, :, k_index].sum(axis=1)
                njk[:, k_index] = nijk[:, :, k_index].sum(axis=0)
                pass
            nk = njk.sum(axis=0)
            tlog = np.zeros((2, 2, parents_count))
            tlog.fill(np.nan)
            for k in range(parents_count):
                tX = np.array([nik[:, k]]).T
                tY = np.array([njk[:, k]])
                tdijk = tX.dot(tY)
                tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
                pass
            pass
        log_tlog = np.log(tlog)
        G2 = np.nansum(2 * nijk * log_tlog)
        _logger.debug('G2 = %f' % G2)
        p_val = chi2.sf(G2, dof)
        _logger.info('p_val = %s' % str(p_val))
        return p_val

    @staticmethod
    def _g_square_discrete(data, x, y, s, levels):
        """
        G square test for discrete data.
        Args:
            data: the data matrix to be used (as a numpy.ndarray).
            x: the first node (as an integer).
            y: the second node (as an integer).
            s: the set of neighbouring nodes of x and y (as a set()).
            levels: levels of each column in the data matrix
                (as a list()).
        Returns:
            p_val: the p-value of conditional independence.
        """

        def _calculate_tlog(x, y, s, dof, levels, dm):
            prod_levels = np.prod(list(map(lambda x: levels[x], s)))
            nijk = np.zeros((levels[x], levels[y], prod_levels))
            s_size = len(s)
            z = []
            for z_index in range(s_size):
                z.append(s.pop())
                pass
            for row_index in range(dm.shape[0]):
                i = dm[row_index, x]
                j = dm[row_index, y]
                k = []
                k_index = 0
                for s_index in range(s_size):
                    if s_index == 0:
                        k_index += dm[row_index, z[s_index]]
                    else:
                        lprod = np.prod(list(map(lambda x: levels[x], z[:s_index])))
                        k_index += (dm[row_index, z[s_index]] * lprod)
                        pass
                    pass
                nijk[i, j, k_index] += 1
                pass
            nik = np.ndarray((levels[x], prod_levels))
            njk = np.ndarray((levels[y], prod_levels))
            for k_index in range(prod_levels):
                nik[:, k_index] = nijk[:, :, k_index].sum(axis=1)
                njk[:, k_index] = nijk[:, :, k_index].sum(axis=0)
                pass
            nk = njk.sum(axis=0)
            tlog = np.zeros((levels[x], levels[y], prod_levels))
            tlog.fill(np.nan)
            for k in range(prod_levels):
                tx = np.array([nik[:, k]]).T
                ty = np.array([njk[:, k]])
                tdijk = tx.dot(ty)
                tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
                pass
            return (nijk, tlog)

        _logger.debug('Edge %d -- %d with subset: %s' % (x, y, s))
        row_size = data.shape[0]
        s_size = len(s)
        dof = ((levels[x] - 1) * (levels[y] - 1)
               * np.prod(list(map(lambda x: levels[x], s))))
        row_size_required = 10 * dof
        if row_size < row_size_required:
            _logger.warning('Not enough samples. %s is too small. Need %s.'
                            % (str(row_size), str(row_size_required)))
            return 1
        nijk = None
        if s_size < 5:
            if s_size == 0:
                nijk = np.zeros((levels[x], levels[y]))
                for row_index in range(row_size):
                    i = data[row_index, x]
                    j = data[row_index, y]
                    nijk[i, j] += 1
                    pass
                tx = np.array([nijk.sum(axis=1)]).T
                ty = np.array([nijk.sum(axis=0)])
                tdij = tx.dot(ty)
                tlog = nijk * row_size / tdij
                pass
            if s_size > 0:
                nijk, tlog = _calculate_tlog(x, y, s, dof, levels, data)
                pass
            pass
        else:
            nijk = np.zeros((levels[x], levels[y], 1))
            i = data[0, x]
            j = data[0, y]
            k = []
            for z in s:
                k.append(data[:, z])
                pass
            k = np.array(k).T
            parents_count = 1
            parents_val = np.array([k[0, :]])
            nijk[i, j, parents_count - 1] = 1
            for it_sample in range(1, row_size):
                is_new = True
                i = data[it_sample, x]
                j = data[it_sample, y]
                tcomp = parents_val[:parents_count, :] == k[it_sample, :]
                for it_parents in range(parents_count):
                    if np.all(tcomp[it_parents, :]):
                        nijk[i, j, it_parents] += 1
                        is_new = False
                        break
                    pass
                if is_new is True:
                    parents_count += 1
                    parents_val = np.r_[parents_val, [k[it_sample, :]]]
                    nnijk = np.zeros((levels[x], levels[y], parents_count))
                    for p in range(parents_count - 1):
                        nnijk[:, :, p] = nijk[:, :, p]
                        pass
                    nnijk[i, j, parents_count - 1] = 1
                    nijk = nnijk
                    pass
                pass
            nik = np.ndarray((levels[x], parents_count))
            njk = np.ndarray((levels[y], parents_count))
            for k_index in range(parents_count):
                nik[:, k_index] = nijk[:, :, k_index].sum(axis=1)
                njk[:, k_index] = nijk[:, :, k_index].sum(axis=0)
                pass
            nk = njk.sum(axis=0)
            tlog = np.zeros((levels[x], levels[y], parents_count))
            tlog.fill(np.nan)
            for k in range(parents_count):
                tx = np.array([nik[:, k]]).T
                ty = np.array([njk[:, k]])
                tdijk = tx.dot(ty)
                tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
                pass
            pass
        log_tlog = np.log(tlog)
        G2 = np.nansum(2 * nijk * log_tlog)
        _logger.debug('G2 = %f' % G2)
        if dof == 0:
            # dof can be 0 when levels[x] or levels[y] is 1, which is
            # the case that the values of columns x or y are all 0.
            p_val = 1
        else:
            p_val = chi2.sf(G2, dof)
        _logger.info('p_val = %s' % str(p_val))
        return p_val
