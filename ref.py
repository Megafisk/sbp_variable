import scipy.io as spio
import numpy as np
from var_b import Grid


def load_reference(filename, folder='../ref_sols/'):
    ref = spio.loadmat(folder + filename)['ref']
    p = ref['points'][0, 0]

    mb = int(ref['mb'])
    m_ref = 3 * mb
    m = m_ref - 2

    ind = np.rint(p * (m - 1)).astype('int64')
    i = np.arange(len(ind))

    reorder = np.zeros((m, m), 'int32')
    reorder[ind[:, 0], ind[:, 1]] = i
    reorder = reorder.reshape((m * m,))
    v_list = ref['v_list'][0, 0][reorder, :]

    params = {key: float(ref[key]) for key in ('a_center', 'b_center', 'freq', 'amp', 't', 'dt')} \
        | {key: int(ref[key]) for key in ('mb', 'mt')}
    params['m'] = m
    params['N'] = m * m
    return v_list, params


def interpolate_ref(vl, params, grid: Grid):
    m_ref = params['m']
    N_ref = vl.shape[0]
    ind = np.arange(N_ref).reshape((m_ref, m_ref))
    x_ind = np.rint(grid.x * (m_ref - 1)).astype('int64')
    y_ind = np.rint(grid.y * (m_ref - 1)).astype('int64')
    rows = ind[x_ind, y_ind].reshape((grid.N,))
    return vl[rows, :]
