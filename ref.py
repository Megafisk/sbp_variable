import scipy.io as spio
import numpy as np
import numpy.linalg as nplg
from scipy.interpolate import interpn

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


def interpolate(vl, g_ref: Grid, g: Grid):
    """Interpolates vl on grid g_ref onto grid g"""
    if (g_ref.m - 1) % (g.m - 1) == 0:
        # if you can pick every x points from g_ref, just pick the corresponding points
        m_ref = g_ref.m
        N_ref = g_ref.N
        ind = np.arange(N_ref).reshape((m_ref, m_ref))
        x_ind = np.rint(g.x * (m_ref - 1)).astype('int64')
        y_ind = np.rint(g.y * (m_ref - 1)).astype('int64')
        rows = ind[x_ind, y_ind].reshape((g.N,))
        return vl[rows, :]
    else:
        p = (g_ref.xvec, g_ref.yvec)
        return np.hstack([interpn(p, v.reshape(g_ref.shape), g.xy, method='cubic').reshape(g.N, 1) for v in vl.T])


def error(v, v_ref, g: Grid):
    return nplg.norm(v - v_ref, axis=0) * g.h


def error_interp(v, v_ref, g, g_ref):
    v_ref_i = interpolate(v_ref, g_ref, g)  # v_ref interpolated onto g
    return error(v, v_ref_i, g)


def errors_interp(gs, vs, v_ref, g_ref):
    return np.vstack([error_interp(vs[i], v_ref, gs[i], g_ref) for i in range(len(vs))]).T
