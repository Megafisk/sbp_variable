import scipy.io as spio
import numpy as np
import numpy.linalg as nplg
from scipy.interpolate import interpn
from scipy.optimize import curve_fit

from var_b import Grid

FOLDER = '/Users/hermanbergkvist/Documents/Kanddata/'
REF_FOLDER = FOLDER + 'ref_sols/'
CALC_FOLDER = FOLDER + 'calc/'


def load_reference(filename, folder=REF_FOLDER):
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

    params = {key: float(ref[key]) for key in ('a_center', 'b_center', 'freq', 'amp', 't', 'T', 'dt')} \
             | {key: int(ref[key]) for key in ('mb', 'mt')}
    params['m'] = m
    params['N'] = m * m
    params['frames'] = ref['saved'][0, 0].ravel()
    params['saved_times'] = np.linspace(0, params['T'], params['mt'] + 1)[params['frames']]
    if 'source_pos' in ref.dtype.fields:
        params['source_pos'] = ref['source_pos'][0, 0]
    g_ref = Grid(m, False)
    return v_list, g_ref, params


# def prob_params(params, ht=False, frames):
#     return {k: params[k] for k in {'a_center', 'b_center', ''}}


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
        return np.hstack([interpn(p, v.reshape(g_ref.shape), g.xy, method='splinef2d').reshape(g.N, 1) for v in vl.T])


def error(v, v_ref, g: Grid):
    return nplg.norm(v - v_ref, axis=0) * g.h


def error_interp(v, v_ref, g, g_ref):
    if g.m > g_ref.m:
        g_ref, g = g, g_ref
        v_ref, v = v, v_ref
    v_ref_i = interpolate(v_ref, g_ref, g)  # v_ref interpolated onto g
    return error(v, v_ref_i, g)


def errors_interp(gs, vs, v_ref, g_ref):
    return np.vstack([error_interp(vs[i], v_ref, gs[i], g_ref) for i in range(len(vs))])


def calculate_q(ers, ms):
    if isinstance(ms[0], Grid):
        ms = np.array([g.m for g in ms])
    return -np.log10(ers[:-1, :] / ers[1:, :]) / np.log10(ms[:-1, :] / ms[1:, :])


def error_conv(x, c, q): return c * (x ** q)


def fit_q(hs, ers, q=None):
    return curve_fit(error_conv if q is None else lambda x, c: c * (x ** q), hs.ravel(), ers.ravel())


def save_conv(fn, ers, mbs, order, q, params, **other):
    mdict = {'ers': ers, 'mbs': mbs, 'order': order, 'q': q, **other, **params}
    spio.savemat(CALC_FOLDER + fn, mdict)


def load_conv(fn, folder=CALC_FOLDER): return spio.loadmat(folder + fn)


def parse_str(s: str, to=float):
    return np.array([[to(n.replace(',', '.')) for n in ll.split('\t')] for ll in s.splitlines()])
