import scipy.io as spio
import numpy as np


def load_reference(filename):
    ref = spio.loadmat(filename)['ref']
    v_ref = ref['v'][0, 0]
    p = ref['points'][0, 0]

    mb = int(ref['mb'])
    m_ref = 3 * mb
    m = m_ref - 2

    ind = np.rint(p * (m - 1)).astype('int64')

    v = np.zeros((m, m))
    for (i, (xi, yi)) in enumerate(ind):
        v[xi, yi] = v_ref[i]
    v = v.reshape((m * m, 1))

    params = {key: float(ref[key]) for key in ('a_center', 'b_center', 'freq', 'amp', 't')}
    params['m'] = m
    params['mb'] = mb
    return v, params


def interpolate_ref(ref_fn, x, y):
    v, params = load_reference(ref_fn)
    m_ref = params['m']
    ref_2d = v.reshape((m_ref, m_ref))
    x_ind = np.rint(x * (m_ref - 1)).astype('int64')
    y_ind = np.rint(y * (m_ref - 1)).astype('int64')
    ref_interp = ref_2d[x_ind, y_ind]
    return ref_interp
