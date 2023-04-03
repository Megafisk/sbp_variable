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
        v[yi, xi] = v_ref[i]

    params = {key: float(ref[key]) for key in ('a_center', 'b_center', 'freq', 'amp', 't')}
    params['m'] = m
    params['mb'] = mb
    return v, params
