import matplotlib.pyplot as plt
import numpy as np

import grid
import ref
import var_b
import plotting
import rungekutta4 as rk4
import var_b_test


def io(mb, T, ac, bc, ht, order=4, **kwargs):
    g = grid.Grid(mb)

    u0 = var_b.initial_zero(g.N)
    g_in = var_b.inflow_wave(g.m, 3, 0.1)

    Ai, Bi = var_b.wave_block(g, ac, bc, 'inner')
    Ao, Bo = var_b.wave_block(g, ac, bc, 'outer')
    ts_inner = rk4.RK4Timestepper(T, ht, var_b.build_ops(order, Ai, Bi, g_in, g), u0, **kwargs).run_sim()
    ts_outer = rk4.RK4Timestepper(T, ht, var_b.build_ops(order, Ao, Bo, g_in, g), u0, **kwargs).run_sim()
    return g, ts_inner, ts_outer


def inner_outer(comp_fn):
    vl, params = ref.load_reference(comp_fn)
    g_ref = grid.Grid(params['mb'] - 1)  # 60 * 3 + 1
    ac = params['a_center']
    bc = params['b_center']
    ht = params['dt']
    frames = params['frames']
    T = params['T']

    g20, tsi20, tso20 = io(50, T, ac, bc, ht, save_every=frames)
    g40, tsi40, tso40 = io(150, T, ac, bc, ht, save_every=frames)

    ers = ref.errors_interp([g20, g20, g40, g40], [tsi20.vl, tso20.vl, tsi40.vl, tso40.vl], vl, g_ref)
    plt.plot(ers)
    plt.legend(['i20', 'o20', 'i40', 'o40'])
    plt.show()
    # plotting.compare_line(tsi20.t_vec,
    #                       [g_ref, g20, g20, g40, g40],
    #                       [vl, tsi20.vl, tso20.vl, tsi40.vl, tso40.vl],
    #                       ['ref', 'i20', 'o20', 'i40', 'o40'])
    pass


def run_mbs(mbs, **kwargs):
    res = [var_b_test.run_ref_prob(mb, **kwargs)
           for mb in mbs]
    vls = [r[0].vl for r in res]
    grids = [r[1] for r in res]
    return vls, grids


def close_comp(vl_ref, params, mbs, order=2, **kwargs):
    """Compares grids of similar size"""
    g_ref = grid.Grid(params['mb'] - 1)  # 60 * 3 + 1
    if 'ht' not in kwargs:
        kwargs['ht'] = params['dt']
    if 'save_every' in kwargs:
        fi = np.in1d(params['frames'], kwargs['save_every']).nonzero()[0]
    else:
        fi = [-1]

    res = [var_b.reference_problem(mb, params['T'], order, params['a_center'], params['b_center'], 3, 0.1,
                                   **kwargs) for mb in mbs]

    grids = [g for _, g in res]
    vls = [ts.vl if 'save_every' in kwargs else ts.v() for ts, _ in res]
    ers = ref.errors_interp(grids, vls, vl_ref[:, fi], g_ref)
    return ers, grids, vls


def append_ers(vl_ref, params, ers: np.ndarray, grids, vls, mbs, new_mbs, **kwargs):
    e, g, v = close_comp(vl_ref, params, new_mbs, **kwargs)
    return np.hstack((ers, e)), g + grids, vls + v, mbs + new_mbs


def grid_variants(comp_fn):
    mbs = np.array([10, 15, 20, 25, 30, 50])
    mv = [3 * mbs, 3 * mbs + 1, 3 * mbs + 2]

    es, gs, vls = zip(*[close_comp('ref300.mat', ml, is_mb=False, order=2) for ml in mv])


def compare_orders(vl_ref, params, **kwargs):
    """Compares operators of different orders"""
    mbs = [10, 20, 30, 40, 60, 120]
    orders = [2, 4, 6]
    gs = [grid.Grid(mb) for mb in mbs]

    g_ref = grid.Grid(params['mb'] - 1)
    fs = params['frames']

    ers, _, vls = zip(*[close_comp(vl_ref, params, mbs, order=o, save_every=fs) for o in orders])
    ers3 = np.array(ers)[:, 0, :]

    ls = plt.plot(ers)
    plt.legend(['2', '4', '6'])
    ls[0].set_marker('.')
    ls[2].set_marker('^')
    plt.show()
    pass


def compare_timesteps():
    vl, params = ref.load_reference('ref300.mat')
    g_ref = grid.Grid(params['mb'] - 1)
    ac = params['a_center']
    bc = params['b_center']
    ht = params['dt']
    T = 1.9

    mb = 30
    ts1, g = var_b_test.run_ref_prob(mb, 4, draw_every_n=-1, margin=1, save_every=1, ht=2 * ht)
    ts2, _ = var_b_test.run_ref_prob(mb, 4, draw_every_n=-1, save_every=2, ht=ht)
    ts3, _ = var_b_test.run_ref_prob(mb, 4, draw_every_n=-1, save_every=4, ht=ht / 2)

    vvl = vl[:, :2 * ts1.vl.shape[1]:2]
    plotting.compare_line(ts1.t_vec, [g_ref, g, g, g], [vvl, ts1.vl, ts2.vl, ts3.vl], ['ref', 1, 0.5, 0.25])
    pass


def calculate_q(ers, gs):
    ms = np.array([g.m for g in gs]).T
    return -np.log10(ers[:-1] / ers[1:]).T / np.log10(ms[:-1] / ms[1:])

