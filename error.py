import matplotlib.pyplot as plt
import numpy as np

from grid import Grid
import ref
import var_b
import plotting
import rungekutta4 as rk4
import var_b_test


def io(mb, T, ac, bc, ht, order=4, **kwargs):
    g = Grid(mb)

    u0 = var_b.initial_zero(g.N)
    g_in = var_b.inflow_wave(3, 0.1)

    Ai, Bi = var_b.wave_block(g, ac, bc, 'inner')
    Ao, Bo = var_b.wave_block(g, ac, bc, 'outer')
    ts_inner = rk4.RK4Timestepper(T, ht, var_b.build_ops(order, Ai, Bi, g_in, g), u0, **kwargs).run_sim()
    ts_outer = rk4.RK4Timestepper(T, ht, var_b.build_ops(order, Ao, Bo, g_in, g), u0, **kwargs).run_sim()
    return g, ts_inner, ts_outer


def inner_outer(comp_fn):
    vl, params = ref.load_reference(comp_fn)
    g_ref = Grid(params['mb'] - 1)  # 60 * 3 + 1
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
    g_ref = Grid(params['mb'] - 1)  # 60 * 3 + 1
    if 'save_every' in kwargs:
        kwargs['ht'] = params['dt']
        if kwargs['save_every'] is True:
            kwargs['save_every'] = params['frames']
            fi = params['frames']
        else:
            fi = np.in1d(params['frames'], kwargs['save_every']).nonzero()[0]
    else:
        fi = [-1]

    res = [var_b.reference_problem(mb, params['T'], order, params['a_center'], params['b_center'], 3, 0.1, **kwargs)
           for mb in mbs]

    grids = [g for _, g in res]
    vls = [ts.vl if 'save_every' in kwargs else ts.v() for ts, _ in res]
    ers = ref.errors_interp(grids, vls, vl_ref[:, fi], g_ref)
    hs = Grid.get_attr(grids, 'h').reshape(ers.shape)
    return ers, grids, vls, hs


def append_ers(vl_ref, params, ers: np.ndarray, grids, vls, mbs, new_mbs, **kwargs):
    e, g, v, _ = close_comp(vl_ref, params, new_mbs, **kwargs)
    return np.hstack((ers, e)), g + grids, vls + v, mbs + new_mbs


def grid_variants(comp_fn, output_fn):
    mbs = np.array([15, 25, 50, 90, 150])
    mv = [3 * mbs, 3 * mbs + 1, 3 * mbs + 2]

    vl_ref, g_ref, params = ref.load_reference(comp_fn)
    es, gs, vls = zip(*[close_comp(vl_ref, params, ml, is_mb=False) for ml in mv])
    es = np.vstack(es)
    mv = np.vstack(mv)
    gs = np.vstack(gs)
    q = ref.calculate_q(es, mv)
    ref.save_conv(output_fn, es, mbs, 2, q, params, grid_variants='3mb 3mb+1 3mb+2')
    plotting.plot_errors(gs, es)
    plt.show()


def compare_orders(vl_ref, params, **kwargs):
    """Compares operators of different orders"""
    mbs = [10, 20, 30, 40, 60, 120]
    orders = [2, 4, 6]
    gs = [Grid(mb) for mb in mbs]

    g_ref = Grid(params['mb'] - 1)
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
    g_ref = Grid(params['mb'] - 1)
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


# same jump in c but with different b
def compare_params(**kwargs):
    mb = 10
    g = Grid(mb)

    mags = np.arange(-4, 7)
    vs = [var_b.reference_problem(mb, 0.42, a_center=10. ** (mag - 2), b_center=10. ** mag, freq=2.6, amp=0.1,
                                  order=4, **kwargs)[0].v()
          for mag in mags]
    ls = [g.get_hor_line(v, 0.5) for v in vs]


# ers2, _, vs2, _ = error.close_comp(vl_ref, params, mbs, order=2, block_type='inner', block_margin=1)
# ls += ax.plot(hs, ers2 / vnorm, label='')
# plt.legend()
# ers = np.hstack((ers, ers2))
# vls.append(vs2)


if __name__ == '__main__':
    # grid_variants('ref450l', 'ref450l grid variants')
    # vl_ref, g_ref, params = ref.load_reference('faster180')
    # mbs = np.vstack([6, 10, 15, 20, 30, 45, 60])
    # close_comp(vl_ref, params, mbs.flatten(), order=2, block_type='mixed', block_margin=1)
    pass
