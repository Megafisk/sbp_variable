from matplotlib import pyplot as plt
import scipy.sparse as spsp
import scipy.sparse.linalg as splg
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import D2_Variable as D2Var
import rungekutta4 as rk4

from grid import Grid
from plotting import plot_v

import time


def initial_gaussian(g, sigma, x0, y0, a):
    v0 = a * np.exp(-(g.x - x0) ** 2 / sigma ** 2 - (g.y - y0) ** 2 / sigma ** 2)
    v_t = np.zeros((g.N, 1))
    u = np.vstack((v0, v_t))
    return u


def initial_zero(N):
    return np.zeros((2 * N, 1))


def inflow_wave(freq, amp):
    omega = 2 * np.pi * freq

    def g(t): return amp * omega * np.sin(omega * t)

    return g


def inflow_gaussian(amp, w, t0):
    def g(t): return amp * -2 * (t - t0) / (w ** 2) * np.exp(-(t - t0) ** 2 / (w ** 2))

    return g


def calc_timestep(order, A, B, G: Grid, mb_ref=6, margin=0.5):
    """
    Calculates timestep from the D-operator for a coarse grid with mb = mb_ref points
    per block, then assuming linear converge. Will slightly overestimate the correct value by maybe 0-2%.
    """
    a_interp = RegularGridInterpolator((G.xvec, G.yvec), A)
    b_interp = RegularGridInterpolator((G.xvec, G.yvec), B)
    cg = Grid(mb_ref)
    a = a_interp(cg.xy)
    b = b_interp(cg.xy)
    ops_1d_coarse = D2Var.D2_Variable(cg.m, cg.h, order)
    H, HI, D1, D2_fun, e_l, e_r, d1_l, d1_r = ops_1d_coarse
    HH, HHI, (D2x, D2y), (eW, eE, eS, eN), (d1_W, d1_E, d1_S, d1_N) = D2Var.ops_2d(cg.m, b, ops_1d_coarse)
    AAI = spsp.diags(1 / a)
    D = AAI @ (D2x + D2y) - AAI @ HHI @ (
            - eW @ H @ spsp.diags(eW.T @ a) @ d1_W.T + eE @ H @ spsp.diags(eE.T @ a) @ d1_E.T
            + eN @ H @ spsp.diags(eN.T @ a) @ d1_N.T - eS @ H @ spsp.diags(eS.T @ a) @ d1_S.T)
    c = 1 / np.sqrt(abs(splg.eigs(D, 1)[0][0])) / cg.h  # 0.355, 0.31, 0.188 for order 2, 4, 6 for ref problem
    return margin * 2.8 * c * G.h


def build_ops(order, A, B, g, grid, output=True, retops=False, ops=None):
    m, N, h = grid.mnh()
    a = A.reshape((N,))
    b = B.reshape((N,))

    if output:
        print(f'building order {order} operators with m={grid.m} points...')
    if ops is None:
        ops_1d = D2Var.D2_Variable(m, h, order)
        ops_2d = D2Var.ops_2d(m, b, ops_1d)
    else:
        ops_1d, ops_2d = ops
    H, HI, D1, D2_fun, e_l, e_r, d1_l, d1_r = ops_1d
    HH, HHI, (D2x, D2y), (eW, eE, eS, eN), (d1_W, d1_E, d1_S, d1_N) = ops_2d

    if output:
        print('building DD...')

    AAI = spsp.diags(1 / a)

    tau_E = spsp.diags(np.sqrt(A * B)[-1, :])
    E = - AAI @ HHI @ eE @ H @ tau_E @ eE.T
    G = AAI @ HHI @ eW @ H
    D = AAI @ (D2x + D2y) - AAI @ HHI @ (
            - eW @ H @ spsp.diags(B[0, :]) @ d1_W.T + eE @ H @ spsp.diags(B[-1, :]) @ d1_E.T
            + eN @ H @ spsp.diags(B[:, -1]) @ d1_N.T - eS @ H @ spsp.diags(B[:, 0]) @ d1_S.T)

    DD = spsp.bmat(((None, spsp.eye(N)), (D, E)))
    zeros_N = np.zeros((N, 1))

    def rhs(t, u):
        return DD @ u + np.vstack((zeros_N, G @ g(t)))

    if output:
        print('operators done!')
    if not retops:
        return rhs
    return DD, (D, G, E, AAI), ops_1d, ops_2d


def plot_every(interval, img, title, m, ht):
    if isinstance(interval, float):
        interval = int(np.ceil(interval / ht))  # TODO use same generation as save_every in rk4

    def u_plot_every_n(ts: rk4.RK4Timestepper):
        if (ts.t_i % interval) == 0 or ts.t_i == ts.mt:
            img.set_array(ts.v().reshape((m, m), order='F'))
            title.set_text(f't = {round(ts.t, 2):.2f}')
            plt.pause(0.01)

    return u_plot_every_n


def wave_block(g, a_center, b_center, a0=1, b0=1, block_type='outer', block_margin=1, **kwargs):
    A = np.ones(g.shape) * a0
    B = np.ones(g.shape) * b0
    if g.m % 3 != 1:
        i1 = np.argmax(g.xvec > 1 / 3) - (block_margin - 1)
        i2 = np.argmax(g.xvec > 2 / 3) + (block_margin - 1)
        sl = slice(i1, i2)
    elif block_type == 'outer':
        sl = slice(g.mb + 1 - block_margin, 2 * g.mb + block_margin)
    elif block_type == 'inner':
        sl = slice(g.mb + block_margin, 2 * g.mb + 1 - block_margin)
    else:
        raise ValueError(f'Invalid block type: {block_type}')
    A[sl, sl] = a_center
    B[sl, sl] = b_center

    return A, B


def mixed_block(g: Grid, order, a_center, b_center, block_margin=1, **kwargs):
    si = slice(g.mb + block_margin, 2 * g.mb + 1 - block_margin)
    so = slice(g.mb, 2 * g.mb + 1)

    Ao = np.ones(g.shape)  # use inner for A
    Ao[so, so] = a_center
    Bo = np.ones(g.shape)
    Bo[so, so] = b_center

    Bw = np.ones(g.shape)
    Bt = np.ones(g.shape)
    wide = np.zeros(g.shape, bool)
    tall = np.zeros(g.shape, bool)
    wide[so, si] = True
    tall[si, so] = True
    Bw[wide] = b_center
    Bt[tall] = b_center

    # inner, so Dx should be tall, and Dy wide
    ops_1d = D2Var.D2_Variable(g.m, g.h, order)
    ops_2d = list(D2Var.ops_2d(g.m, Bt.reshape((g.N,)), ops_1d))
    D2y_wide = D2Var.ops_2d(g.m, Bw.reshape((g.N,)), ops_1d)[2][1]

    ops_2d[2] = (ops_2d[2][0], D2y_wide)
    return Ao, Bo, (ops_1d, ops_2d)


def reference_problem(mb, T, order, a_center, b_center, freq, amp, draw_every=-1, save_every=-1,
                      zlim=(-0.4, 0.4), ht=None, is_mb=True, margin=0.5, block_type='outer', **kwargs):
    start = time.time()
    grid = Grid(mb, is_mb)
    m, N, h, X, Y, x, y = grid.params()

    ops = None
    # define wave speeds
    if isinstance(a_center, np.ndarray):
        A, B = a_center, b_center
    elif block_type == 'mixed':
        A, B, ops = mixed_block(grid, order, a_center, b_center, **kwargs)
    else:
        A, B = wave_block(grid, a_center, b_center, block_type=block_type, **kwargs)

    u0 = initial_zero(N)
    if not isinstance(freq, (int, float)):
        g = freq
    else:
        g = inflow_wave(freq, amp)

    rhs = build_ops(order, A, B, g, grid, ops=ops, **kwargs)

    if ht is None:
        ht = calc_timestep(order, A, B, grid, margin=margin)
    print(ht)

    if draw_every > 0:
        fig, ax, img = plot_v(u0[:N], g, zlim, draw_block=True)
        title = plt.title("t = 0.00")
        plt.draw()
        plt.pause(0.5)

        update = plot_every(draw_every, img, title, m, ht)
    else:
        def update(*args):
            pass

    if save_every is True:
        save_every = 1
    ts = rk4.RK4Timestepper(T, ht, rhs, u0, update, save_every)
    ts.run_sim()

    print(f'Done, elapsed time={time.time() - start:.3f} s')

    return ts, grid
