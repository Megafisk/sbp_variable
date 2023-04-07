from matplotlib import pyplot as plt
import scipy.sparse as spsp
import scipy.sparse.linalg as splg
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import D2_Variable as D2Var
import rungekutta4 as rk4

from grid import Grid
from plotting import plot_v


def initial_gaussian(x, y, N, sigma, x0, y0):
    v0 = np.exp(-(x - x0) ** 2 / sigma ** 2 - (y - y0) ** 2 / sigma ** 2)
    v_t = np.zeros((N, 1))
    u = np.vstack((v0, v_t))
    return u


def initial_zero(N):
    return np.zeros((2 * N, 1))


def inflow_wave(m, freq, amp):
    omega = 2 * np.pi * freq

    def g(t): return amp * omega * np.sin(omega * np.ones((m, 1)) * t)

    return g


def inflow_gaussian(m, amp, w, t0):
    def g(t): return amp * -2 * (t - t0) / (w ** 2) * np.exp(-(t - t0) ** 2 / (w ** 2)) * np.ones((m, 1))

    return g


def calc_timestep(order, A, B, G: Grid, mb_ref=6):
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
    c = 1 / np.sqrt(abs(splg.eigs(D, 1)[0][0])) / cg.h
    return 0.5 * 2.8 * c * G.h


def build_ops(order, A, B, g, grid, output=True):
    m, N, h = grid.mnh()
    a = A.reshape((N,))
    b = B.reshape((N,))

    if output:
        print('building D2...')
    ops_1d = D2Var.D2_Variable(m, h, order)
    H, HI, D1, D2_fun, e_l, e_r, d1_l, d1_r = ops_1d
    HH, HHI, (D2x, D2y), (eW, eE, eS, eN), (d1_W, d1_E, d1_S, d1_N) = D2Var.ops_2d(m, b, ops_1d)
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
    return rhs


def plot_every(interval, img, title, N, m):
    def u_plot_every_n(ts: rk4.RK4Timestepper):
        if (ts.t_i % interval) == 0 or ts.t_i == ts.mt:
            v = ts.v()
            img.set_array(v.reshape((m, m), order='F'))
            title.set_text(f't = {round(ts.t, 2):.2f}')
            plt.pause(0.01)

    return u_plot_every_n


def wave_block(grid, a_center, b_center, a0=1, b0=1):
    A = np.ones(grid.shape) * a0
    B = np.ones(grid.shape) * b0
    A[grid.mb:2 * grid.mb + 1, grid.mb:2 * grid.mb + 1] = a_center  # block of different wave speeds
    B[grid.mb:2 * grid.mb + 1, grid.mb:2 * grid.mb + 1] = b_center

    return A, B


def reference_problem(mb, T, order, a_center, b_center, freq, amp, draw_every_n=-1, store_vl=False, zlim=(-0.4, 0.4)):
    grid = Grid(mb)
    m, N, h, X, Y, x, y = grid.params()

    # define wave speeds
    A, B = wave_block(grid, a_center, b_center)

    u0 = initial_zero(N)
    g = inflow_wave(m, freq, amp)

    rhs = build_ops(order, A, B, g, grid)

    ht = calc_timestep(order, A, B, grid)
    print(ht)

    if draw_every_n > 0:
        fig, ax, img = plot_v(u0[:N], m, zlim)
        title = plt.title("t = 0.00")
        plt.draw()
        plt.pause(0.5)

        update = plot_every(draw_every_n, img, title, N, m)
    else:
        def update(*args):
            pass

    ts = rk4.RK4Timestepper(T, ht, rhs, u0, update, store_vl)
    ts.run_sim()

    return ts, grid
