import grid
import var_b
from plotting import plot_v
from grid import Grid
import numpy as np
import matplotlib.pyplot as plt
import rungekutta4 as rk4
import D2_Variable as D2Var


def block_corner_inner(mb, order, T, a_center, b_center, block_margin=1, **kwargs):
    g = grid.Grid(mb)
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

    ts, g = var_b.reference_problem(g.mb, T, order, Ao, Bo, 3, 0.1, ops=(ops_1d, ops_2d), **kwargs)
    return ts, g


def run_ref_prob(mb, order=4, **kwargs):
    T = 0.75
    a_center = 10
    b_center = 1000
    # a_center = b_center = 1
    freq = 3
    amp = 0.1

    ts, g = var_b.reference_problem(mb, T, order, a_center, b_center, freq, amp, **kwargs)

    return ts, g


def have_fun():
    mb = 30
    T = 5
    order = 2
    draw_every_n = 0.01

    g = Grid(mb)
    m, N, h, X, Y, x, y = g.params()

    # define wave speeds
    a0 = 1
    a1 = 2
    b0 = 1
    b1 = 0.2
    A = np.ones((m, m)) * a0
    B = np.ones((m, m)) * b0
    # B[mb:2 * mb + 1, mb:2 * mb + 1] = b1  # block of different wave speeds
    # A[mb:2*mb, mb:2*mb] = a1
    # B[(X - Y < 1 / 3) & (X - Y > 0) & (1 / 3 < X) & (X < 2 / 3) & (1 / 3 < Y) & (Y < 2 / 3)] = b1
    # B[(1 / 3 < X) & (X < 2 / 3)] = b1

    zlow = -0.4
    zhigh = 0.4

    # initial data
    sigma = 0.05
    x0 = 0.6
    y0 = 0.1
    # u0 = initial_zero(N)
    u0 = var_b.initial_gaussian(g, sigma, x0, y0, 1)

    # gaussian inflow data
    # t0 = 0.25
    # w = 0.1
    # amp = 0.4
    # g = inflow_gaussian(m, amp, w, t0)

    # wave inflow data
    freq = 3
    amp = 0
    g_in = var_b.inflow_wave(m, freq, amp)

    # time stuff
    # print("calculating eigs...")
    # ht = 0.5 * 2.8 / np.sqrt(abs(spsplg.eigs(D, 1)[0][0]))
    # print("eigs done!")
    ht = 0.14 / mb

    rhs = var_b.build_ops(order, A, B, g_in, g)

    fig, ax, img = plot_v(u0[:N], m, (zlow, zhigh), draw_block=False)
    title = plt.title("t = 0.00")
    plt.draw()
    plt.pause(0.5)

    update = var_b.plot_every(draw_every_n, img, title, m, ht)
    ts = rk4.RK4Timestepper(T, ht, rhs, u0, update)
    ts.run_sim()
    plt.show()


def run_sim(A, B, g_in, u0, g, T, draw_every_n=1, order=2, vlim=(-0.5, 0.5)):
    rhs = var_b.build_ops(order, A, B, g_in, g)

    fig, ax, img = plot_v(u0[:g.N], g.m, vlim)
    title = plt.title("t = 0.00")
    plt.draw()
    plt.pause(0.5)

    ht = var_b.calc_timestep(order, A, B, g)
    update = var_b.plot_every(draw_every_n, img, title, g.m, ht)
    ts = rk4.RK4Timestepper(T, ht, rhs, u0, update)
    ts.run_sim()
    plt.show()


def test_calc_timestep(mb, order, a=0.7, b=0.22):
    a0 = 1
    b0 = 1

    g = Grid(mb)
    A = np.ones((g.m, g.m)) * a0
    B = np.ones((g.m, g.m)) * b0
    A, B = var_b.wave_block(g, a, b, a0, b0)
    dt = var_b.calc_timestep(order, A, B, g)
    return dt


def line(g, ac, bc, pos):
    A = np.ones(g.shape)
    B = np.ones(g.shape)
    ind = g.X >= pos
    A[ind] = ac
    B[ind] = bc
    return A, B


if __name__ == '__main__':
    have_fun()
    # ts, g = run_ref_prob(10, 2, draw_every_n=0.01)
    # ts, g = block_corner_inner(2, 10, 100, 10, 0.75, draw_every=0.01)
    # plt.show()

    # print(test_calc_timestep(21, 4))
    pass
