import grid
import var_b
from plotting import plot_v
from grid import Grid
import numpy as np
import matplotlib.pyplot as plt
import rungekutta4 as rk4
import D2_Variable as D2Var
from scipy.interpolate import interpn


def run_ref_prob(mb, order=4, **kwargs):
    T = 0.75
    a_center = 10
    b_center = 100
    # a_center = b_center = 1
    freq = 3
    amp = 0.1

    ts, g = var_b.reference_problem(mb, T, order, a_center, b_center, freq, amp, **kwargs)

    return ts, g


def have_fun():
    # mb = 30
    T = 5
    order = 2
    draw_every_n = 0.01

    # g = Grid(mb)
    Bc = plt.imread('B.tiff') / 255
    gc = grid.Grid(Bc.shape[0], False)
    g = grid.Grid(200, False)

    m, N, h, X, Y, x, y = g.params()

    # define wave speeds
    a0 = 1
    a1 = 2
    b0 = 1
    b1 = 0.2
    A = np.ones((m, m)) * a0
    # B = np.ones((m, m)) * b0
    # B[mb:mb + 4, :] = b1  # block of different wave speeds
    # A[mb:2*mb, mb:2*mb] = a1
    # B[(X - Y < 1 / 3) & (X - Y > 0) & (1 / 3 < X) & (X < 2 / 3) & (1 / 3 < Y) & (Y < 2 / 3)] = b1
    # B[(1 / 3 < X) & (X < 2 / 3)] = b1

    gc = grid.Grid(20, False)
    Bc = 0.03 + 0.2 * np.random.random(gc.shape)
    # Bc[Bc == 0] = b1
    B = interpn((gc.xvec, gc.yvec), Bc, g.xy, method='nearest').reshape(g.shape)
    B[X < 0.4] = 1
    plot_v(B, g.m, (0, np.max(B)))

    zlow = -0.4
    zhigh = 0.4

    # initial data
    sigma = 0.05
    x0 = 0.1
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
    g_in = var_b.inflow_wave(freq, amp)

    # time stuff
    # print("calculating eigs...")
    # ht = 0.5 * 2.8 / np.sqrt(abs(spsplg.eigs(D, 1)[0][0]))
    # print("eigs done!")
    ht = 0.42 / g.m

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
    # have_fun()
    # ts, g = var_b.reference_problem(40, 2, 2, 1, 0.3, 3, 0.1, draw_every=0.01, source_pos=(0.123, 0.453), phi_order=4)
    # run_ref_prob(90, 2, is_mb=False)

    plt.show()
    # print(test_calc_timestep(21, 4))
    pass
