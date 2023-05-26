import grid
import plotting
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


FOLDER_WEIRD = '../komplicerat/'


def have_fun(fn, T, **kwargs):
    # mb = 30
    order = 2
    draw_every_n = -0.01

    # g = Grid(mb)
    Bc = plt.imread(FOLDER_WEIRD + fn) / 255
    Bc = np.rot90(Bc, k=-1)
    # Bc = np.flipud(Bc)
    # Bc = np.fliplr(Bc)
    gc = grid.Grid(Bc.shape[0], False)
    g = grid.Grid(gc.m, False)

    m, N, h, X, Y, x, y = g.params()

    # define wave speeds
    a0 = 1
    a1 = 10
    b0 = 1
    b1 = 1000
    A = np.ones((m, m)) * a0
    # B = np.ones((m, m)) * b0
    # B[mb:mb + 4, :] = b1  # block of different wave speeds
    # A[mb:2*mb, mb:2*mb] = a1
    # B[(X - Y < 1 / 3) & (X - Y > 0) & (1 / 3 < X) & (X < 2 / 3) & (1 / 3 < Y) & (Y < 2 / 3)] = b1
    # B[(1 / 3 < X) & (X < 2 / 3)] = b1

    # gc = grid.Grid(20, False)
    # Bc = 0.02 + 10 * np.random.random(gc.shape)
    A[Bc == 0] = a1
    Bc[Bc == 0] = b1
    Bc[Bc == 1] = b0
    # A[Bc == 0] = a1
    # B = interpn((gc.xvec, gc.yvec), Bc, g.xy, method='nearest').reshape(g.shape)
    # B[X < 0.4] = 1
    B = Bc.astype(float)
    plot_v(B.reshape((g.N,), order='C'), g, (0, np.max(B)))

    # initial data
    sigma = 0.02
    x0 = 0.1
    y0 = 0.2
    # u0 = var_b.initial_zero(N)
    u0 = var_b.initial_gaussian(g, sigma, x0, y0, 1)

    # gaussian inflow data
    # t0 = 0.25
    # w = 0.1
    # amp = 0.4
    # g = inflow_gaussian(m, amp, w, t0)

    # wave inflow data
    # freq = 3
    # amp = 0
    # g_in = var_b.inflow_wave(freq, amp)

    return var_b.reference_problem(g.m, T, order, A, B, is_mb=False, **kwargs)
    # plt.show()


def run_sim(A, B, g_in, u0, g, T, draw_every_n=1, order=2, vlim=(-0.5, 0.5)):
    rhs = var_b.build_ops(order, A, B, g_in, g)

    fig, ax, img = plot_v(u0[:g.N], g, vlim)
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
    ts, g = have_fun('waveguide.tiff', 1, amp=0, freq=5, save_every=0.005)
    # anim = plotting.plot_anim(ts.vl, g)
    # have_fun(draw_every=0.01)
    # ts, g = var_b.reference_problem(20, 3, 2, 1e2, 1e4, 6, 2, save_every=0.005)
    plotting.plot_anim(ts.vl, g, interval=50, cmap='gray')
    # run_ref_prob(90, 2, is_mb=False)

    # plt.show()
    # print(test_calc_timestep(21, 4))
    pass
