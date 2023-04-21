from var_b import *
from plotting import plot_v


def run_ref_prob(mb, order=4, **kwargs):
    T = 1.31
    a_center = 0.7
    b_center = 0.22
    # a_center = b_center = 1
    freq = 3
    amp = 0.1

    ts, g = reference_problem(mb, T, order, a_center, b_center, freq, amp, **kwargs)
    print(ts.t)

    return ts, g


def have_fun():
    mb = 30
    T = 5
    order = 2
    draw_every_n = 1

    grid = Grid(mb)
    m, N, h, X, Y, x, y = grid.params()

    # define wave speeds
    a0 = 1
    a1 = 2
    b0 = 1
    b1 = 0.25
    A = np.ones((m, m)) * a0
    B = np.ones((m, m)) * b0
    # B[mb:2 * mb + 1, mb:2 * mb + 1] = b1  # block of different wave speeds
    # A[mb:2*mb, mb:2*mb] = a1
    B[(X - Y < 1 / 3) & (X - Y > 0) & (1 / 3 < X) & (X < 2 / 3) & (1 / 3 < Y) & (Y < 2 / 3)] = b1
    # B[(Y > 1 / 3) & (1 / 3 < X) & (X < 2 / 3)] = b1  # SKAPAR INSTABIL!

    zlow = -0.4
    zhigh = 0.4

    # initial data
    sigma = 0.05
    x0 = 0.6
    y0 = 0.1
    # u0 = initial_zero(N)
    u0 = initial_gaussian(x, y, N, sigma, x0, y0)

    # gaussian inflow data
    # t0 = 0.25
    # w = 0.1
    # amp = 0.4
    # g = inflow_gaussian(m, amp, w, t0)

    # wave inflow data
    freq = 3
    amp = 0
    g = inflow_wave(m, freq, amp)

    # time stuff
    # print("calculating eigs...")
    # ht = 0.5 * 2.8 / np.sqrt(abs(spsplg.eigs(D, 1)[0][0]))
    # print("eigs done!")
    ht = 0.14 / mb

    rhs, _ = build_ops(order, A, B, g, grid)

    fig, ax, img = plot_v(u0[:N], m, (zlow, zhigh))
    title = plt.title("t = 0.00")
    plt.draw()
    plt.pause(0.5)

    update = plot_every(draw_every_n, img, title, m)
    ts = rk4.RK4Timestepper(T, ht, rhs, u0, N, update)
    ts.run_sim()
    plt.show()


def test_calc_timestep(mb, order):
    a0 = 1
    a1 = 0.7
    b0 = 1
    b1 = 0.22

    grid = Grid(mb)
    A = np.ones((grid.m, grid.m)) * a0
    B = np.ones((grid.m, grid.m)) * b0
    A[mb:2 * mb + 1, mb:2 * mb + 1] = a1  # block of different wave speeds
    B[mb:2 * mb + 1, mb:2 * mb + 1] = b1  # block of different wave speeds
    dt = calc_timestep(order, A, B, grid)
    return dt


if __name__ == '__main__':
    # have_fun()
    ts, g = run_ref_prob(30, 4, draw_every_n=1, margin=1)
    plt.show()

    # print(test_calc_timestep(21, 4))
    pass
