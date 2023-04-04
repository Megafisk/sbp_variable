from var_b import *


if __name__ == '__main__':
    mb = 20
    T = 2
    order = 2
    draw_every_n = 5

    m = 3 * mb
    N = m * m
    h, X, Y, x, y = grid(m)

    # define wave speeds
    a0 = 1
    a1 = 1
    b0 = 1
    b1 = 0.25
    A = np.ones((m, m)) * a0
    B = np.ones((m, m)) * b0
    B[mb:2*mb+1, mb:2*mb+1] = b1  # block of different wave speeds
    # A[mb:2*mb, mb:2*mb] = a1
    # B[(X - Y < 1 / 3) & (X - Y > 0) & (1 / 3 < X) & (X < 2 / 3) & (1 / 3 < Y) & (Y < 2 / 3)] = b1
    # B[(Y > 1 / 3) & (1 / 3 < X) & (X < 2 / 3)] = b1  # SKAPAR INSTABIL!

    zlow = -0.4
    zhigh = 0.4

    # initial data
    sigma = 0.05
    x0 = 0.6
    y0 = 0.1
    u0 = initial_zero(N)
    # u0 = initial_gaussian(x, y, N, sigma, x0, y0)

    # gaussian inflow data
    # t0 = 0.25
    # w = 0.1
    # amp = 0.4
    # g = inflow_gaussian(m, amp, w, t0)

    # wave inflow data
    freq = 3
    amp = 0.1
    g = inflow_wave(m, freq, amp)

    # time stuff
    # print("calculating eigs...")
    # ht = 0.5 * 2.8 / np.sqrt(abs(spsplg.eigs(D, 1)[0][0]))
    # print("eigs done!")
    ht = 0.14 / mb

    rhs = build_ops(order, A, B, g, N, m, h)

    fig, ax, img = plot_v(u0[:N], m, (zlow, zhigh))
    title = plt.title("t = 0.00")
    plt.draw()
    plt.pause(0.5)

    update = plot_every(draw_every_n, img, title, N, m)

    run_sim(u0, rhs, T, ht, update)
    plt.show()


