from matplotlib import pyplot as plt
import scipy.sparse as spsp
import numpy as np
import D2_Variable as D2Var
import rungekutta4 as rk4


def plot_v(v, m, vlim=(-0.4, 0.4)):
    ax: plt.Axes
    fig: plt.Figure
    fig, ax = plt.subplots()
    img = ax.imshow(v.reshape((m, m), order='F'),
                    origin='lower',
                    extent=[0, 1, 0, 1],
                    vmin=vlim[0], vmax=vlim[1])
    fig.colorbar(img, ax=ax)
    plt.xlabel("x")
    plt.ylabel("y")
    return fig, ax, img


def grid(m):
    xvec, h = np.linspace(0, 1, m, retstep=True)
    yvec = np.linspace(0, 1, m)
    X, Y = np.meshgrid(xvec, yvec, indexing='ij')
    x = X.reshape((m * m, 1))
    y = Y.reshape((m * m, 1))

    return h, X, Y, x, y


def initial_gaussian(x, y, N, sigma, x0, y0):
    v0 = np.exp(-(x - x0) ** 2 / sigma ** 2 - (y - y0) ** 2 / sigma ** 2)
    v_t = np.zeros((N, 1))
    u = np.vstack((v0, v_t))
    return u


def initial_zero(N):
    return np.zeros((2 * N, 1))


def inflow_wave(m, freq, amp):
    omega = 2 * np.pi * freq

    def g(t): return amp * omega * np.sin(freq * 2 * np.pi * np.ones((m, 1)) * t)

    return g


def inflow_gaussian(m, amp, w, t0):
    def g(t): return amp * -2 * (t - t0) / (w ** 2) * np.exp(-(t - t0) ** 2 / (w ** 2)) * np.ones((m, 1))

    return g


def build_ops(order, A, B, g, N, m, h):
    a = A.reshape((N,))
    b = B.reshape((N,))

    print('building D2...')
    ops_1d = D2Var.D2_Variable(m, h, order)
    H, HI, D1, D2_fun, e_l, e_r, d1_l, d1_r = ops_1d
    HH, HHI, (D2x, D2y), (eW, eE, eS, eN), (d1_W, d1_E, d1_S, d1_N) = D2Var.ops_2d(m, b, ops_1d)
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

    def rhs(t, u): return DD @ u + np.vstack((zeros_N, G @ g(t)))

    print('operators done!')
    return rhs


def run_sim(u0, rhs, T, ht, update=lambda u, t, t_i, mt: None):
    mt = int(np.ceil(T / ht) + 1)
    tvec, ht = np.linspace(0, T, mt, retstep=True)

    u = u0
    t = 0
    for t_i in range(mt - 1):
        # Take one step with the fourth order Runge-Kutta method.
        u, t = rk4.step(rhs, u, t, ht)
        update(u, t, t_i, mt)

    return u, t


def plot_every(interval, img, title, N, m):
    def u_plot_every_n(u, t, t_i, mt):
        if (t_i % interval) == 0 or t_i == mt - 2:
            v = u[:N]
            img.set_array(v.reshape((m, m), order='F'))
            title.set_text(f't = {t:.2f}')
            plt.pause(0.01)

    return u_plot_every_n
