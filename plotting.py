import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from grid import Grid


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


def plot_anim(vl: np.ndarray, g: Grid, vlim=(-0.4, 0.4)):
    ax: plt.Axes
    fig: plt.Figure
    fig, ax = plt.subplots()
    plt.xlabel("x")
    plt.ylabel("y")
    ims = []
    for i in range(vl.shape[1]):
        im = ax.imshow(vl[:, i].reshape(g.shape, order='F'),
                       origin='lower',
                       extent=[0, 1, 0, 1],
                       vmin=vlim[0], vmax=vlim[1])
        ims.append([im])

    ani = anim.ArtistAnimation(fig, ims, interval=10, blit=True)
    # fig.colorbar(img, ax=ax)
    plt.show()

    return ani


def compare_frame(vl1, vl2, e, emag, g: Grid, frame: int):
    plot_v(vl1[:, frame], g.m)
    plot_v(vl2[:, frame], g.m)
    plot_v(e[:, frame], g.m, (-emag, emag))


def compare_hor_line(vl1, vl2, g1: Grid, g2: Grid, frame: int, y):
    i1 = np.where(g1.yvec == y)[0][0]
    i2 = np.where(g2.yvec == y)[0][0]
    plt.plot(g1.xvec, vl1[:, frame].reshape(g1.shape)[:, i1])
    plt.plot(g2.xvec, vl2[:, frame].reshape(g2.shape)[:, i2])
