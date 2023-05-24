from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.widgets import Slider
import numpy as np
from grid import Grid


def plot_v(v, g: Grid, vlim: Union[float, int, tuple] = 0.4, ax: plt.Axes = None, title='', draw_block=True,
           decorate=True, cbar_label=None, **imshow_kwargs):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(layout='constrained')
    if isinstance(vlim, (float, int)):
        vlim = (-vlim, vlim)
    img = ax.imshow(v.reshape(g.shape, order='F'),
                    origin='lower',
                    extent=[-g.h/2, 1 + g.h/2, -g.h/2, 1 + g.h/2],
                    vmin=vlim[0], vmax=vlim[1], **imshow_kwargs)
    if decorate:
        ax.figure.colorbar(img, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
    if cbar_label is not None:
        cbar = ax.figure.colorbar(img, ax=ax)
        cbar.set_label(cbar_label)
    if draw_block:
        rect = matplotlib.patches.Rectangle((1/3, 1/3), 1/3, 1/3, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    if fig is None:
        return img
    else:
        return fig, ax, img


def plot_anim(vl: np.ndarray, g: Grid, vlim=(-0.4, 0.4), interval=10, **kwargs):
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

    ani = anim.ArtistAnimation(fig, ims, blit=True, interval=interval, **kwargs)
    # fig.colorbar(img, ax=ax)
    plt.show()

    return ani


def compare_frame(vl1, vl2, e, emag, g: Grid, frame: int):
    plot_v(vl1[:, frame], g)
    plot_v(vl2[:, frame], g)
    plot_v(e[:, frame], g, (-emag, emag))


def compare_line(tvec, grids, vls, labels):
    init_y = 0.5
    init_frame = len(tvec) // 2
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots()
    lines = [ax.plot(grids[i].xvec, grids[i].get_hor_line(vls[i][:, init_frame], init_y), label=labels[i])[0]
             for i in range(len(vls))]
    fig.legend()
    fig.subplots_adjust(left=0.25, bottom=0.25)

    ax_y = fig.add_axes([0.1, 0.25, 0.025, 0.63])
    y_slider = Slider(
        ax=ax_y,
        label='y',
        valmin=0,
        valmax=1,
        valinit=init_y,
        orientation='vertical'
    )

    ax_f = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    f_slider = Slider(
        ax=ax_f,
        label='Frame',
        valmin=0,
        valmax=len(tvec) - 1,
        valstep=1,
        valinit=init_frame
    )

    def update(val):
        for i in range(len(vls)):
            lines[i].set_ydata(grids[i].get_hor_line(vls[i][:, f_slider.val], y_slider.val))
        fig.canvas.draw_idle()

    f_slider.on_changed(update)
    y_slider.on_changed(update)

    plt.show()
    return fig, ax


# transformation functions, needs to handle division by zero
def m2h(m): return np.divide(1, m-1, out=np.full_like(m, np.Inf), where=m != 1, casting='unsafe')
def h2m(h): return np.divide(1, h, out=np.full_like(h, np.Inf), where=h != 0, casting='unsafe') + 1


def plot_errors(hs, ers, fmt='x'):
    fig: plt.Figure
    ax: plt.Axes
    fig, ax = plt.subplots(layout="constrained")
    if isinstance(hs, np.ndarray) and isinstance(hs.flatten()[0], Grid):
        hs = np.array([g.h for g in hs.flatten()]).reshape(hs.shape)

    lines = ax.loglog(hs, ers, fmt)
    ax2 = ax.secondary_xaxis('top', functions=(h2m, m2h))
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())

    ax.grid(visible=True, which='both', linestyle=':')
    ax.set_xlabel('$h$')
    ax2.set_xlabel('$m$')
    ax.set_ylabel('$||e||_h \\, / \\, ||v||$')

    return fig, ax, ax2, lines
