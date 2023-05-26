import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt

import figures
import ref
from grid import Grid
import plotting


# def plot_t(r, g: Grid, t: float, **kwargs):
#     return plotting.plot_v(r['vl'][:, ti], g, **kwargs)


def figure_complicated():
    r = spio.loadmat(ref.CALC_FOLDER + 'waveguide.mat')
    g = Grid(int(r['m'][0, 0]), False)
    ts = [0.05, 0.4, 0.8, 2.5]

    fs_t = r['t_vec'][:, r['saved_frames']]

    shp = (2, 2)
    f, axes = plt.subplots(*shp, layout='constrained', figsize=(5, 4.4), sharey='row', sharex='col', dpi=1200)
    axes = axes.reshape(shp)
    imgs = np.zeros(4, dtype='object')
    for i, ax in enumerate(axes.flatten()):
        t = ts[i]
        ti = np.argmax(fs_t >= t)

        ax: plt.Axes
        imgs[i] = plotting.plot_v(r['vl'][:, ti], g, ax=ax, decorate=False, draw_block=False)
        ax.set_title(f'$t={t}$')
    for ax in axes[:, 0]:
        # ax.set_yticks([0, 1], labels=['$0$', '$1$'])
        ax.set_ylabel('$y$')
    for ax in axes[-1, :]:
        ax.set_xlabel('$x$')
        # ax.set_xticks([0, 1], ['$0$', '$1$'])
    cbar = f.colorbar(imgs[0], ax=axes[:, -1], location='right', label='$v$', shrink=0.7, fraction=0.15, aspect=30)
    f.savefig(figures.FIG_FOLDER + 'komp.pdf')


def figure_comp_b():
    r = spio.loadmat(ref.CALC_FOLDER + 'waveguide.mat')
    g = Grid(int(r['m'][0, 0]), False)

    B = plt.imread('../komplicerat/' + r['img'][0]) / 255
    B = np.rot90(B, k=-1).astype(float)
    f: plt.Figure
    ax: plt.Axes
    f, ax = plt.subplots(layout='constrained', figsize=(3, 3), dpi=500)
    img = plotting.plot_v(B.reshape((g.N,), order='C'), g, (0, np.max(B)), ax=ax,
                          cmap='gray', decorate=False, draw_block=False)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    f.savefig(figures.FIG_FOLDER + 'komp_b.pdf')


if __name__ == '__main__':
    plt.rcParams['text.usetex'] = True
    figure_complicated()
    # figure_comp_b()
