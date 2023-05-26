import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mpt
import matplotlib.patches as mplp

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

    f, axes = plt.subplots(2, 2, layout='constrained', figsize=(5, 4.4), sharey='row', sharex='col')
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
    for ax in axes[1, :]:
        ax.set_xlabel('$x$')
        # ax.set_xticks([0, 1], ['$0$', '$1$'])
    cbar = f.colorbar(imgs[0], ax=axes[:, -1], location='right', label='$v$', shrink=0.7, fraction=0.15, aspect=30)
    plt.show()


if __name__ == '__main__':
    plt.rcParams['text.usetex'] = True
    figure_complicated()

