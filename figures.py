import matplotlib.pyplot as plt
import matplotlib.ticker as mpt
import numpy as np

from grid import Grid
import plotting
import ref


FIG_FOLDER = '/Users/hermanbergkvist/Unigrejs/kand/figures/'


def savefig(f: plt.Figure, fn: str):
    f.savefig(fn, format='pdf')


def conv_inv():
    vl_ref, g_ref, params = ref.load_reference('inv120')
    vnorm = g_ref.h * np.linalg.norm(vl_ref[:, -1])

    r = ref.load_conv('conv_inv_last')
    es = r['ers'] / vnorm
    hs = 1 / r['mbs'].T / 3
    f, ax, ax2, ls = plotting.plot_errors(hs, es)

    r2 = ref.load_conv('conv_inv_last 3mb')
    ls.append(ax.plot(r2['hs'], r2['ers'] / vnorm, 'o')[0])

    ls[0].set_marker('.')
    ls[2].set_marker('^')
    ax.set_ylabel('$||e||_h \\, / \\, ||v||$')
    ax.set_xlabel('$h$')
    ax2.set_xlabel('$m$')

    ax2x = ax2.get_xaxis()
    ax2x.set_minor_formatter(mpt.NullFormatter())

    qs = [2, 1, 1]
    cs = [9608, 71.6, 50.9]
    # cs = np.array([ref.fit_q(hs[4:], es[4:, i], q)[0] for i, q in enumerate(qs)])

    hx = np.linspace(hs[-1], hs[4])
    for i in range(3):
        ax.plot(hx, ref.error_conv(hx, cs[i], qs[i]), c=ls[i].get_color(), linestyle=':')

    marker_legend = [f'$p={p}$' for p in [2, 4, 6]] + ['$p=2, m=3 \\cdot m_b$']
    fit_legend = [f'${cs[i]} h{q}$' for i, q in enumerate(['^2', '', ''])]
    plt.legend(marker_legend + fit_legend, ncols=1, loc='lower right')
    ax.set_title('$a=0.1$, $b=1/600$, $c=0.129$, $t=3$, yttre block')
    ax.grid(visible=True, which='both', linestyle=':')
    f.set_size_inches(5, 4)

    plt.show()


def block_types():
    vl_ref, g_ref, params = ref.load_reference('airglass60')
    vnorm = g_ref.h * np.linalg.norm(vl_ref[:, -1])

    r = ref.load_conv('airglass block types')
    es = r['es'][:, [0, 2, 1]] / vnorm
    hs = 1 / r['mbs'].T / 3

    r4 = ref.load_conv('airglass mix o4')
    e4 = r4['ers'].T / vnorm
    h4 = 1 / r4['mbs'].T / 3

    f, ax, ax2, ls = plotting.plot_errors(hs, es)
    ls[0].set_marker('.')
    ls[2].set_marker('^')
    l4 = ax.plot(h4, e4, 's')[0]
    ls.append(l4)
    ax.set_title('Luft-glas, $t=1.31$')

    # cs = np.hstack([*[ref.fit_q(hs, es[:, i], q)[0] for i, q in enumerate([2, 2, 1])],
    #                 ref.fit_q(h4, e4, 1)[0]])
    line_params = [[198, 2], [151, 2], [17.0, 1], [1.71, 1]]
    marker_legend = [f'$p={p}$, {bt}' for p, bt in [(2, 'inre'), (2, 'blandad'), (2, 'yttre'), (4, 'blandad')]]
    fit_legend = [f'${line_params[i][0]} \cdot h{q}$' for i, q in enumerate(['^2', '^2', '', ''])]
    hx = np.linspace(min(hs), max(hs))
    fit_lines = np.array([ref.error_conv(hx, *p) for p in line_params])[:, :, 0].T
    fit_ls = ax.plot(hx, fit_lines, linestyle=':', label=fit_legend)

    for i, line in enumerate(ls):
        fit_ls[i].set_color(line.get_color())

    plt.legend(marker_legend + fit_legend, ncols=2, loc='lower right')

    ax.grid(visible=True, which='both', linestyle=':')
    ax.set_xlabel('$h$')
    ax2.set_xlabel('$m$')
    ax.set_ylabel('$||e||_h \\, / \\, ||v||$')
    ax.set_xlim(0.0049, 0.0355)
    ax.set_ylim(0.004, 1)

    f.set_size_inches(6, 5)
    savefig(f, 'luftglas blocktyper.pdf')
    plt.show()


def grid_variants():
    vl_ref, g_ref, params = ref.load_reference('ref450l')
    vnorm = g_ref.h * np.linalg.norm(vl_ref[:, -1])
    r = ref.load_conv('ref450l grid variants full')

    gs = np.array([Grid(m, False) for m in r['ms'].flatten()]).reshape(r['ms'].shape)

    ind = [1, 3, 0, 2]  # 3mb+1 outer, 3mb+1 inner, 3mb, 3mb+2
    hs = Grid.get_attr(gs, 'h')[:, ind]
    ers = r['ers'][:, ind]
    f, ax, ax2, ls = plotting.plot_errors(hs, ers / vnorm, '-x')
    ax2.get_xaxis().set_minor_formatter(mpt.NullFormatter())
    ls[0].set_marker('o')
    ls[2].set_marker('^')
    ls[3].set_marker('s')
    f.set_size_inches(4, 3.5)

    marker_legend = [f'$3 \\cdot m_b{a}' for a in [' + 1$ yttre', ' + 1$ inre', '$', ' + 2$']]
    plt.legend(marker_legend)
    ax.set_title('Gridvarianter, $a=0.7$, $b=0.22$, c=$0.56$, $p=2$, $t=1.31$')
    plt.show()


if __name__ == '__main__':
    plt.rcParams['text.usetex'] = True
    # grid_variants()

    # block_types()
    conv_inv()
