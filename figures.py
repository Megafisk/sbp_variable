import matplotlib.pyplot as plt
import matplotlib.ticker as mpt
import matplotlib.patches as mplp
import numpy as np

import var_b
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
    lss = ['--', ':', ':']
    for i in range(3):
        ax.plot(hx, ref.error_conv(hx, cs[i], qs[i]), c=ls[i].get_color(), linestyle=lss[i])

    marker_legend = [f'$p={p}$' for p in [2, 4, 6]] + ['$p=2, m=3 m_b$']
    fit_legend = [f'${cs[i]} h{q}$' for i, q in enumerate(['^2', '', ''])]
    plt.legend(marker_legend + fit_legend, ncols=2, loc='lower right', columnspacing=-3.5)
    ax.set_title('$a=0.1$, $b=1/600$, $c=0.129$, $t=3$, yttre block')
    ax.grid(visible=True, which='both', linestyle=':')
    # f.set_size_inches(5, 4)
    f.set_size_inches(4, 3.5)

    plt.show()


def airglass():
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
    ax2.get_xaxis().set_minor_formatter(mpt.ScalarFormatter())
    # ax2.set_xticks([0.01, 30, 50, 100, 200], labels=[f'${i}$' for i in [1, 30, 50, 100, 200]])

    # cs = np.hstack([*[ref.fit_q(hs, es[:, i], q)[0] for i, q in enumerate([2, 2, 1])],
    #                 ref.fit_q(h4, e4, 1)[0]])
    line_params = [[198, 2], [151, 2], [17.0, 1], [1.71, 1]]
    marker_legend = [f'$p={p}$, {bt}' for p, bt in
                     [(2, 'inre'), (2, 'blandad, $s=1$'), (2, 'yttre'), (4, 'blandad, $s=2$')]]
    fit_legend = [f'${line_params[i][0]} h{q}$' for i, q in enumerate(['^2', '^2', '', ''])]
    hx = np.linspace(min(hs), max(hs))
    fit_lines = np.array([ref.error_conv(hx, *p) for p in line_params])[:, :, 0].T
    fit_ls = ax.plot(hx, fit_lines, linestyle=':', label=fit_legend)
    fit_ls[0].set_linestyle('--')
    fit_ls[1].set_linestyle('--')

    for i, line in enumerate(ls):
        fit_ls[i].set_color(line.get_color())

    ax.grid(visible=True, which='both', linestyle=':')
    ax.set_xlabel('$h$')
    ax2.set_xlabel('$m$')
    ax.set_ylabel('$||e||_h \\, / \\, ||v||$')
    ax.set_xlim(0.0049, 0.0355)
    ax.set_ylim(0.004, 1)

    f.legend(marker_legend + fit_legend, ncols=2, loc='outside lower right', fontsize=9)

    f.set_size_inches(3.5, 4)

    # run this after showing the plot
    for t in ax2.xaxis.get_ticklabels(which='minor'):
        if t.get_position()[0] not in (30, 50, 200):
            t.set_visible(False)
    plt.show()


def grid_variants():
    vl_ref, g_ref, params = ref.load_reference('ref450l')
    vnorm = g_ref.h * np.linalg.norm(vl_ref[:, -1])
    r = ref.load_conv('ref450l grid variants full')

    gs = np.array([Grid(m, False) for m in r['ms'].flatten()]).reshape(r['ms'].shape)

    ind = [1, 3, 0, 2]  # 3mb+1 outer, 3mb+1 inner, 3mb, 3mb+2
    hs = Grid.get_attr(gs, 'h')[:, ind]
    ers = r['ers'][:, ind]
    ax: plt.Axes
    f, ax, ax2, ls = plotting.plot_errors(hs, ers / vnorm, 'x')
    markers = 'ox^s'
    for l, m in zip(ls, markers):
        l.set_marker(m)
    f.set_size_inches(4, 3.5)

    hxs = np.array([[0.01, max(hs[:, 2])], [min(hs[:, 2]), 0.0085], hs[[0, -1], 1]])
    cs = [160, 1.23, 6.27, 3.58]
    ls += ax.plot(hxs[0], cs[0] * hxs[0] ** 2, 'C2--', label=f'${cs[0]}h^2$')
    ls += ax.plot(hxs[1], cs[1] * hxs[1], 'C2:', label=f'${cs[1]}h$')
    ls += ax.plot(hxs[2], cs[2] * hxs[2], 'C1:', label=f'${cs[2]}h$')
    ls += ax.plot(hxs[2], cs[3] * hxs[2], 'C3:', label=f'${cs[3]}h$')

    marker_legend = [f'$3 \\cdot m_b{a}' for a in [' + 1$ yttre', ' + 1$ inre', '$', ' + 2$']]
    leg_m = ax.legend(marker_legend)
    ax.add_artist(leg_m)
    leg_ls = ax.legend(handles=ls[-4:], loc='lower right')
    ax.set_title('$a=0.7$, $b=0.22$, c=$0.56$, $p=2$, $t=1.31$')

    plt.show()


# def ref450_q():
#     r = ref.load_conv('ref450l grid variants full')
#
#     ind = [1, 3, 0, 2]
#     f, ax = plt.subplots(nrows=2, ncols=1, sharex='all', layout='constrained')
#     ls = plotting.plot_q(r['hs'][1:, ind], r['q'], 'ox^s', '-', ax)
#     ax.set_ylim(0, 2.5)
#     f.set_size_inches(4, 3.5)


def grid_variants_faster():
    vl_ref, g_ref, params = ref.load_reference('faster180')
    vnorm = g_ref.h * np.linalg.norm(vl_ref[:, -1])
    r = ref.load_conv('faster180 outer')
    f, ax, ax2, ls = plotting.plot_errors(r['hs'], r['ers'] / vnorm, 'x-')
    # ax2.get_xaxis().set_minor_formatter(mpt.NullFormatter())
    markers = 'ox^+1s'
    # linestyles = ['-', '-', '-', '--', '--']
    for i, ll in enumerate(ls):
        ll.set_marker(markers[i])
        # ll.set_linestyle(linestyles[i])

    ind = [0, 1, 5, 3, 2, 4]
    legs = ['$p=2$, inre', '$p=2$, blandad, $s=1$', '$p=4$, blandad, $s=1$',
            '$p=2$, $m=3m_b+2$', '$p=4$, $m=3m_b+2$', '$p=2$, yttre']
    lg = f.legend(np.array(ls)[ind], np.array(legs)[ind], loc='outside lower right', ncols=2, fontsize=9)
    ax.set_title('$a=1$, $b=5$, $t=1.3$')
    f.set_size_inches(3.5, 4)
    plt.show()


def block_jump_illustration():
    ps = [2, 4, 6]
    t = 1.072
    g = Grid(12)
    vs = [var_b.reference_problem(g.mb, t, p, 1e3, 1e5, 3, 0.1)[0].v() for p in ps]
    vs_i = [var_b.reference_problem(g.mb, t, [2, 6][i], 1e3, 1e5, 3, 0.1,
                                    block_type='mixed', block_margin=[1, 3][i])[0].v()
            for i in [0, 1]]
    # for i in range(3):
    #     axes[0, i].set_title(f'$p={ps[i]}$, yttre')
    #     axes[1, i].set_title(f'$p={ps[i]}$, blandad, $s={i}$')

    vl_ref, g_ref, params = ref.load_reference('a1e3 b1e5')
    vs_i.append(vl_ref[:, -1])

    f: plt.Figure
    f, axes = plt.subplots(2, 3, layout='constrained', sharey='row')
    imgs = np.zeros(axes.shape, dtype='object')
    lower_titles = ['$p=2$, blandad, $s=1$', '$p=6$, blandad, $s=3$', 'referens']
    for i in range(axes.shape[1]):
        imgs[0, i] = plotting.plot_v(vs[i], g, 0.4, ax=axes[0, i], decorate=False)
        imgs[1, i] = plotting.plot_v(vs_i[i], g, 0.4, ax=axes[1, i], decorate=False)
        axes[1, i].set_xlabel('$x$')
        axes[0, i].set_title(f'$p={ps[i]}$, yttre', fontsize=11)
        axes[1, i].set_title(lower_titles[i], fontsize=11)

    axes[0, 0].set_ylabel('$y$')
    axes[1, 0].set_ylabel('$y$')
    # cbar = f.colorbar(imgs[1], ax=axes, location='bottom', label='$v$', shrink=0.6, fraction=0.1, aspect=25)
    cbar = f.colorbar(imgs[0, 0], ax=axes[:, 2], location='right', label='$v$', shrink=0.7, fraction=0.15, aspect=25)
    f.suptitle(f'$m_b={g.mb}$, $m={g.m}$, $a_c=10^3$, $b_c=10^5$, $t={t}$')
    f.set_size_inches(6, 4.2)


def block_jump_smaller():
    a = 1
    b = 100
    t = 1.072
    p = 6
    g = Grid(12)
    v = var_b.reference_problem(g.mb, t, p, a, b, 3, 0.1)[0].v()
    f: plt.Figure
    f, ax = plt.subplots(layout='constrained')
    img = plotting.plot_v(v, g, 0.4, ax=ax, decorate=False)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(f'$m_b=12$, $a={a}$, $b={b}$, $p={p}$, yttre')
    cbar = f.colorbar(img, ax=ax, label='$v$', shrink=0.7)
    f.set_size_inches(3.5, 3)


def grid_type(ax: plt.Axes, g, title, B=None, **kwargs):
    ax: plt.Axes
    if B is None:
        A, B = var_b.wave_block(g, 1, 1, 0, 0, **kwargs)
    mid_tick_i = np.nonzero(B == 1)[0][0]

    img = plotting.plot_v(B, g, (0, 1), ax=ax, decorate=False, cmap='Greys')

    ticks = np.linspace(-g.h / 2, 1 + g.h / 2, g.m + 1)
    mid_ticks = [ticks[mid_tick_i], ticks[-mid_tick_i - 1]]
    ax.set_xticks(ticks, minor=True)
    ax.set_yticks(ticks, minor=True)

    major_ticks = [0, 1] + mid_ticks
    labels = ['$0$', '$1$'] + [f'${round(n, 3)}$' for n in mid_ticks]
    ax.set_yticks(major_ticks, labels=labels)
    ax.set_xticks([0, 1], labels=[])

    ax.set_axisbelow(True)
    ax.grid(which='both')
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-g.h / 4, 1 + g.h / 4)
    ax.set_ylim(-g.h / 4, 1 + g.h / 4)

    inner = (B == 1).reshape((g.N, 1), order='F')
    # edge = (g.x == 0) + (g.y == 0) + (g.x == 1) + (g.y == 1)
    # outer = np.invert(inner + edge)
    outer = np.invert(inner)
    ax.plot(g.x[inner], g.y[inner], 'r^')
    # ax.plot(g.x[edge], g.y[edge], 'k.')
    ax.plot(g.x[outer], g.y[outer], 'C0.')

    rect = mplp.Rectangle((0, 0), 1, 1, edgecolor='k', facecolor='none')
    ax.add_patch(rect)


def grid_types(mb, **kwargs):
    ms = [(3 * mb + 1, '$m=3m_b+1$, yttre'), (3 * mb, '$m=3m_b$'), (3 * mb + 2, '$m=3m_b+2$'),
          (3 * mb + 1, '$m=3m_b+1$, inre'),
          (3 * mb + 1, '$b_X$, $s=1$'),
          (3 * mb + 1, '$b_Y$, $s=1$')]
    gs = [Grid(m, False) for m, _ in ms]
    titles = [t for _, t in ms]
    # mids = [mb, mb, mb + 1, mb]

    g = Grid(mb)
    block_margin = 1
    si = slice(g.mb + block_margin, 2 * g.mb + 1 - block_margin)
    so = slice(g.mb, 2 * g.mb + 1)
    Bx = np.zeros(g.shape)
    By = np.zeros(g.shape)
    Bi = np.zeros(g.shape)
    Bx[so, si] = 1
    By[si, so] = 1
    Bi[si, si] = 1
    Bs = [None, None, None, Bi, Bx, By]

    f: plt.Figure
    f, axes = plt.subplots(2, 3, layout='constrained', figsize=(6, 4))
    for i, ax in enumerate(axes.flatten()):
        grid_type(ax, gs[i], titles[i], B=Bs[i])
    for ax in axes[:, 0]:
        # ax.set_yticks([0, 1], labels=['$0$', '$1$'])
        ax.set_ylabel('$y$')
    for ax in axes[1, :]:
        ax.set_xlabel('$x$', labelpad=-5)
        ax.set_xticks([0, 1], ['$0$', '$1$'])
    return f, axes
    # f.suptitle(f'$m_b = {mb}$')


if __name__ == '__main__':
    plt.rcParams['text.usetex'] = True
    # grid_variants()

    airglass()
    # conv_inv()
    # grid_variants_faster()
    # ref450_q()
    plt.show()
