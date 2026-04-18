#!/usr/bin/env python3
import argparse
import json
import math
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ---------- Generic exact inclusion--exclusion utilities ----------

def all_nonempty_subsets(n):
    return [tuple(i for i in range(n) if (mask >> i) & 1) for mask in range(1, 1 << n)]

SUBSETS = {}

def get_subsets(n):
    if n not in SUBSETS:
        SUBSETS[n] = all_nonempty_subsets(n)
    return SUBSETS[n]


def indicator_ie(points, kind='hv'):
    pts = np.asarray(points, dtype=float)
    n, d = pts.shape
    total = 0.0
    for J in get_subsets(n):
        mins = np.min(pts[list(J)], axis=0)
        sign = 1.0 if (len(J) % 2 == 1) else -1.0
        if kind == 'hv':
            total += sign * float(np.prod(mins))
        elif kind == 'mag':
            total += sign * float(np.prod(1.0 + 0.5 * mins))
        else:
            raise ValueError(kind)
    return float(total)


def subgradient_ie(points, kind='hv', tol=1e-12):
    pts = np.asarray(points, dtype=float)
    n, d = pts.shape
    grad = np.zeros_like(pts)
    for J in get_subsets(n):
        mins = np.min(pts[list(J)], axis=0)
        sign = 1.0 if (len(J) % 2 == 1) else -1.0
        for a in range(d):
            minimizers = [j for j in J if abs(pts[j, a] - mins[a]) <= tol]
            if not minimizers:
                continue
            if kind == 'hv':
                coeff = float(np.prod([mins[b] for b in range(d) if b != a]))
            else:
                coeff = 0.5 * float(np.prod([1.0 + 0.5 * mins[b] for b in range(d) if b != a]))
            share = sign * coeff / len(minimizers)
            for j in minimizers:
                grad[j, a] += share
    return grad


def nondominated_unique_indices(points, tol=1e-12):
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    unique = []
    for i in range(n):
        if any(np.allclose(pts[i], pts[j], atol=tol, rtol=0) for j in unique):
            continue
        dominated = False
        for k in range(n):
            if k == i:
                continue
            if np.all(pts[k] >= pts[i] - tol) and np.any(pts[k] > pts[i] + tol):
                dominated = True
                break
        if not dominated:
            unique.append(i)
    return unique


def project_simplex(v):
    v = np.asarray(v, dtype=float)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)


def write_tikz_coords_2d(path, pts):
    with open(path, 'w', encoding='utf-8') as f:
        for x, y in pts:
            f.write(f'({x:.6f},{y:.6f}) ')
        f.write('\n')


def write_tikz_coords_3d(path, pts):
    with open(path, 'w', encoding='utf-8') as f:
        for x, y, z in pts:
            f.write(f'({x:.6f},{y:.6f},{z:.6f}) ')
        f.write('\n')


def save_convergence_csv(path, history):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('iter,value\n')
        for i, v in enumerate(history):
            f.write(f'{i},{v:.12f}\n')

# ---------- Introductory exact example ----------

def intro_three_point_example(outdir, make_png=True, make_tikz=True):
    pts = np.array([[1.0, 3.0], [3.0, 2.0], [5.0, 1.0]])
    hv = indicator_ie(pts, 'hv')
    mag = indicator_ie(pts, 'mag')
    if make_png:
        fig, ax = plt.subplots(figsize=(5.8, 4.2), dpi=180)
        cols = ['#d9d9d9', '#a9c7e8', '#f2b6b8']
        for (x, y), c in zip(pts, cols):
            ax.add_patch(plt.Rectangle((0, 0), x, y, facecolor=c, edgecolor='black', lw=1.0, alpha=0.45))
        ax.plot(pts[:, 0], pts[:, 1], 'ko', ms=4)
        ax.set_xlim(0, 5.5)
        ax.set_ylim(0, 3.5)
        ax.set_xlabel('$f_1$')
        ax.set_ylabel('$f_2$')
        ax.set_title('Introductory dominated-set example')
        ax.set_aspect('equal', adjustable='box')
        fig.tight_layout()
        fig.savefig(outdir / 'plot_intro_three_point_example.png')
        plt.close(fig)
    if make_tikz:
        write_tikz_coords_2d(outdir / 'tikz_intro_three_point_points.tex', pts)
    return {'points': pts.tolist(), 'hv': hv, 'mag': mag}

# ---------- Problem 1: exact projected normalized gradient on branch ----------

def hv1_value_grad_t(t):
    t = np.sort(np.array(t, dtype=float))
    mu = len(t)
    t0 = np.concatenate(([0.0], t))
    hv = sum((t[i] - t0[i]) * (4.0 - t[i]) for i in range(mu))
    g = np.zeros(mu)
    g[0] = t[1] - 2.0 * t[0]
    for i in range(1, mu - 1):
        g[i] = t[i - 1] + t[i + 1] - 2.0 * t[i]
    g[mu - 1] = 4.0 - 2.0 * t[mu - 1] + t[mu - 2]
    return hv, g


def mag1_value_grad_t(t):
    hv, gh = hv1_value_grad_t(t)
    t = np.sort(np.array(t, dtype=float))
    mag = 1.0 + (4.0 - t[0] + t[-1]) / 2.0 + hv / 4.0
    g = gh / 4.0
    g[0] -= 0.5
    g[-1] += 0.5
    return mag, g


def run_problem1(kind='hv', mu=8, max_iter=6000, eta0=0.02, decay=0.999):
    x = np.array([0.2076, 0.3903, 0.7841, 1.0471, 1.2343, 1.6137, 1.8890, 1.9537], dtype=float)
    history = []
    for k in range(max_iter):
        order = np.argsort(x)
        xs = x[order]
        t = xs ** 2
        val, gt = hv1_value_grad_t(t) if kind == 'hv' else mag1_value_grad_t(t)
        history.append(val)
        gx = 2.0 * xs * gt
        dirs = np.sign(gx)
        xs = np.clip(xs + eta0 * (decay ** k) * dirs, 0.0, 2.0)
        x[order] = xs
    x = np.sort(x)
    t = x ** 2
    val = hv1_value_grad_t(t)[0] if kind == 'hv' else mag1_value_grad_t(t)[0]
    return x, history, val

# ---------- Problem 2: full 2D projected normalized gradient with pull-back ----------

def F_problem2(z):
    x, y = z[..., 0], z[..., 1]
    return np.stack([1.0 - (x - 1.0) ** 2 - y ** 2, 1.0 - (x + 1.0) ** 2 - y ** 2], axis=-1)


def J_problem2(z):
    x, y = z
    return np.array([[-2.0 * (x - 1.0), -2.0 * y], [-2.0 * (x + 1.0), -2.0 * y]], dtype=float)


def run_problem2(kind='hv', mu=8, max_iter=250, seed=3):
    rng = np.random.default_rng(seed)
    xs = np.linspace(-1.2, 1.2, mu) + 0.08 * rng.standard_normal(mu)
    ys = 0.6 * np.sin(np.linspace(0, np.pi, mu)) + 0.08 * rng.standard_normal(mu)
    P = np.stack([xs, ys], axis=1)
    anchor = np.array([-4.0, -4.0])
    history = []
    for _ in range(max_iter):
        Y = F_problem2(P) - anchor
        active = nondominated_unique_indices(Y)
        Yact = Y[active]
        val = indicator_ie(Yact, kind)
        history.append(val)
        g_obj = subgradient_ie(Yact, kind)
        G = np.zeros_like(P)
        active_set = set(active)
        for local_idx, idx in enumerate(active):
            g_pull = J_problem2(P[idx]).T @ g_obj[local_idx]
            nrm = np.linalg.norm(g_pull)
            if nrm > 1e-14:
                G[idx] = g_pull / nrm
        surrogate = np.array([0.5, 0.5])
        for idx in range(mu):
            if idx not in active_set:
                g_pull = J_problem2(P[idx]).T @ surrogate
                nrm = np.linalg.norm(g_pull)
                if nrm > 1e-14:
                    G[idx] = g_pull / nrm
        accepted = False
        step = 0.12
        for _ in range(18):
            Pnew = np.clip(P + step * G, -2.0, 2.0)
            Ynew = F_problem2(Pnew) - anchor
            act_new = nondominated_unique_indices(Ynew)
            val_new = indicator_ie(Ynew[act_new], kind)
            if val_new >= val - 1e-12:
                P = Pnew
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    Y = F_problem2(P) - anchor
    active = nondominated_unique_indices(Y)
    return P, history, indicator_ie(Y[active], kind)

# ---------- 3D simplex exact projected normalized gradient ----------

def das_dennis_G3():
    H = 3
    pts = []
    for i in range(H + 1):
        for j in range(H + 1 - i):
            pts.append(np.array([i / H, j / H, (H - i - j) / H], dtype=float))
    return np.array(sorted(pts, key=lambda p: (p[0], p[1], p[2])))


def project_tangent(g):
    return g - np.mean(g) * np.ones_like(g)


def run_simplex(kind='hv', mu=9, max_iter=30):
    P = das_dennis_G3()
    if mu == 9:
        P = np.array([p for p in P if not np.allclose(p, [1/3, 1/3, 1/3])], dtype=float)
    elif mu != 10:
        raise ValueError('mu must be 9 or 10')
    history = []
    for _ in range(max_iter):
        val = indicator_ie(P, kind)
        history.append(val)
        g_obj = subgradient_ie(P, kind)
        G = np.zeros_like(P)
        for i in range(len(P)):
            gt = project_tangent(g_obj[i])
            nrm = np.linalg.norm(gt)
            if nrm > 1e-14:
                G[i] = gt / nrm
        if np.allclose(G, 0.0):
            break
        accepted = False
        step = 0.10
        for _ in range(24):
            Pnew = np.array([project_simplex(P[i] + step * G[i]) for i in range(len(P))])
            val_new = indicator_ie(Pnew, kind)
            if val_new >= val - 1e-12:
                P = Pnew
                accepted = True
                break
            step *= 0.5
        if not accepted:
            break
    return P, history, indicator_ie(P, kind)

# ---------- Plotting ----------

def plot_problem1(outdir, x_hv, x_mag, make_png=True, make_tikz=True):
    t_hv, t_mag = x_hv ** 2, x_mag ** 2
    if make_png:
        fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=180)
        ax.axhspan(-2, 2, color='0.88')
        ax.axhline(0, color='k', ls='--', lw=1)
        ax.scatter(x_hv, np.zeros_like(x_hv), c='tab:blue', s=28, label='HV-optimal 8 points')
        ax.scatter(x_mag, np.zeros_like(x_mag), c='tab:red', marker='s', s=28, label='Magnitude-optimal 8 points')
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False)
        fig.tight_layout(); fig.savefig(outdir / 'plot_problem1_decision.png'); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=180)
        tt = np.linspace(0, 4, 100)
        ax.plot(1 - tt, tt, 'k-', lw=1.5, label='Pareto front')
        ax.scatter(1 - t_hv, t_hv, c='tab:blue', s=28, label='HV')
        ax.scatter(1 - t_mag, t_mag, c='tab:red', marker='s', s=28, label='Magnitude')
        ax.scatter([-3], [0], c='k', marker='D', s=18)
        ax.text(-2.95, 0.12, 'anchor $r=(-3,0)$', fontsize=8)
        ax.set_xlabel('$F_1$'); ax.set_ylabel('$F_2$')
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(outdir / 'plot_problem1_front.png'); plt.close(fig)
    if make_tikz:
        write_tikz_coords_2d(outdir / 'tikz_problem1_decision_hv_coords.tex', [(v, 0.0) for v in x_hv])
        write_tikz_coords_2d(outdir / 'tikz_problem1_decision_mag_coords.tex', [(v, 0.0) for v in x_mag])
        write_tikz_coords_2d(outdir / 'tikz_problem1_front_hv_coords.tex', list(zip(1 - t_hv, t_hv)))
        write_tikz_coords_2d(outdir / 'tikz_problem1_front_mag_coords.tex', list(zip(1 - t_mag, t_mag)))


def plot_problem2(outdir, P_hv, P_mag, make_png=True, make_tikz=True):
    Y_hv, Y_mag = F_problem2(P_hv), F_problem2(P_mag)
    if make_png:
        fig, ax = plt.subplots(figsize=(6.5, 3.8), dpi=180)
        ax.axhspan(-2, 2, color='0.88')
        xx = np.linspace(-1, 1, 100)
        ax.plot(xx, np.zeros_like(xx), 'k-', lw=1.5, label='efficient set')
        ax.scatter(P_hv[:, 0], P_hv[:, 1], c='tab:blue', s=28, label='HV terminal population')
        ax.scatter(P_mag[:, 0], P_mag[:, 1], c='tab:red', marker='s', s=28, label='Magnitude terminal population')
        ax.set_xlim(-2.2, 2.2); ax.set_ylim(-2.2, 2.2)
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False)
        fig.tight_layout(); fig.savefig(outdir / 'plot_problem2_decision.png'); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=180)
        xs = np.linspace(-1, 1, 200)
        front = np.stack([1 - (xs - 1) ** 2, 1 - (xs + 1) ** 2], axis=1)
        ax.plot(front[:, 0], front[:, 1], 'k-', lw=1.5, label='Pareto front')
        ax.scatter(Y_hv[:, 0], Y_hv[:, 1], c='tab:blue', s=28, label='HV')
        ax.scatter(Y_mag[:, 0], Y_mag[:, 1], c='tab:red', marker='s', s=28, label='Magnitude')
        ax.scatter([-4], [-4], c='k', marker='D', s=18)
        ax.text(-3.95, -3.7, 'anchor $r=(-4,-4)$', fontsize=8)
        ax.set_xlabel('$F_1$'); ax.set_ylabel('$F_2$')
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(outdir / 'plot_problem2_front.png'); plt.close(fig)
    if make_tikz:
        write_tikz_coords_2d(outdir / 'tikz_problem2_decision_hv_coords.tex', P_hv)
        write_tikz_coords_2d(outdir / 'tikz_problem2_decision_mag_coords.tex', P_mag)
        write_tikz_coords_2d(outdir / 'tikz_problem2_front_hv_coords.tex', Y_hv)
        write_tikz_coords_2d(outdir / 'tikz_problem2_front_mag_coords.tex', Y_mag)


def plot_simplex(outdir, points, filename, color, title, make_png=True):
    if not make_png:
        return
    fig = plt.figure(figsize=(5.5, 4.8), dpi=180)
    ax = fig.add_subplot(111, projection='3d')
    verts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tri = Poly3DCollection([verts], alpha=0.20, facecolor='0.75', edgecolor='k')
    ax.add_collection3d(tri)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=24, depthshade=False)
    ax.set_xlabel('$f_1$'); ax.set_ylabel('$f_2$'); ax.set_zlabel('$f_3$')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.view_init(elev=24, azim=35)
    ax.set_title(title)
    fig.tight_layout(); fig.savefig(outdir / filename); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Reproduce the gradient-method examples from the magnitude report.')
    parser.add_argument('--only-2d', action='store_true', help='Run only the 2D examples (including the introductory exact example).')
    parser.add_argument('--only-3d', action='store_true', help='Run only the 3D simplex examples.')
    parser.add_argument('--no-png', action='store_true', help='Do not generate matplotlib PNG plots.')
    parser.add_argument('--no-tikz', action='store_true', help='Do not generate TikZ coordinate files.')
    parser.add_argument('--outdir', default='.', help='Output directory. Default: current directory.')
    args = parser.parse_args()

    if args.only_2d and args.only_3d:
        parser.error('Choose at most one of --only-2d and --only-3d.')

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    make_png = not args.no_png
    make_tikz = not args.no_tikz
    do_2d = not args.only_3d
    do_3d = not args.only_2d

    results = {}

    if do_2d:
        results['intro_three_point_example'] = intro_three_point_example(outdir, make_png=make_png, make_tikz=make_tikz)

        x_hv, hist_hv1, val_hv1 = run_problem1('hv')
        x_mag, hist_mag1, val_mag1 = run_problem1('mag')
        results['problem1'] = {'x_hv': x_hv.tolist(), 'x_mag': x_mag.tolist(), 'hv_value': val_hv1, 'mag_value': val_mag1}
        save_convergence_csv(outdir / 'convergence_problem1_hv.csv', hist_hv1)
        save_convergence_csv(outdir / 'convergence_problem1_mag.csv', hist_mag1)
        plot_problem1(outdir, x_hv, x_mag, make_png=make_png, make_tikz=make_tikz)

        P2_hv, hist_hv2, val_hv2 = run_problem2('hv')
        P2_mag, hist_mag2, val_mag2 = run_problem2('mag')
        results['problem2'] = {'P_hv': P2_hv.tolist(), 'P_mag': P2_mag.tolist(), 'hv_value': val_hv2, 'mag_value': val_mag2}
        save_convergence_csv(outdir / 'convergence_problem2_hv.csv', hist_hv2)
        save_convergence_csv(outdir / 'convergence_problem2_mag.csv', hist_mag2)
        plot_problem2(outdir, P2_hv, P2_mag, make_png=make_png, make_tikz=make_tikz)

    if do_3d:
        u = (62 + 5 * math.sqrt(13)) / 153
        v = (43 + math.sqrt(13)) / 153
        w = (48 - 6 * math.sqrt(13)) / 153
        P6_hv = np.array(sorted(set(itertools.permutations((u, v, w), 3))), dtype=float)
        P6_mag = np.array(sorted(set(itertools.permutations((7/9, 2/9, 0.0), 3))), dtype=float)
        results['simplex6'] = {'hv_points': P6_hv.tolist(), 'mag_points': P6_mag.tolist()}
        plot_simplex(outdir, P6_hv, 'plot_simplex_6_hv.png', 'tab:blue', '6-point HV symmetric orbit', make_png=make_png)
        plot_simplex(outdir, P6_mag, 'plot_simplex_6_mag.png', 'tab:red', '6-point Magnitude symmetric orbit', make_png=make_png)
        if make_tikz:
            write_tikz_coords_3d(outdir / 'tikz_simplex_6_hv_coords.tex', P6_hv)
            write_tikz_coords_3d(outdir / 'tikz_simplex_6_mag_coords.tex', P6_mag)

        P9_hv, hist_9hv, val_9hv = run_simplex('hv', 9)
        P9_mag, hist_9mag, val_9mag = run_simplex('mag', 9)
        P10_hv, hist_10hv, val_10hv = run_simplex('hv', 10)
        P10_mag, hist_10mag, val_10mag = run_simplex('mag', 10)
        results['simplex9'] = {'hv_points': P9_hv.tolist(), 'mag_points': P9_mag.tolist(), 'hv_value': val_9hv, 'mag_value': val_9mag}
        results['simplex10'] = {'hv_points': P10_hv.tolist(), 'mag_points': P10_mag.tolist(), 'hv_value': val_10hv, 'mag_value': val_10mag}
        save_convergence_csv(outdir / 'convergence_simplex_9_hv.csv', hist_9hv)
        save_convergence_csv(outdir / 'convergence_simplex_9_mag.csv', hist_9mag)
        save_convergence_csv(outdir / 'convergence_simplex_10_hv.csv', hist_10hv)
        save_convergence_csv(outdir / 'convergence_simplex_10_mag.csv', hist_10mag)
        plot_simplex(outdir, P9_hv, 'plot_simplex_9_hv.png', 'tab:blue', '9-point HV simplex run', make_png=make_png)
        plot_simplex(outdir, P9_mag, 'plot_simplex_9_mag.png', 'tab:red', '9-point Magnitude simplex run', make_png=make_png)
        plot_simplex(outdir, P10_hv, 'plot_simplex_10_hv.png', 'tab:blue', '10-point HV simplex run', make_png=make_png)
        plot_simplex(outdir, P10_mag, 'plot_simplex_10_mag.png', 'tab:red', '10-point Magnitude simplex run', make_png=make_png)
        if make_tikz:
            write_tikz_coords_3d(outdir / 'tikz_simplex_9_hv_coords.tex', P9_hv)
            write_tikz_coords_3d(outdir / 'tikz_simplex_9_mag_coords.tex', P9_mag)
            write_tikz_coords_3d(outdir / 'tikz_simplex_10_hv_coords.tex', P10_hv)
            write_tikz_coords_3d(outdir / 'tikz_simplex_10_mag_coords.tex', P10_mag)

    (outdir / 'example_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(f'Wrote outputs to: {outdir}')
    if 'simplex9' in results:
        print('9-point HV value =', results['simplex9']['hv_value'])
    if 'simplex10' in results:
        print('10-point HV value =', results['simplex10']['hv_value'])


if __name__ == '__main__':
    main()
