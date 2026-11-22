#!/usr/bin/env python3
"""Boundary of the choice for a non-exponential failure duration.

General condition. The global point wins when
    E[min(D, t_G)] - E[min(D, t_L)] < r * E[(D - t_L)^+],
and since E[(D - t)^+] = E[D] - E[min(D, t)], this is the same as
    g(t_G) - g(t_L) < r * (m - g(t_L)),   g(t) = E[min(D, t)].

For an exponential D this gives the closed form k* = ln(1/(1-r)), where
k = (t_G - t_L)/m. For the other distributions g(t) is integrated numerically
and k* is found by root bracketing. All distributions have the same mean m,
otherwise the comparison would make no sense.

Usage: python3 model/tail_boundary.py        # table and figures/fig2.pdf
"""
from __future__ import annotations

import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate, optimize, stats

M = 1.0            # mean failure duration, the same for every distribution
T_LOCAL = 0.0      # the local point reacts at once, so k = t_G / m

FIGURES = pathlib.Path(__file__).resolve().parent.parent / "figures"


def make_dist(kind: str, shape: float | None = None):
    """Frozen scipy distribution with mean M."""
    if kind == "exp":
        return stats.expon(scale=M)
    if kind == "det":
        return None                                   # handled separately
    if kind == "pareto":
        alpha = shape
        if alpha <= 1:
            raise ValueError("alpha must exceed 1")
        x_min = M * (alpha - 1.0) / alpha
        return stats.pareto(b=alpha, scale=x_min)
    if kind == "lognorm":
        sigma = shape
        mu_log = math.log(M) - 0.5 * sigma * sigma
        return stats.lognorm(s=sigma, scale=math.exp(mu_log))
    if kind == "weibull":
        c = shape
        scale = M / math.gamma(1.0 + 1.0 / c)
        return stats.weibull_min(c=c, scale=scale)
    raise ValueError(kind)


def g(t: float, kind: str, shape: float | None) -> float:
    """E[min(D, t)], the area under the survival curve from 0 to t."""
    if t <= 0:
        return 0.0
    if kind == "det":
        return min(M, t)
    dist = make_dist(kind, shape)
    value, _ = integrate.quad(lambda u: dist.sf(u), 0.0, t, limit=200)
    return float(value)


def k_star(r: float, kind: str, shape: float | None = None) -> float:
    """Boundary in t_G (with t_L = 0), in units of the mean duration."""
    if kind == "exp":
        return -math.log(1.0 - r)                     # closed form
    g_local = g(T_LOCAL, kind, shape)
    rhs = r * (M - g_local)

    def f(t):                                         # g(t_G) - g(t_L) - rhs
        return g(t, kind, shape) - g_local - rhs

    lo, hi = 1e-9, 1e-3
    while f(hi) < 0 and hi < 1e4:                     # find the upper bracket
        hi *= 2.0
    if f(hi) < 0:
        return float("inf")
    return float(optimize.brentq(f, lo, hi, xtol=1e-10)) / M


DISTS = [("exp", None, "exponential"),
         ("det", None, "deterministic"),
         ("weibull", 0.5, r"Weibull, shape 0.5"),
         ("lognorm", 1.0, r"log-normal $\sigma=1$"),
         ("lognorm", 2.0, r"log-normal $\sigma=2$"),
         ("pareto", 3.0, r"Pareto $\alpha=3$"),
         ("pareto", 2.0, r"Pareto $\alpha=2$"),
         ("pareto", 1.5, r"Pareto $\alpha=1.5$")]


def plain(label: str) -> str:
    """Label without TeX markup."""
    return label.replace("$", "").replace("\\", "")


def print_table():
    rs = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
    print(f"{'distribution':22s}" + "".join(f"  r={r:<5.2f}" for r in rs))
    for kind, shape, label in DISTS:
        values = [k_star(r, kind, shape) for r in rs]
        print(f"{label:22s}" + "".join(f"  {v:7.4f}" for v in values))

    print("\nthe global point wins if (t_G - t_L)/m < k*.")
    print("ratio to the exponential case at the same r:")
    for r in (0.1, 0.3, 0.5):
        e = k_star(r, "exp")
        line = f"  r={r:<4}: exp k*={e:.4f}"
        for kind, shape, label in DISTS[1:]:
            v = k_star(r, kind, shape)
            line += f" | {plain(label):18s} {v:.4f} ({v / e:4.2f}x)"
        print(line)


def plot_figure():
    """With t_L = 0 the global point wins while g(t_G) < r*m. The figure shows
    g(k*m)/m as a function of k: the boundary k* is where the curve reaches
    the level r."""
    ks = np.logspace(-2, 2, 160)
    fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=300)
    styles = ["-", "--", (0, (1, 1)), "-.", (0, (5, 1)), (0, (3, 1, 1, 1)),
              (0, (4, 2)), (0, (2, 2))]
    for (kind, shape, label), style in zip(DISTS, styles):
        ys = [g(float(k) * M, kind, shape) / M for k in ks]
        ax.plot(ks, ys, linestyle=style, lw=1.0, color="0.15", label=label)
    ax.axhline(0.5, color="0.45", lw=0.6, ls=":")
    ax.text(0.0115, 0.52, r"level $r=0.5$", fontsize=5, color="0.3")
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"delay of the global point $k=\Delta/m$", fontsize=6)
    ax.set_ylabel(r"accumulated area $\frac{1}{m}\int_0^{km} P(D>u)\,du$", fontsize=6)
    ax.tick_params(labelsize=5.5)
    ax.grid(ls=":", lw=0.4, color="0.8")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=4.4, loc="lower right", frameon=False, ncol=1,
              handlelength=2.2)
    fig.tight_layout(pad=0.3)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig2.pdf")
    print("\nfigure:", FIGURES / "fig2.pdf")


def main():
    print_table()
    plot_figure()


if __name__ == "__main__":
    main()
