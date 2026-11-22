#!/usr/bin/env python3
"""Decision map: for which mean failure duration m and residual share r the
global point is preferable to the local one.

Axes: r on the x axis, m on the y axis (log scale). Each line is the boundary
Delta/m = ln(1/(1-r)), that is m = Delta / ln(1/(1-r)), for one value of the
extra delay Delta of the global point.

Delta is taken as a range and not as a difference of medians per placement
point: the published reaction times measure different things (some include
detection, some do not), so a median per placement point says little about
how fast that point really is. The range covers the spread of the absolute
reaction times in survey/survey_table.csv, which are printed for reference.

Usage: python3 model/decision_map.py        # figures/fig3.pdf
"""
from __future__ import annotations

import csv
import math
import pathlib
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = pathlib.Path(__file__).resolve().parent.parent
FIGURES = BASE / "figures"

DELTAS = [(1e-3, r"$\Delta=1$ ms"), (1e-2, r"$\Delta=10$ ms"), (1e-1, r"$\Delta=100$ ms")]


def reaction_times():
    """Absolute reaction times of the survey, grouped by placement point."""
    with (BASE / "survey/survey_table.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    by_placement = {}
    for row in rows:
        if row["tau_c_seconds"] and row["placement"]:
            by_placement.setdefault(row["placement"], []).append(float(row["tau_c_seconds"]))
    return by_placement


def main():
    by_placement = reaction_times()
    medians = {p: statistics.median(v) for p, v in by_placement.items() if len(v) >= 2}
    print("median published reaction time per placement point:")
    for p, v in sorted(medians.items(), key=lambda kv: kv[1]):
        print(f"  {p:22s} {v:.3e} s  (n={len(by_placement[p])})")

    values = sorted(v for vs in by_placement.values() for v in vs)
    print(f"\nabsolute reaction times: {len(values)}, from {values[0]:.2e} to {values[-1]:.2e} s")

    rs = np.linspace(0.02, 0.95, 200)
    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)
    styles = ["-", "--", "-."]
    for (delta, label), style in zip(DELTAS, styles):
        y = delta / (-np.log(1.0 - rs))         # boundary m = Delta / ln(1/(1-r))
        ax.plot(rs, y, style, lw=1.1, color="0.15", label=label)

    ax.set_yscale("log")
    ax.set_xlabel(r"residual share $r$ of the local point", fontsize=6)
    ax.set_ylabel(r"mean failure duration $m$, s", fontsize=6)
    ax.tick_params(labelsize=5.5)
    ax.grid(ls=":", lw=0.4, color="0.8")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    ax.text(0.62, 0.86, "global point wins", transform=ax.transAxes, fontsize=5.5, color="0.1")
    ax.text(0.45, 0.03, "local point wins", transform=ax.transAxes, fontsize=5.5, color="0.1")
    ax.legend(fontsize=4.4, loc="lower left", frameon=False, handlelength=2.0, labelspacing=0.7)
    fig.tight_layout(pad=0.3)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "fig3.pdf")
    print("\nfigure: figures/fig3.pdf")

    print("\nmean failure duration above which the global point wins:")
    for delta, _ in DELTAS:
        line = f"  Delta={delta * 1e3:6.1f} ms: "
        for r in (0.1, 0.3, 0.5):
            line += f" r={r}: {delta / (-math.log(1 - r)) * 1e3:8.1f} ms |"
        print(line)


if __name__ == "__main__":
    main()
