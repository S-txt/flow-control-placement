#!/usr/bin/env python3
"""Print the LaTeX table "closed form versus simulation".

The rows for unavailable decision points are simulated as well, so that the
random stream of the later rows stays the same, but they are printed to
stderr and are not part of the table.

Usage: python3 model/make_table.py --runs 1000000 --seed 42 > model/table_sim.tex
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sim import (analytic_avail, analytic_base, analytic_both, analytic_det,
                 boundary, ci95, simulate_both, simulate_single)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m", type=float, default=1.0)
    p.add_argument("--t-local", dest="t_local", type=float, default=0.01)
    p.add_argument("--t-global", dest="t_global", type=float, default=0.30)
    p.add_argument("--r", type=float, default=0.30)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    m, t_l, t_g, r = args.m, args.t_local, args.t_global, args.r
    rows = []   # (label, simulated sample, closed form, part of the main table?)

    loss_l, loss_g = simulate_single(rng, args.runs, m, t_l, t_g, r)
    ana_l, ana_g = analytic_base(m, t_l, t_g, r)
    rows.append((r'exponential $D$, $\LossL$', loss_l, ana_l, True))
    rows.append((r'exponential $D$, $\LossG$', loss_g, ana_g, True))

    a_l, a_g, w = 0.99, 0.90, m
    loss_l, loss_g = simulate_single(rng, args.runs, m, t_l, t_g, r,
                                     avail_local=a_l, avail_global=a_g,
                                     w_local=w, w_global=w)
    ana_l, ana_g = analytic_avail(m, t_l, t_g, r, a_l, a_g, w, w)
    rows.append((r'appendix A, $\LossL$', loss_l, ana_l, False))
    rows.append((r'appendix A, $\LossG$', loss_g, ana_g, False))

    loss_l, loss_g = simulate_single(rng, args.runs, m, t_l, t_g, r, dist='det')
    ana_l, ana_g = analytic_det(m, t_l, t_g, r)
    rows.append((r'deterministic $D$, $\LossL$', loss_l, ana_l, True))
    rows.append((r'deterministic $D$, $\LossG$', loss_g, ana_g, True))

    # both points together go last, so the rows above keep their random stream
    loss_b, _, _ = simulate_both(rng, args.runs, m, t_l, t_g, r)
    rows.append((r'together, $\LossLG$', loss_b, analytic_both(m, t_l, t_g, r), True))

    print(r'\begin{table}[t]')
    print(r'\caption{Closed-form expected loss versus Monte Carlo simulation of the same model. '
          r'%d runs per row, seed %d, $m=%g$, $\tL=%g$, $\tG=%g$, $r=%g$}'
          % (args.runs, args.seed, m, t_l, t_g, r))
    print(r'\label{tab:sim}')
    print(r'\centering')
    print(r'\footnotesize')
    print(r'\begin{tabular}{@{}lrrr@{}}')
    print(r'\toprule')
    print(r'Case & Simulated (95\,\% CI) & Closed form & Dev. \\')
    print(r'\midrule')
    for label, sample, ana, in_table in rows:
        mean, half = ci95(sample)
        dev = abs(mean - ana) / ana * 100 if ana else 0.0
        line = f'{label} & ${mean:.5f}\\pm{half:.5f}$ & ${ana:.5f}$ & {dev:.2f}\\,\\% \\\\'
        if in_table:
            print(line)
        else:
            print('[appendix] ' + line, file=sys.stderr)
    print(r'\bottomrule')
    print(r'\end{tabular}')
    print(r'\end{table}')
    print(file=sys.stderr)
    print(f'k = (t_G-t_L)/m = {(t_g - t_l) / m:.4f}, k* = {boundary(r):.4f}', file=sys.stderr)


if __name__ == '__main__':
    main()
