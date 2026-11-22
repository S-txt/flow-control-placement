#!/usr/bin/env python3
"""Draw the map of surveyed mechanisms on the plane "reaction time x view".

Source: survey/survey_table.csv (columns placement, tau_c_seconds,
control_failure_considered, tau_c_trigger).

Usage:  python3 survey/survey_map.py
Output: figures/fig1.pdf and figures/fig1.png
"""
import collections
import csv
import pathlib
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).resolve().parent.parent
FIGURES = BASE / 'figures'

# y axis: placement points in the order of their view
ORDER = ['switch silicon', 'end host', 'software edge', 'distributed routing', 'central controller']
LABEL = {'switch silicon': 'switch silicon\n(local load)',
         'end host': 'end host\n(own paths)',
         'software edge': 'software edge\n(paths from edge)',
         'distributed routing': 'distributed routing\n(topology after conv.)',
         'central controller': 'central controller\n(whole network)'}


def main():
    with (BASE / 'survey/survey_table.csv').open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    ypos = {p: i for i, p in enumerate(ORDER)}

    points = []
    for r in rows:
        p, t = r['placement'], r['tau_c_seconds']
        if p in ypos and t:
            try:
                points.append((float(t), ypos[p], r['control_failure_considered'] == 'yes',
                               r.get('tau_c_trigger', '') == 'failure', r['mechanism']))
            except ValueError:
                pass

    fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)
    random.seed(7)
    # marker: triangle = failure of the control means is considered;
    # filled = the published time is a reaction to a failure
    for x, y, control_failure, failure, _ in points:
        y_jitter = y + random.uniform(-0.17, 0.17)      # separate coinciding points
        ax.scatter(x, y_jitter, s=26 if not control_failure else 40,
                   marker='o' if not control_failure else '^',
                   facecolors='0.25' if failure else 'none',
                   edgecolors='0.15', linewidths=0.8, zorder=3)

    ax.set_xscale('log')
    ax.set_xlim(1e-8, 1e4)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([LABEL[p] for p in ORDER], fontsize=5.5)
    ax.set_xlabel('published reaction time, s (log scale)', fontsize=6)
    ax.tick_params(axis='x', labelsize=5.5)
    ax.grid(axis='x', ls=':', lw=0.4, color='0.75', zorder=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.scatter([], [], s=26, marker='o', facecolors='0.25', edgecolors='0.15',
               linewidths=0.8, label='reaction to a failure')
    ax.scatter([], [], s=26, marker='o', facecolors='none', edgecolors='0.15',
               linewidths=0.8, label='congestion or other')
    ax.scatter([], [], s=40, marker='^', facecolors='0.6', edgecolors='0.15',
               linewidths=0.8, label='control failure considered')
    # the legend sits above the axes so that it does not cover the points
    ax.legend(fontsize=4.6, loc='lower center', bbox_to_anchor=(0.45, 1.0), ncol=3,
              frameon=False, handletextpad=0.2, columnspacing=0.9)
    fig.tight_layout(pad=0.3)
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / 'fig1.pdf')
    fig.savefig(FIGURES / 'fig1.png')
    print(f'points on the map: {len(points)} of {len(rows)} rows of the table')
    print('by placement:', dict(collections.Counter(ORDER[y] for _, y, _, _, _ in points)))


if __name__ == '__main__':
    main()
