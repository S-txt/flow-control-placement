#!/usr/bin/env python3
"""Print every number of the survey that is used in the paper.

Sources: survey/survey_table.csv (included mechanisms) and
survey/assessed_papers.csv (records of the assessment stage).

Usage: python3 survey/survey_stats.py
"""
import collections
import csv
import math
import pathlib
import statistics

BASE = pathlib.Path(__file__).resolve().parent
csv.field_size_limit(10**9)
Counter = collections.Counter

# placement points in the order of their view
ORDER = ['switch silicon', 'end host', 'software edge', 'distributed routing', 'central controller']


def read(name):
    with (BASE / name).open(encoding='utf-8') as f:
        return list(csv.DictReader(f))


def pct(a, b):
    return f'{a} of {b} ({100 * a / b:.1f} %)'


def main():
    table = read('survey_table.csv')
    review = read('assessed_papers.csv')
    n = len(table)

    print('== selection ==')
    print('full texts assessed:', len(review),
          '| included:', sum(r['verdict'] == 'include' for r in review),
          '| excluded:', sum(r['verdict'] == 'exclude' for r in review))
    print('unanimous verdicts:', sum(r['unanimous'] == 'yes' for r in review),
          '| records with fewer than 3 passes:', sum(r['votes'] != '3' for r in review),
          dict(Counter(r['votes'] for r in review)))

    print('\n== placement (included) ==')
    print(dict(Counter(r['placement'] or '(not assigned)' for r in table)))

    print('\n== failure duration and control failure ==')
    print('failure lasts a random time: assessed',
          pct(sum(r['failure_as_process'] == 'yes' for r in review), len(review)),
          '| included', pct(sum(r['failure_as_process'] == 'yes' for r in table), n))
    print('failure of the control means: assessed',
          pct(sum(r['control_failure'] == 'yes' for r in review), len(review)),
          '| included', pct(sum(r['control_failure_considered'] == 'yes' for r in table), n))

    print('\n== published reaction times ==')
    filled = [r for r in table if r['tau_c_kind'].strip()]
    kinds = Counter(r['tau_c_kind'] for r in filled)
    print('reaction time field filled:', len(filled), '| classes:', dict(kinds))
    print('a number of any kind:', pct(len(filled) - kinds['none'], n))
    loop = [r for r in table if r['tau_c_kind'] in ('loop_closure', 'detection_to_reroute')]
    absolute = [r for r in table if r['tau_c_seconds']]
    print('reaction time:', len(loop), '| of them absolute values:', len(absolute))
    print('detection among the absolute values:',
          dict(Counter(r['tau_c_includes_detection'] or '(empty)' for r in absolute)))
    print('reaction to:', dict(Counter(r['tau_c_trigger'] for r in absolute)))
    print('measured on:', dict(Counter(r['tau_c_measured_on'] or '(empty)' for r in absolute)))

    print('\n== map and range ==')
    placed = [r for r in absolute if r['placement']]
    values = [float(r['tau_c_seconds']) for r in absolute]
    print('absolute values:', len(absolute), '| with a placement point (points on the map):', len(placed))
    print(f'range: {min(values):.3g} .. {max(values):.3g} s = '
          f'{math.log10(max(values) / min(values)):.2f} orders of magnitude')
    print('triangles (control failure considered):',
          dict(Counter(r['placement'] for r in placed if r['control_failure_considered'] == 'yes')))
    failure = [r for r in placed if r['tau_c_trigger'] == 'failure']
    for name, selection in (('all values', placed), ('reaction to a failure only', failure)):
        print(f'-- {name} --')
        for p in ORDER:
            x = sorted(float(r['tau_c_seconds']) for r in selection if r['placement'] == p)
            if x:
                print(f'  {p:20s} n={len(x):2d}  min={min(x):.3g}  '
                      f'median={statistics.median(x):.3g}  max={max(x):.3g}')
            else:
                print(f'  {p:20s} n= 0')

    print('\n== table of surveyed mechanisms: fastest and slowest reaction to a failure '
          'per placement point, and every one that considers control failure ==')
    keep = {}
    for p in ORDER:
        x = sorted((r for r in failure if r['placement'] == p), key=lambda r: float(r['tau_c_seconds']))
        if x:
            keep[x[0]['id']] = x[0]
            keep[x[-1]['id']] = x[-1]
    for r in failure:
        if r['control_failure_considered'] == 'yes':
            keep[r['id']] = r
    for r in sorted(keep.values(), key=lambda r: (ORDER.index(r['placement']), float(r['tau_c_seconds']))):
        print(f"  {r['placement']:20s} {float(r['tau_c_seconds']):9.3g} s  "
              f"det={r['tau_c_includes_detection'] or '-':8s}"
              f" on={r['tau_c_measured_on'] or '-':10s} proc={r['failure_as_process']:3s}"
              f" cf={r['control_failure_considered']:3s}"
              f" {r['year']} {r['mechanism'][:44]}")

    print('\n== residual degradation ==')
    if 'reports_tail_metric' in table[0]:
        print('tail metric or packet loss rate (keyword rule):',
              pct(sum(r['reports_tail_metric'] == 'yes' for r in table), n))
        print('deadlines mentioned (keyword rule):', sum(r['reports_deadline_miss'] == 'yes' for r in table),
              '| manual reading:', dict(Counter(r['deadline_miss_share'] for r in table if r['deadline_miss_share'])))
    else:
        print('the keyword-rule columns are missing: run python3 survey/keyword_rule.py first')


if __name__ == '__main__':
    main()
