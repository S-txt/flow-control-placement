#!/usr/bin/env python3
"""What the included mechanisms report about residual degradation.

A reproducible keyword search over the full texts of the included mechanisms.
The rule looks for key phrases (see PATTERNS) and does not judge the meaning:
  reports_tail_metric    the text has a tail metric (a 9x-th percentile,
                         tail latency, tail FCT) or a packet loss rate;
  reports_deadline_miss  the text has a share of flows that miss the deadline
                         (deadline miss ratio, missed deadlines and the like).
The reference list of each paper is cut off, so that titles of other papers
do not produce matches.

The texts where the deadline rule fired were then read by hand; the outcome
is kept in DEADLINE_MANUAL and goes to the column deadline_miss_share.

The full texts are not part of this repository. The script expects one plain
text file per mechanism, <id>.txt, where <id> is the id column of
survey/survey_table.csv. Its result is already stored in that table.

Usage:  python3 survey/keyword_rule.py --texts DIR          # print and write the columns
        python3 survey/keyword_rule.py --texts DIR --dry    # print only
"""
import argparse
import csv
import pathlib
import re
import sys

BASE = pathlib.Path(__file__).resolve().parent
csv.field_size_limit(10**9)

PATTERNS = {
    'reports_tail_metric': re.compile(
        r'9[059](\.9+)?\s*(th|-th)?[\s-]*(percentile|%-?ile|%\s*tile)|\bp9[59]\b|'
        r'tail\s+(latency|fct|flow\s+completion|delay|performance)|'
        r'packet\s+(loss|drop)\s+(rate|ratio|probability)|loss\s+rate', re.I),
    'reports_deadline_miss': re.compile(
        r'deadline[\s-]+miss|miss(ed|ing|es)?\s+(their\s+|the\s+|its\s+)?deadlines?|'
        r'deadline[\s-]+(meet|met)|meet(ing|s)?\s+(their\s+|the\s+|its\s+)?deadlines?|'
        r'fraction\s+of\s+flows\s+that\s+(miss|meet)', re.I),
}

# Manual reading of the 12 texts where reports_deadline_miss fired (2026-11-15).
# 'reported'               the share of flows, requests or jobs that miss the
#                          deadline is given as a result;
# 'reported under failure' the same, in an experiment with failures;
# 'mention only'           deadlines appear in the related work or in the
#                          problem statement, there is no result.
# In all three 'reported under failure' papers the share is given for the
# whole experiment and is not split into the intervals before and after the
# reaction.
DEADLINE_MANUAL = {
    'pfe182aa693': 'reported', 'pdc3156302f': 'reported', 'p228d8f7588': 'reported',
    'pf42e1cea06': 'reported',
    'p5da390d00d': 'reported under failure', 'pcd529ae069': 'reported under failure',
    'p5553978b26': 'reported under failure',
    'p730cc04abb': 'mention only', 'pe6ae8348d6': 'mention only', 'p808c019671': 'mention only',
    'pe4fc8d16f3': 'mention only', 'p32e58976b2': 'mention only',
}


def body(text: str) -> str:
    """Text of a paper without its reference list (cut at the last References heading)."""
    m = list(re.finditer(r'\n\s*(\d+\.?\s*)?(REFERENCES|References|R EFERENCES)\s*\n', text))
    return text[:m[-1].start()] if m else text


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--texts', type=pathlib.Path, default=BASE / 'fulltext',
                   help='directory with the full texts, one <id>.txt per mechanism')
    p.add_argument('--dry', action='store_true', help='print the summary, do not write the table')
    args = p.parse_args()

    path = BASE / 'survey_table.csv'
    with path.open(encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = list(reader)
    for col in list(PATTERNS) + ['deadline_miss_share']:
        if col not in fields:
            fields.append(col)

    missing = 0
    for r in rows:
        text_file = args.texts / f"{r['id']}.txt"
        if not text_file.exists():
            missing += 1
            for col in PATTERNS:
                r[col] = ''
            continue
        text = body(text_file.read_text(encoding='utf-8', errors='replace'))
        text = re.sub(r'-\s*\n\s*', '', text)          # words split at line ends
        text = re.sub(r'\s+', ' ', text)
        for col, pattern in PATTERNS.items():
            r[col] = 'yes' if pattern.search(text) else 'no'

    if missing == len(rows):
        sys.exit(f'no full texts found in {args.texts}, nothing to do')

    hit = {r['id'] for r in rows if r['reports_deadline_miss'] == 'yes'}
    if hit != set(DEADLINE_MANUAL):
        print('WARNING: the set of texts with a deadline match has changed, '
              'update the manual reading:', sorted(hit ^ set(DEADLINE_MANUAL)))
    for r in rows:
        r['deadline_miss_share'] = DEADLINE_MANUAL.get(r['id'], '')

    n = len(rows)
    print(f'included mechanisms: {n}; without a full text: {missing}')
    for col in PATTERNS:
        print(f'  {col}: {sum(r[col] == "yes" for r in rows)} of {n}')
    both = sum(r['reports_tail_metric'] == 'yes' and r['reports_deadline_miss'] == 'yes' for r in rows)
    print(f'  both: {both}')
    for v in ('reported', 'reported under failure', 'mention only'):
        print(f'  deadline_miss_share = {v}: {sum(r["deadline_miss_share"] == v for r in rows)}')

    if not args.dry:
        with path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        print(f'columns written to {path}')


if __name__ == '__main__':
    main()
