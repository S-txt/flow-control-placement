#!/usr/bin/env python3
"""The decision rule applied to published statistics of failure duration.

No measurements are made here. Only numbers printed in two papers are used,
with page references:

  Gill, Jain, Nagappan. Understanding Network Failures in Data Centers.
  SIGCOMM 2011, pp. 350-361.
  Meza, Xu, Veeraraghavan, Mutlu. A Large Scale Study of Data Center Network
  Reliability. IMC 2018, pp. 393-407.

Threshold: the mean failure duration above which the global point is
preferable, d* = Delta / ln(1/(1-r)), for the Delta and r of the decision map.

Usage: python3 model/illustration.py
"""
import math

DELTAS = [1e-3, 1e-2, 1e-1]            # s, as on the decision map
RS = [0.1, 0.3, 0.5]

# (source, page, what is published, value in seconds, kind of statistic)
PUBLISHED = [
    ("Gill 2011", "p. 356, Fig. 8(a), 9(a)", "lower end of the time-to-repair CDFs (devices, links)", 1e2, "lower bound of recorded durations"),
    ("Gill 2011", "p. 356, Sec. 4.5.1", "median time to repair, LB-1 load balancers (shortest device failures)", 3.7 * 60, "median"),
    ("Gill 2011", "p. 356, Sec. 4.5.1", "median time to repair, LB-3 load balancers", 4.5 * 60, "median"),
    ("Gill 2011", "p. 356, Sec. 4.5.1", "majority of link failures resolved within", 5 * 60, "upper bound for >50%"),
    ("Meza 2018", "p. 396, Table 1", "average automated repair time, rack switches (RSW)", 2.91, "mean, repair step only"),
    ("Meza 2018", "p. 396, Table 1", "average automated repair time, fabric switches (FSW)", 4.45, "mean, repair step only"),
    ("Meza 2018", "p. 396, Table 1", "average automated repair time, core devices", 30.1, "mean, repair step only"),
    ("Meza 2018", "p. 396, Table 1", "average wait before automated repair, core devices", 4 * 60, "mean"),
    ("Meza 2018", "p. 403, Sec. 6.1", "50% of backbone edge nodes recover within", 10 * 3600, "median"),
    ("Meza 2018", "p. 404, Sec. 6.2", "50% of fiber vendors repair links within", 13 * 3600, "median"),
]


def threshold(delta, r):
    return delta / math.log(1.0 / (1.0 - r))


def main():
    print("thresholds d* = Delta/ln(1/(1-r)), ms:")
    print("  Delta \\ r    " + "".join(f"{r:>10}" for r in RS))
    for delta in DELTAS:
        print(f"  {delta * 1e3:6.0f} ms    " + "".join(f"{threshold(delta, r) * 1e3:10.1f}" for r in RS))
    thresholds = [threshold(delta, r) for delta in DELTAS for r in RS]
    lo, hi = min(thresholds), max(thresholds)
    print(f"  range of the thresholds: {lo * 1e3:.1f} ms .. {hi * 1e3:.0f} ms")
    print(f"  approximation Delta/r: {min(DELTAS) / max(RS) * 1e3:.0f} ms .. {max(DELTAS) / min(RS) * 1e3:.0f} ms")

    print("\npublished statistics and their ratio to the largest threshold:")
    for source, page, what, value, kind in PUBLISHED:
        print(f"  {source:9s} {page:26s} {value:9.4g} s  x{value / hi:9.1f}  [{kind}] {what}")
    print(f"\nall {len(PUBLISHED)} published values are above the largest threshold: "
          f"{all(value > hi for _, _, _, value, _ in PUBLISHED)}")
    d_min = PUBLISHED[0][3]
    print(f"Gill 2011: the lower end of the recorded durations is {d_min:.0f} s, so the share of recorded "
          f"failures above the largest threshold of {hi:.2f} s is 100 %.")
    print("Meza 2018: the distribution of durations is not published (only means and per-node quantiles),"
          " so the share cannot be computed.")
    print("\nextra delay Delta at which the rule changes sign for the shortest recorded failures"
          f" (d = {d_min:.0f} s):")
    for r in RS:
        print(f"  r={r}: Delta* = d*ln(1/(1-r)) = {d_min * math.log(1 / (1 - r)):.1f} s")


if __name__ == "__main__":
    main()
