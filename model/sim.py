#!/usr/bin/env python3
"""Monte Carlo check of the loss model for a local and a global decision point.

The simulator does not evaluate the closed-form expressions. It draws a failure
duration, builds the timeline of one failure (start, reaction of the decision
point, end of the failure) and integrates the share of late flows over that
timeline. The closed forms live in the analytic_* functions and are used for
comparison only.

Model:
  * a failure lasts D; by default D is exponential with mean m;
  * the local point reacts at t_L, after which a share r of the affected flows
    stays late until the failure ends;
  * the global point reacts at t_G > t_L and removes the degradation;
  * before the reaction all affected flows are late (share 1);
  * the loss is the time integral of the late share, so its unit is the time
    during which the traffic is lost completely.

Scenarios:
  base      one failure at a time, exponential D
  avail     a decision point is available with probability a; otherwise it
            reacts after an extra restore time, exponential with mean w
  tails     deterministic, Pareto and log-normal D with the same mean m
  overlap   failures arrive as a Poisson process with rate lam and may overlap
  hybrid    both points together: the local point reacts at t_L, then the
            global point removes the residual degradation at t_G
  grid      which point wins on a grid of (k, r), written to a CSV file

Usage:
    python3 model/sim.py --runs 1000000 --seed 42
    python3 model/sim.py --scenario hybrid --runs 1000000
    python3 model/sim.py --scenario grid --out model/grid.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys

import numpy as np


# --------------------------------------------------------------------------
# failure duration
# --------------------------------------------------------------------------


def draw_duration(rng: np.random.Generator, n: int, m: float,
                  dist: str = "exp", shape: float | None = None) -> np.ndarray:
    """Draw n failure durations with mean m from the given distribution."""
    if dist == "exp":
        return rng.exponential(m, n)
    if dist == "det":
        return np.full(n, m)
    if dist == "pareto":
        # mean = x_min * alpha / (alpha - 1), so x_min = m * (alpha - 1) / alpha
        alpha = shape if shape else 2.0
        if alpha <= 1.0:
            raise ValueError("Pareto: alpha must exceed 1, otherwise the mean does not exist")
        x_min = m * (alpha - 1.0) / alpha
        return x_min * (1.0 + rng.pareto(alpha, n))
    if dist == "lognorm":
        # shape is sigma; the location is chosen so that the mean equals m
        sigma = shape if shape else 1.0
        mu_log = math.log(m) - 0.5 * sigma * sigma
        return rng.lognormal(mu_log, sigma, n)
    raise ValueError(f"unknown distribution: {dist}")


# --------------------------------------------------------------------------
# simulation: the late share integrated over the realised timeline
# --------------------------------------------------------------------------


def loss_timeline(D: np.ndarray, t_react: np.ndarray, residual: float) -> np.ndarray:
    """Loss of one failure handled by a single decision point.

    The late share is 1 before t_react and `residual` after it (0 for the
    global point, r for the local one). The failure ends at D. The integral
    has two parts: [0, min(D, t_react)] and [min(D, t_react), D].
    """
    before = np.minimum(D, t_react)             # full loss
    after = np.maximum(D - t_react, 0.0)        # residual loss
    return before * 1.0 + after * residual


def loss_timeline_both(D: np.ndarray, t_local: float, t_global: float,
                       r: float) -> np.ndarray:
    """Loss of one failure when both points are present.

    Three parts: [0, min(D, t_local)] with share 1, then
    [min(D, t_local), min(D, t_global)] with share r, and no loss afterwards.
    """
    full = np.minimum(D, t_local)
    residual = np.maximum(np.minimum(D, t_global) - t_local, 0.0)
    return full * 1.0 + residual * r


def simulate_both(rng, runs, m, t_local, t_global, r, dist="exp", shape=None):
    """One failure per run. Returns the loss with both points and, on the same
    sample of D, the loss of the local point alone and of the global point alone."""
    D = draw_duration(rng, runs, m, dist, shape)
    loss_both = loss_timeline_both(D, t_local, t_global, r)
    loss_local = loss_timeline(D, np.full(runs, t_local), r)
    loss_global = loss_timeline(D, np.full(runs, t_global), 0.0)
    return loss_both, loss_local, loss_global


def simulate_single(rng, runs, m, t_local, t_global, r,
                    dist="exp", shape=None, avail_local=1.0, avail_global=1.0,
                    w_local=None, w_global=None):
    """One failure per run, no overlap. Returns the loss of the local point
    and the loss of the global point on the same sample of D."""
    D = draw_duration(rng, runs, m, dist, shape)

    # a point that is down reacts only after it is restored
    def reaction_time(t, avail, w):
        if avail >= 1.0:
            return np.full(runs, t)
        up = rng.random(runs) < avail
        restore = rng.exponential(w, runs)
        return np.where(up, t, t + restore)

    react_local = reaction_time(t_local, avail_local, w_local if w_local else m)
    react_global = reaction_time(t_global, avail_global, w_global if w_global else m)

    loss_local = loss_timeline(D, react_local, r)
    loss_global = loss_timeline(D, react_global, 0.0)
    return loss_local, loss_global


def simulate_overlap(rng, runs, m, lam, t_local, t_global, r, horizon=None):
    """Poisson arrivals of failures, overlaps allowed.

    The late share at time t is 1 if at least one active failure has not been
    reacted to yet, `residual` if all active failures have been reacted to,
    and 0 if no failure is active. Returns the mean loss per unit of time
    (not per failure) for the local and for the global point.
    """
    if horizon is None:
        horizon = runs / max(lam, 1e-9) / 50.0      # about runs/50 failures
    n = rng.poisson(lam * horizon)
    if n == 0:
        return 0.0, 0.0
    starts = np.sort(rng.random(n) * horizon)
    durations = rng.exponential(m, n)
    ends = starts + durations

    def integrate(t, residual):
        react = np.minimum(starts + t, ends)        # no reaction after the failure has ended
        full = union_measure(starts, react)         # time with an unhandled failure
        any_active = union_measure(starts, ends)    # time with any active failure
        return (full + residual * (any_active - full)) / horizon

    return integrate(t_local, r), integrate(t_global, 0.0)


def union_measure(a: np.ndarray, b: np.ndarray) -> float:
    """Total length of the union of the intervals [a_i, b_i)."""
    order = np.argsort(a)
    a, b = a[order], b[order]
    total = 0.0
    cur_a, cur_b = a[0], b[0]
    for i in range(1, len(a)):
        if a[i] > cur_b:
            total += cur_b - cur_a
            cur_a, cur_b = a[i], b[i]
        else:
            cur_b = max(cur_b, b[i])
    return total + (cur_b - cur_a)


# --------------------------------------------------------------------------
# closed forms (for comparison only, the simulator does not use them)
# --------------------------------------------------------------------------


def analytic_base(m, t_local, t_global, r):
    """Expected loss of the local and of the global point, exponential D."""
    loss_global = (1.0 - math.exp(-t_global / m)) * m
    loss_local = (1.0 - math.exp(-t_local / m)) * m + r * math.exp(-t_local / m) * m
    return loss_local, loss_global


def analytic_avail(m, t_local, t_global, r, a_local, a_global, w_local, w_global):
    """Expected loss when a point is available with probability a and is
    otherwise restored after an exponential time with mean w."""
    base_local, base_global = analytic_base(m, t_local, t_global, r)
    e_g, e_l = math.exp(-t_global / m), math.exp(-t_local / m)
    loss_global = a_global * base_global + (1 - a_global) * (
        (1 - e_g) * m + e_g * (m * w_global / (m + w_global)))
    loss_local = a_local * base_local + (1 - a_local) * (
        (1 - e_l) * m + e_l * (m * w_local / (m + w_local)
                               + r * m * m / (m + w_local)))
    return loss_local, loss_global


def analytic_both(m, t_local, t_global, r):
    """Expected loss with both points present, exponential D."""
    e_l, e_g = math.exp(-t_local / m), math.exp(-t_global / m)
    return (1.0 - e_l) * m + r * (e_l - e_g) * m


def analytic_det(m, t_local, t_global, r):
    """Loss of the local and of the global point for a deterministic D = m."""
    loss_global = min(m, t_global)
    loss_local = min(m, t_local) + r * max(m - t_local, 0.0)
    return loss_local, loss_global


def boundary(r: float) -> float:
    """Boundary k* = ln(1/(1-r)); the global point wins when k = (t_G - t_L)/m < k*."""
    return -math.log(1.0 - r)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def ci95(x: np.ndarray):
    """Sample mean and half-width of the 95 % confidence interval."""
    mean = float(np.mean(x))
    half = 1.959964 * float(np.std(x, ddof=1)) / math.sqrt(len(x))
    return mean, half


def agrees(sim_mean, sim_half, ana):
    """True if the deviation is below 1 % or the closed form is inside the CI."""
    return (abs(sim_mean - ana) / ana * 100 < 1.0) or (abs(sim_mean - ana) <= sim_half)


def report_row(name, sim_mean, sim_half, ana):
    dev = abs(sim_mean - ana) / ana * 100.0 if ana else 0.0
    inside = abs(sim_mean - ana) <= sim_half
    flag = "OK" if (dev < 1.0 or inside) else "MISMATCH"
    return (f"  {name:28s} sim {sim_mean:10.6f} +-{sim_half:8.6f} | "
            f"closed form {ana:10.6f} | dev {dev:5.2f}% | {flag}")


def winner(loss_local, loss_global):
    return "global" if loss_global < loss_local else "local"


def run_scenarios(args):
    rng = np.random.default_rng(args.seed)
    m, t_l, t_g, r = args.m, args.t_local, args.t_global, args.r
    ok = True
    print(f"parameters: m={m}  t_L={t_l}  t_G={t_g}  r={r}  "
          f"runs={args.runs}  seed={args.seed}\n")

    if args.scenario in ("all", "base"):
        print("SCENARIO base (exponential D, no overlap)")
        loss_l, loss_g = simulate_single(rng, args.runs, m, t_l, t_g, r)
        ana_l, ana_g = analytic_base(m, t_l, t_g, r)
        for name, sample, ana in (("local point Loss_L", loss_l, ana_l),
                                  ("global point Loss_G", loss_g, ana_g)):
            mean, half = ci95(sample)
            print(report_row(name, mean, half, ana))
            ok &= agrees(mean, half, ana)
        k = (t_g - t_l) / m
        print(f"  k = (t_G-t_L)/m = {k:.4f}; k* = ln(1/(1-r)) = {boundary(r):.4f}")
        print(f"  rule: the {'global' if k < boundary(r) else 'local'} point wins; "
              f"simulation: the {winner(np.mean(loss_l), np.mean(loss_g))} point wins\n")

    if args.scenario in ("all", "avail"):
        print("SCENARIO avail (decision points may be unavailable)")
        a_l, a_g, w = 0.99, 0.90, m
        loss_l, loss_g = simulate_single(rng, args.runs, m, t_l, t_g, r,
                                         avail_local=a_l, avail_global=a_g,
                                         w_local=w, w_global=w)
        ana_l, ana_g = analytic_avail(m, t_l, t_g, r, a_l, a_g, w, w)
        print(f"  a_L={a_l}  a_G={a_g}  w={w}")
        for name, sample, ana in (("local point Loss_L", loss_l, ana_l),
                                  ("global point Loss_G", loss_g, ana_g)):
            mean, half = ci95(sample)
            print(report_row(name, mean, half, ana))
            ok &= agrees(mean, half, ana)
        print()

    if args.scenario in ("all", "tails"):
        print("SCENARIO tails (other distributions of D, same mean)")
        for dist, shape, label in (("det", None, "deterministic"),
                                   ("pareto", 1.5, "Pareto alpha=1.5"),
                                   ("pareto", 2.0, "Pareto alpha=2"),
                                   ("pareto", 3.0, "Pareto alpha=3"),
                                   ("lognorm", 1.0, "log-normal sigma=1")):
            loss_l, loss_g = simulate_single(rng, args.runs, m, t_l, t_g, r,
                                             dist=dist, shape=shape)
            mean_l, half_l = ci95(loss_l)
            mean_g, half_g = ci95(loss_g)
            extra = ""
            if dist == "det":
                ana_l, ana_g = analytic_det(m, t_l, t_g, r)
                extra = f" | closed form Loss_L={ana_l:.4f} Loss_G={ana_g:.4f}"
                ok &= (abs(mean_l - ana_l) <= half_l
                       or abs(mean_l - ana_l) / ana_l * 100 < 1.0)
            print(f"  {label:22s} Loss_L={mean_l:8.5f}+-{half_l:.5f}  "
                  f"Loss_G={mean_g:8.5f}+-{half_g:.5f}"
                  f"  {winner(mean_l, mean_g)} wins{extra}")
        print()

    if args.scenario in ("all", "overlap"):
        print("SCENARIO overlap (failures may overlap)")
        print("  the loss is per unit of time, not per failure")
        for lam_m in (0.01, 0.1, 0.5, 1.0):
            lam = lam_m / m
            loss_l, loss_g = simulate_overlap(rng, args.runs, m, lam, t_l, t_g, r)
            print(f"  lambda*m={lam_m:5.2f}  Loss_L={loss_l:.6f}  Loss_G={loss_g:.6f}"
                  f"  {winner(loss_l, loss_g)} wins")
        print("  as lambda*m -> 0 the order is that of the single-failure model;\n"
              "  a larger lambda shifts the advantage, so lambda enters the choice\n"
              "  through the probability of overlap.\n")

    # This scenario runs last, so the random stream of the scenarios above
    # does not depend on whether it is present.
    if args.scenario in ("all", "hybrid"):
        print("SCENARIO hybrid (local point first, then the global point)")
        loss_b, loss_l, loss_g = simulate_both(rng, args.runs, m, t_l, t_g, r)
        ana_b = analytic_both(m, t_l, t_g, r)
        ana_l, ana_g = analytic_base(m, t_l, t_g, r)
        mean, half = ci95(loss_b)
        print(report_row("both points Loss_LG", mean, half, ana_b))
        ok &= agrees(mean, half, ana_b)
        for name, diff, ana in (("gain over the local point", loss_l - loss_b, ana_l - ana_b),
                                ("gain over the global point", loss_g - loss_b, ana_g - ana_b)):
            mean, half = ci95(diff)
            print(report_row(name, mean, half, ana))
            ok &= agrees(mean, half, ana)
        worse = int(np.sum((loss_b > loss_l + 1e-12) | (loss_b > loss_g + 1e-12)))
        print(f"  runs where both points lose more than one point alone: {worse} of {args.runs}")
        ok &= (worse == 0)
        for dist, shape, label in (("det", None, "deterministic"),
                                   ("pareto", 2.0, "Pareto alpha=2"),
                                   ("lognorm", 1.0, "log-normal sigma=1")):
            loss_b, loss_l, loss_g = simulate_both(rng, args.runs, m, t_l, t_g, r, dist, shape)
            worse = int(np.sum((loss_b > loss_l + 1e-12) | (loss_b > loss_g + 1e-12)))
            print(f"  {label:22s} Loss_LG={np.mean(loss_b):8.5f}  Loss_L={np.mean(loss_l):8.5f}  "
                  f"Loss_G={np.mean(loss_g):8.5f}  worse: {worse}")
            ok &= (worse == 0)
        print()

    return ok


def run_grid(args):
    """Empirical boundary on a grid of (k, r): where the global point wins."""
    rng = np.random.default_rng(args.seed)
    rows = []
    rs = np.linspace(0.05, 0.95, 19)
    ks = np.linspace(0.02, 3.0, 30)
    runs = max(20000, args.runs // 20)
    for r in rs:
        for k in ks:
            m, t_local = 1.0, 0.0
            t_global = k * m
            loss_l, loss_g = simulate_single(rng, runs, m, t_local, t_global, r)
            rows.append({"r": round(float(r), 4), "k": round(float(k), 4),
                         "loss_local": float(np.mean(loss_l)),
                         "loss_global": float(np.mean(loss_g)),
                         "global_wins_sim": int(np.mean(loss_g) < np.mean(loss_l)),
                         "global_wins_rule": int(k < boundary(r))})
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    agree = sum(row["global_wins_sim"] == row["global_wins_rule"] for row in rows)
    print(f"grid: {len(rows)} points -> {args.out}")
    print(f"simulation agrees with the rule k* = ln(1/(1-r)): {agree}/{len(rows)} "
          f"({100 * agree / len(rows):.1f}%)")
    return agree / len(rows) > 0.95


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", default="all",
                   choices=["all", "base", "avail", "tails", "overlap", "hybrid", "grid"])
    p.add_argument("--runs", type=int, default=200000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m", type=float, default=1.0, help="mean failure duration")
    p.add_argument("--t-local", dest="t_local", type=float, default=0.01,
                   help="reaction time of the local point")
    p.add_argument("--t-global", dest="t_global", type=float, default=0.30,
                   help="reaction time of the global point")
    p.add_argument("--r", type=float, default=0.30,
                   help="share of flows that stay late after the local reaction")
    p.add_argument("--out", default="model/grid.csv")
    args = p.parse_args()

    ok = run_grid(args) if args.scenario == "grid" else run_scenarios(args)
    print("RESULT:", "the simulation agrees with the closed forms" if ok
          else "MISMATCH, see the rows above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
