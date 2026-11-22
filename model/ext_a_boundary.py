#!/usr/bin/env python3
"""How unavailable decision points and overlapping failures shift the choice.

Part 1. A decision point is available at the failure instant with probability
a; a point that is down reacts only after it is restored, which takes an
exponential time with mean w. The expected losses are the closed forms that
sim.py checks in its `avail` scenario. The boundary k* is found numerically
in k = (t_G - t_L)/m with t_L = 0.

Part 2. Failures arrive as a Poisson process with rate lambda and may overlap.
The loss per unit of time is simulated for several values of lambda*m to see
where the order of the two points changes.

Usage: python3 model/ext_a_boundary.py
"""
import math
import pathlib
import sys

import numpy as np
from scipy import optimize

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from sim import simulate_overlap

M = 1.0     # mean failure duration


def loss(t, residual, avail, w):
    """Expected loss of a point with reaction time t, availability `avail`,
    mean restore time w and residual share `residual` after its reaction."""
    e = math.exp(-t / M)
    up = (1 - e) * M + residual * e * M
    if avail >= 1.0:
        return up
    down = (1 - e) * M + e * (M * w / (M + w) + residual * M * M / (M + w))
    return avail * up + (1 - avail) * down


def k_star(r, a_local=1.0, a_global=1.0, w=M):
    """k* such that the global point wins when (t_G - t_L)/m < k*."""
    loss_local = loss(0.0, r, a_local, w)
    f = lambda k: loss(k * M, 0.0, a_global, w) - loss_local
    if f(1e-9) > 0:          # the global point loses even with no extra delay
        return 0.0
    hi = 1e-3
    while f(hi) < 0 and hi < 1e4:
        hi *= 2
    return float(optimize.brentq(f, 1e-9, hi, xtol=1e-12)) if f(hi) > 0 else float('inf')


def main():
    print("boundary k* with an unavailable global point (w = m, a_L = 1):")
    a_values = (1.00, 0.99, 0.95, 0.90, 0.50)
    print(f"{'r':>6}" + "".join(f"  a_G={a:<5.2f}" for a in a_values))
    for r in (0.1, 0.3, 0.5, 0.7):
        row = f"{r:6.2f}"
        for a_global in a_values:
            row += f"  {k_star(r, 1.0, a_global):8.4f}"
        print(row)

    print("\nthe same with an unavailable local point (a_G = 1):")
    a_values = (1.00, 0.99, 0.95, 0.90)
    print(f"{'r':>6}" + "".join(f"  a_L={a:<5.2f}" for a in a_values))
    for r in (0.1, 0.3, 0.5):
        row = f"{r:6.2f}"
        for a_local in a_values:
            row += f"  {k_star(r, a_local, 1.0):8.4f}"
        print(row)

    print("\noverlapping failures: loss per unit of time against lambda*m")
    rng = np.random.default_rng(7)
    t_local, t_global, r = 0.01, 0.30, 0.30
    for lam_m in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        loss_l, loss_g = simulate_overlap(rng, 400000, M, lam_m / M, t_local, t_global, r)
        print(f"  lambda*m={lam_m:4.2f}:  Loss_L={loss_l:.6f}  Loss_G={loss_g:.6f}  "
              f"{'global' if loss_g < loss_l else 'local'} wins")


if __name__ == "__main__":
    main()
