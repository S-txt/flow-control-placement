#!/usr/bin/env python3
"""Symbolic check of the formulas with sympy.

Nothing is simulated here. Every expression is obtained by integrating over
the density of the failure duration and is compared with the closed form.
The derivation is done with the rate mu = 1/m and then repeated in the
notation of the paper (m, t_L, t_G, r, k).

Usage: python3 model/verify_sympy.py
The exit code is 0 if all checks pass.
"""
import sys

import sympy as sp

mu, nu, tl, tg, b, t, x = sp.symbols('mu nu tau_l tau_g beta t x', positive=True)
f = mu * sp.exp(-mu * t)                      # density of D ~ Exp(mu)
checks = []


def E_min(tau):                               # E[min(D, tau)]
    return sp.integrate(t * f, (t, 0, tau)) + tau * sp.integrate(f, (t, tau, sp.oo))


def E_plus(tau):                              # E[(D - tau)^+]
    return sp.integrate((t - tau) * f, (t, tau, sp.oo))


def same(name, lhs, rhs):
    ok = sp.simplify(lhs - rhs) == 0
    checks.append(ok)
    print(f"{'OK ' if ok else 'FAIL'} {name}")


# ---- one decision point ----
same("E[min(D,tau)] = (1-exp(-mu tau))/mu", E_min(tg), (1 - sp.exp(-mu * tg)) / mu)
same("E[(D-tau)^+] = exp(-mu tau)/mu", E_plus(tl), sp.exp(-mu * tl) / mu)
L_g = E_min(tg)
L_l = E_min(tl) + b * E_plus(tl)
same("loss of the global point", L_g, (1 - sp.exp(-mu * tg)) / mu)
same("loss of the local point", L_l, (1 - sp.exp(-mu * tl)) / mu + b * sp.exp(-mu * tl) / mu)

# boundary: L_l - L_g = 0  <=>  mu (tau_g - tau_l) = ln(1/(1-beta))
D = sp.symbols('Delta', positive=True)
diff_lg = sp.simplify((L_l - L_g).subs(tg, tl + D) * mu * sp.exp(mu * tl))
same("local - global  ~  exp(-mu Delta) - (1 - beta)", diff_lg, sp.exp(-mu * D) - (1 - b))
root = sp.solve(sp.Eq(sp.exp(-mu * D), 1 - b), D)
same("boundary: Delta* = ln(1/(1-beta))/mu", root[0], sp.log(1 / (1 - b)) / mu)
same("sensitivity d k*/d beta = 1/(1-beta)", sp.diff(-sp.log(1 - b), b), 1 / (1 - b))

# ---- both points together ----
# L_b = E[min(D,tau_l)] + beta * E[(min(D,tau_g) - tau_l)^+],  tau_g > tau_l
mid = sp.integrate((t - tl) * f, (t, tl, tg)) + (tg - tl) * sp.integrate(f, (t, tg, sp.oo))
L_b = E_min(tl) + b * mid
same("loss with both points", L_b,
     (1 - sp.exp(-mu * tl)) / mu + b * (sp.exp(-mu * tl) - sp.exp(-mu * tg)) / mu)
same("gain over the local point alone = beta exp(-mu tau_g)/mu",
     L_l - L_b, b * sp.exp(-mu * tg) / mu)
gain_g = sp.simplify(L_g - L_b)
same("gain over the global point alone = (1-beta)(exp(-mu tau_l)-exp(-mu tau_g))/mu",
     gain_g, (1 - b) * (sp.exp(-mu * tl) - sp.exp(-mu * tg)) / mu)
# sign: for 0 < beta < 1 and tau_g = tau_l + Delta, Delta > 0, both gains are positive
g2 = sp.simplify(gain_g.subs(tg, tl + D) * mu * sp.exp(mu * tl) / (1 - sp.exp(-mu * D)))
same("gain over the global point alone has the sign of (1 - beta)", g2, 1 - b)

# ---- unavailable decision point ----
# D' is the rest of the failure at the instant tau (memoryless, Exp(mu));
# R ~ Exp(nu) is the restore time, independent of D'
fR = nu * sp.exp(-nu * x)
E_min_DR = sp.integrate(sp.exp(-mu * t) * sp.exp(-nu * t), (t, 0, sp.oo))     # int P(D'>t)P(R>t) dt
E_plus_DR = sp.integrate(sp.integrate((t - x) * f, (t, x, sp.oo)) * fR, (x, 0, sp.oo))
same("E[min(D',R)] = 1/(mu+nu)", E_min_DR, 1 / (mu + nu))
same("E[(D'-R)^+] = nu/(mu(mu+nu))", E_plus_DR, nu / (mu * (mu + nu)))
# loss of a point that is down, with residual share rho: full loss until tau,
# full loss until R, then rho
rho = sp.symbols('rho', nonnegative=True)
down = E_min(tl) + sp.exp(-mu * tl) * (E_min_DR + rho * E_plus_DR)
same("unavailable global point", down.subs({rho: 0, tl: tg}),
     (1 - sp.exp(-mu * tg)) / mu + sp.exp(-mu * tg) / (mu + nu))
same("unavailable local point", down.subs(rho, b),
     (1 - sp.exp(-mu * tl)) / mu + sp.exp(-mu * tl) * (1 / (mu + nu) + b * nu / (mu * (mu + nu))))

# ---- notation of the paper: m = 1/mu, t_L, t_G, r, k = Delta/m ----
m, tL, tG, rL, rG, k = sp.symbols('m t_L t_G r_L r_G k', positive=True)
sub = {mu: 1 / m}
Loss_G = E_min(tG).subs(sub)
Loss_L = (E_min(tL) + b * E_plus(tL)).subs(sub)
same("Loss_G = m(1 - e^(-t_G/m))", Loss_G, m * (1 - sp.exp(-tG / m)))
same("Loss_L = m(1 - e^(-t_L/m)) + r m e^(-t_L/m)", Loss_L,
     m * (1 - sp.exp(-tL / m)) + b * m * sp.exp(-tL / m))
# rule: Loss_G < Loss_L  <=>  e^(-k) > 1 - r  <=>  k < ln(1/(1-r))
lhs = sp.simplify((Loss_L - Loss_G).subs(tG, tL + k * m) * sp.exp(tL / m) / m)
same("Loss_L - Loss_G ~ e^(-k) - (1 - r)", lhs, sp.exp(-k) - (1 - b))
# general condition for any D: g(t_G) - g(t_L) < r (m - g(t_L)), plain algebra over g
gL, gG = sp.symbols('g_L g_G', positive=True)
same("general D: Loss_L - Loss_G = r(m - g(t_L)) - (g(t_G) - g(t_L))",
     (gL + b * (m - gL)) - gG, b * (m - gL) - (gG - gL))
# both points together in the notation of the paper
mid2 = (sp.integrate((t - tl) * f, (t, tl, tg)) + (tg - tl) * sp.integrate(f, (t, tg, sp.oo)))
Loss_LG = (E_min(tl) + b * mid2).subs({tl: tL, tg: tG}).subs(sub)
same("Loss_LG = m(1-e^(-t_L/m)) + r m (e^(-t_L/m) - e^(-t_G/m))", Loss_LG,
     m * (1 - sp.exp(-tL / m)) + b * m * (sp.exp(-tL / m) - sp.exp(-tG / m)))
same("Loss_L - Loss_LG = r m e^(-t_G/m)", sp.simplify(Loss_L - Loss_LG), b * m * sp.exp(-tG / m))
# generalisation: the global point also leaves a residual share, r_G < r_L
# Loss_G' = g(t_G) + r_G (m - g(t_G)),  Loss_L' = g(t_L) + r_L (m - g(t_L)),  D ~ Exp(1/m)
gx = lambda tau: m * (1 - sp.exp(-tau / m))
LossG2 = gx(tG) + rG * (m - gx(tG))
LossL2 = gx(tL) + rL * (m - gx(tL))
d2 = sp.simplify((LossL2 - LossG2).subs(tG, tL + k * m) * sp.exp(tL / m) / m)
same("r_G: Loss_L - Loss_G ~ (1-r_G) e^(-k) - (1-r_L)", d2, (1 - rG) * sp.exp(-k) - (1 - rL))
# root: k* = ln[(1-r_G)/(1-r_L)] makes the difference zero; the difference
# decreases in k, so the global point wins exactly when k < k*
# (this needs r_G < r_L, otherwise k* <= 0)
same("r_G: k* = ln[(1-r_G)/(1-r_L)] is the root", d2.subs(k, sp.log((1 - rG) / (1 - rL))), 0)
same("r_G: difference decreases in k", sp.diff(d2, k), -(1 - rG) * sp.exp(-k))
same("r_G = 0 gives k* = ln(1/(1-r_L))", sp.log((1 - rG) / (1 - rL)).subs(rG, 0), sp.log(1 / (1 - rL)))
# Pareto and deterministic D: with t_L = 0 the boundary is where the area
# under the survival curve up to t_G equals r m
same("t_L = 0: condition is g(t_G) < r m", (gL + b * (m - gL) - gG).subs(gL, 0), b * m - gG)
# unavailable points in the notation m = 1/mu, w = 1/nu
w = sp.symbols('w', positive=True)
same("1/(mu+nu) = m w/(m+w)", (1 / (mu + nu)).subs({mu: 1 / m, nu: 1 / w}), m * w / (m + w))
same("nu/(mu(mu+nu)) = m^2/(m+w)", (nu / (mu * (mu + nu))).subs({mu: 1 / m, nu: 1 / w}), m**2 / (m + w))

print(f"\n{sum(checks)}/{len(checks)} checks passed")
sys.exit(0 if all(checks) else 1)
