#!/usr/bin/env python3
"""r2-scaling final checks: closed-form Kc with contrast factor c; Frechet-lower counterexample; K slice."""
import math
from itertools import combinations
from statistics import NormalDist

import numpy as np

ND = NormalDist()
PHI = np.vectorize(ND.cdf)
PHI_INV = np.vectorize(ND.inv_cdf)
_x, _w = np.polynomial.hermite_e.hermegauss(160)
GHX, GHW = _x, _w / math.sqrt(2 * math.pi)


def plan_success(p, a=None, rho=0.0, R=0):
    q = 1.0 - (1.0 - np.asarray(p, float)) ** (R + 1)
    if a is None:
        a = np.full(len(q), math.sqrt(max(rho, 0.0)))
    a = np.asarray(a, float)
    if np.all(np.abs(a) < 1e-12):
        return float(np.prod(q))
    t = PHI_INV(1.0 - q)
    b = np.sqrt(np.clip(1 - a ** 2, 1e-12, None))
    cond = 1.0 - PHI((t[None, :] - a[None, :] * GHX[:, None]) / b[None, :])
    return float(GHW @ np.prod(cond, axis=1))


def draw_skills(K, d0, kappa, rng):
    return np.clip(1.0 - d0 * rng.random(K) ** (1.0 / kappa), 0.005, 0.9995)


def top_plans(p, H, N, pool_extra=2):
    d = 1.0 - p
    pool = np.argsort(d)[: min(len(p), H + pool_extra)]
    subs = [tuple(sorted(c)) for c in combinations(pool, H)]
    subs.sort(key=lambda s: sum(d[i] for i in s))
    return subs[:N]


def eps_bound(m, N, alpha, s=0.1):
    lt = math.log(2 * N / alpha)
    return math.sqrt(2 * s * (1 - s) * lt / m) + (2.0 / 3.0) * lt / m


def run_library(p, H, N, B, alpha=0.05, rho=0.3, R=0, rng=None, beta=0.0):
    plans = top_plans(p, H, N)
    if len(plans) < 2:
        return None
    a = np.full(H, math.sqrt(rho))
    S = np.array([plan_success(p[list(pi)], a, R=R) for pi in plans])
    n = len(plans)
    m = max(4.0, B / (n * H))
    eps = eps_bound(m, n, alpha)
    Shat = rng.binomial(int(round(m)), np.clip(S, 1e-9, 1 - 1e-9)) / m
    L, U = Shat - eps - beta, Shat + eps + beta
    o = np.argsort(-Shat)
    return dict(ok=bool(L[o[0]] > U[o[1]]), m=m, eps=eps, gap=float(S[o[0]] - S[o[1]]),
                q_top=float(np.sort(p)[:H].mean()), S=S)


def phi_of_K(K, H, d0, kappa, B, alpha=0.05, rho=0.3, R=0, N=10, reps=400,
             seed=0, beta=0.0):
    rng = np.random.default_rng(seed)
    ok, gaps, qs = [], [], []
    for _ in range(reps):
        p = draw_skills(K, d0, kappa, rng)
        r = run_library(p, H, N, B, alpha, rho, R, rng, beta)
        if r:
            ok.append(r["ok"]); gaps.append(r["gap"]); qs.append(r["q_top"])
    if not ok:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(ok)), float(np.mean(gaps)), float(np.mean(qs))


def kc_closed(H, d0, kappa, B, q, N=10, alpha=0.05, s=0.1, c=0.42):
    m = B / (N * H)
    two_eps = 2 * eps_bound(m, N, alpha, s)
    delta0 = q ** (H - 1) * (d0 / kappa) * H ** (1.0 / kappa - 1.0)
    return (delta0 / (c * two_eps)) ** kappa


if __name__ == "__main__":
    print("### F1  measured Kc (Weibull fit) vs closed form  Kc=[q^(H-1) d0 H^(1/k-1)/(k c 2eps)]^k, c=.42")
    Ks = [4, 6, 10, 20, 50, 100, 200, 400]
    for kappa in [1.0, 2.0]:
        for B in [1e6, 1e7, 1e8]:
            ph, qs = [], []
            for K in Ks:
                f, _, q = phi_of_K(K, 3, 0.4, kappa, B, reps=500,
                                   seed=K * 17 + int(B % 991) + int(kappa * 7))
                ph.append(f); qs.append(q)
            pha = np.array(ph)
            z = np.array(Ks, float) ** (1 / kappa)
            yy = -np.log(np.clip(pha, 2e-3, 0.999))
            sl = (z @ yy) / (z @ z)
            r2 = 1 - ((yy - sl * z) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
            kc_fit = 1 / sl
            kc_p = kc_closed(3, 0.4, kappa, B, float(np.mean(qs)))
            print(f"  kappa={kappa} B={B:.0e}: Kc_fit={kc_fit:7.1f} Kc_closed={kc_p:7.1f} "
                  f"ratio={kc_fit/kc_p:4.2f} R2={r2:.3f}  phi=" +
                  " ".join(f"{x:.2f}" for x in pha))

    print("\n### F2  identical marginal q, different joint law: product bound is NOT a lower bound")
    q = np.array([0.70, 0.65, 0.75])
    prod = float(np.prod(q))
    lo, hi = max(0.0, q.sum() - 2), float(q.min())
    print(f"  prod={prod:.4f}  Frechet interval=[{lo:.4f},{hi:.4f}]")
    for name, a in [("indep", np.zeros(3)), ("rho=+0.3 (1-factor)", np.full(3, math.sqrt(.3))),
                    ("rho=+0.6 (1-factor)", np.full(3, math.sqrt(.6)))]:
        print(f"  {name:22s} S={plan_success(q, a):.4f}")
    for S0 in [lo, 0.25, 0.30, prod, 0.40, hi]:
        s0 = S0
        si = q - s0
        okk = si.min() >= -1e-12 and s0 >= 0
        print(f"  explicit law P(exactly one success, margin s0={s0:.3f}) : "
              f"marginals exact={okk} S={s0:.4f} "
              f"{'<-- BELOW prod, admissible' if s0 < prod and okk else ''}")

    print("\n### F3  Delta vs K, K>=50 only")
    for kappa in [1.0, 2.0]:
        Ks2 = [50, 100, 200, 400]
        g = [phi_of_K(K, 3, 0.4, kappa, 1e12, reps=300, seed=K + 11)[1] for K in Ks2]
        print(f"  kappa={kappa}: Delta={[f'{x:.2e}' for x in g]} "
              f"slope={np.polyfit(np.log(Ks2), np.log(g), 1)[0]:+.3f} (pred {-1/kappa:+.2f})")
