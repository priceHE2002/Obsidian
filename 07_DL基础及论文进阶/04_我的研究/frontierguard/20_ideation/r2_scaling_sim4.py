#!/usr/bin/env python3
"""r2-scaling v4: final numbers. E1 Weibull fit (K>=4), E6 sharp sign table, E7 1-D K slice."""
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


def draw_skills(K, d0, kappa, rng, rescale_to=None):
    d = d0 * rng.random(K) ** (1.0 / kappa)
    if rescale_to is not None and d.mean() > 1e-9:
        d = d * (rescale_to / d.mean())
    return np.clip(1.0 - d, 0.005, 0.9995)


def top_plans(p, H, N, pool_extra=2):
    d = 1.0 - p
    pool = np.argsort(d)[: min(len(p), H + pool_extra)]
    if len(pool) < H:
        return []
    subs = [tuple(sorted(c)) for c in combinations(pool, H)]
    subs.sort(key=lambda s: sum(d[i] for i in s))
    return subs[:N]


def eps_bound(m, N, alpha, s=0.1, kind="bernstein"):
    lt = math.log(2 * N / alpha)
    if kind == "hoeffding":
        return math.sqrt(lt / (2 * m))
    return math.sqrt(2 * s * (1 - s) * lt / m) + (2.0 / 3.0) * lt / m


def run_library(p, H, N, B, alpha=0.05, rho=0.3, R=0, rng=None, beta=0.0, m_per=None):
    plans = top_plans(p, H, N)
    if len(plans) < 2:
        return None
    a = np.full(H, math.sqrt(rho))
    S = np.array([plan_success(p[list(pi)], a, R=R) for pi in plans])
    n = len(plans)
    m = m_per if m_per is not None else max(4.0, B / (n * H))
    eps = eps_bound(m, n, alpha)
    Shat = rng.binomial(int(round(m)), np.clip(S, 1e-9, 1 - 1e-9)) / m
    L, U = Shat - eps - beta, Shat + eps + beta
    o = np.argsort(-Shat)
    return dict(ok=bool(L[o[0]] > U[o[1]]), m=m, eps=eps, gap=float(S[o[0]] - S[o[1]]),
                Smax=float(S.max()), q_top=float(np.sort(p)[:H].mean()))


def phi_of_K(K, H, d0, kappa, B, alpha=0.05, rho=0.3, R=0, N=10, reps=300,
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


def bisect_Kc(H, d0, kappa, B, N=10, reps=300, hi=8000.0):
    lo = max(3.0, H + 1.0)
    for _ in range(10):
        mid = math.sqrt(lo * hi)
        f = phi_of_K(int(round(mid)), H, d0, kappa, B, N=N, reps=reps,
                     seed=int(mid * 13) % 99991)[0]
        if np.isnan(f) or f <= 0.5:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


if __name__ == "__main__":
    print("### E1  phi(K) + Weibull fit  phi = exp(-(K/Kc)^(1/kappa))   [H=3,d0=.4,rho=.3,N=10]")
    Ks = [4, 6, 10, 20, 50, 100, 200, 400]
    for kappa in [1.0, 2.0]:
        for B in [1e5, 1e6, 1e7, 1e8]:
            ph, qs = [], []
            for K in Ks:
                f, _, q = phi_of_K(K, 3, 0.4, kappa, B, reps=400, seed=K * 17 + int(B % 991))
                ph.append(f); qs.append(q)
            pha = np.array(ph)
            z = np.array(Ks, float) ** (1 / kappa)
            yy = -np.log(np.clip(pha, 2e-3, 0.999))
            sl = (z @ yy) / (z @ z)
            r2 = 1 - ((yy - sl * z) ** 2).sum() / ((yy - yy.mean()) ** 2).sum()
            q = float(np.mean(qs))
            m = B / 30.0
            eps = eps_bound(m, 10, 0.05)
            kc_pred = (q ** 2 * 0.4 / (2 * eps)) ** 1  # kappa=1 form
            print(f"  kappa={kappa} B={B:8.0f} phi=" + " ".join(f"{x:.2f}" for x in pha) +
                  f" | Kc_fit={1/sl:7.1f} R2={r2:.4f} q_top={q:.3f} "
                  f"eps={eps:.4f} Kc_pred={kc_pred:7.1f}")

    print("\n### E2  K_c(B) slope [pred kappa/2]")
    for kappa in [0.5, 1.0, 2.0]:
        Bs = [1e5, 1e6, 1e7, 1e8]
        kcs = [bisect_Kc(3, 0.4, kappa, B) for B in Bs]
        print(f"  kappa={kappa}: K_c={[round(k,1) for k in kcs]} "
              f"slope={np.polyfit(np.log(Bs), np.log(kcs), 1)[0]:.3f}")

    print("\n### E3  Delta ~ (1-p)^(R+1)   [K=20,H=3,kappa=1, relative spread]")
    for R in [0, 1, 2]:
        ps, ds = [], []
        for pbar in [0.6, 0.7, 0.8, 0.9, 0.95, 0.98]:
            rng = np.random.default_rng(3)
            g = [run_library(draw_skills(20, 1 - pbar, 1.0, rng), 3, 10, 1e12, R=R,
                             rng=rng)["gap"] for _ in range(150)]
            g = [x for x in g if x > 1e-15]
            ps.append(pbar); ds.append(np.mean(g))
        sl = np.polyfit(np.log([1 - x for x in ps]), np.log(ds), 1)[0]
        print(f"  R={R}: slope={sl:+.2f} (pred {R+1})  "
              f"Delta(0.6)={ds[0]:.2e} Delta(0.95)={ds[-2]:.2e}")

    print("\n### E4  beta wall (B=1e7,H=3,kappa=1)")
    for beta in [0.0, 0.005, 0.01, 0.02]:
        row = [phi_of_K(K, 3, 0.4, 1.0, 1e7, reps=300, seed=K + 3, beta=beta)[0]
               for K in [4, 10, 20, 50, 100]]
        print(f"  beta={beta:.3f} phi=" + " ".join(f"{x:.2f}" for x in row))

    print("\n### E6  EXACT: prod q vs true joint S under negative/positive dependence")
    pp = np.array([0.7, 0.65, 0.75])
    print(f"  prod q = {np.prod(pp):.4f}")
    for r in [0.0, 0.2, 0.4, 0.6]:
        for sgn in [+1, -1]:
            a = np.array([1, -1, 1]) * math.sqrt(r) * sgn
            S = plan_success(pp, a)
            corr = np.corrcoef(a * a, a * a) if False else None
            lab = "pos" if sgn > 0 else "neg"
            print(f"  {lab} corr(i,j)=a_i a_j, |a|^2={r:.1f}: S_true={S:.4f} "
                  f"prod-S={np.prod(pp)-S:+.4f} -> product is "
                  f"{'ABOVE (unsafe lower bound)' if S < np.prod(pp) else 'below (safe)'}")

    print("\n### E7  Delta vs K at fixed H,q (1-D slice) [pred -1/kappa]")
    for kappa in [0.5, 1.0, 2.0]:
        Ks2 = [20, 50, 100, 200, 400]
        g = [phi_of_K(K, 3, 0.4, kappa, 1e12, reps=200, seed=K)[1] for K in Ks2]
        print(f"  kappa={kappa}: Delta={[f'{x:.2e}' for x in g]} "
              f"slope={np.polyfit(np.log(Ks2), np.log(g), 1)[0]:+.3f}")
