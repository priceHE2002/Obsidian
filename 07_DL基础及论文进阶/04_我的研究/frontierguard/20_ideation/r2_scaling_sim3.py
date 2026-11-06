#!/usr/bin/env python3
"""r2-scaling v3: fixes for E0 (kappa in draw), E3 (relative spread), E4 (beta), Weibull collapse."""
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


def draw_skills(K, d0, kappa, rng, mean_deficit=None):
    d = d0 * rng.random(K) ** (1.0 / kappa)
    if mean_deficit is not None:
        m = d.mean()
        if m > 1e-9:
            d = d * (mean_deficit / m)          # rescale while keeping the tail shape
    return np.clip(1.0 - d, 0.005, 0.9995)


def top_plans(p, H, N):
    d = 1.0 - p
    pool = np.argsort(d)[: min(len(p), 2 * H + 2)]
    if len(pool) < H:
        return []
    subs = [tuple(sorted(c)) for c in combinations(pool, H)]
    subs.sort(key=lambda s: sum(d[i] for i in s))
    return subs[:N]


def eps_bound(m, N, alpha, kind="bernstein", s=0.1):
    lt = math.log(2 * N / alpha)
    if kind == "hoeffding":
        return math.sqrt(lt / (2 * m))
    return math.sqrt(2 * s * (1 - s) * lt / m) + (2.0 / 3.0) * lt / m


def run_library(p, H, N, B, alpha=0.05, rho=0.3, R=0, rng=None, beta=0.0,
                kind="bernstein", m_per=None):
    plans = top_plans(p, H, N)
    if len(plans) < 2:
        return None
    a = np.full(H, math.sqrt(rho))
    S = np.array([plan_success(p[list(pi)], a, R=R) for pi in plans])
    n = len(plans)
    m = m_per if m_per is not None else max(4.0, B / (n * H))
    eps = eps_bound(m, n, alpha, kind)
    Shat = rng.binomial(int(round(m)), np.clip(S, 1e-9, 1 - 1e-9)) / m
    L, U = Shat - eps - beta, Shat + eps + beta
    o = np.argsort(-Shat)
    return dict(ok=bool(L[o[0]] > U[o[1]]), m=m, eps=eps, S=S,
                gap=float(S[o[0]] - S[o[1]]), Lmax=float(L.max()))


def phi_of_K(K, H, d0, kappa, B, alpha=0.05, rho=0.3, R=0, N=10, reps=200,
             seed=0, beta=0.0, kind="bernstein", mean_def=None):
    rng = np.random.default_rng(seed)
    ok, gaps = [], []
    for _ in range(reps):
        p = draw_skills(K, d0, kappa, rng, mean_def)
        r = run_library(p, H, N, B, alpha, rho, R, rng, beta, kind)
        if r:
            ok.append(r["ok"]); gaps.append(r["gap"])
    if not ok:
        return float("nan"), float("nan")
    return float(np.mean(ok)), float(np.mean(gaps))


def bisect_Kc(H, d0, kappa, B, N=10, reps=250, hi=5000.0):
    lo = max(3.0, H + 1.0)
    for _ in range(10):
        mid = math.sqrt(lo * hi)
        f, _ = phi_of_K(int(round(mid)), H, d0, kappa, B, N=N, reps=reps,
                        seed=int(mid * 13) % 99991)
        if np.isnan(f) or f <= 0.5:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


if __name__ == "__main__":
    # ---------- E0: margin law with kappa actually varying ----------
    print("### E0  log Delta ~ logK + logH + H*log q   (relative-spread skills)")
    for kappa in [1.0, 2.0]:
        X, y = [], []
        for H in [2, 3, 4, 5, 6]:
            for K in [10, 20, 50, 100, 200, 400]:
                for pbar in [0.5, 0.7, 0.85]:
                    rng = np.random.default_rng(K * 7 + H * 13 + int(pbar * 100))
                    p = draw_skills(K, 1 - pbar, kappa, rng)
                    r = run_library(p, H, 10, 1e12, rho=0.3, R=0, rng=rng)
                    if r is None or r["gap"] <= 1e-13:
                        continue
                    q = float(np.clip(np.sort(p)[:H].mean(), 0.02, 0.995))
                    X.append([1, math.log(K), math.log(H), H * math.log(q)])
                    y.append(math.log(r["gap"]))
        X, y = np.array(X), np.array(y)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        ss = 1 - ((y - X @ coef) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  kappa={kappa}: logK={coef[1]:+.3f} (pred {-1/kappa:+.2f})  "
              f"logH={coef[2]:+.3f} (pred {1/kappa-1:+.2f})  "
              f"Hlogq={coef[3]:+.3f} (pred +1)  R2={ss:.3f} n={len(y)}")

    # ---------- E1: Weibull collapse ----------
    print("\n### E1  phi(K) and Weibull collapse  phi=exp(-(K/Kc)^(1/kappa))")
    for kappa in [1.0, 2.0]:
        for B in [1e5, 1e6, 1e7]:
            Ks = [3, 6, 10, 20, 50, 100, 200, 400]
            ph = [phi_of_K(K, 3, 0.4, kappa, B, reps=300, seed=K * 17 + int(B % 991))[0]
                  for K in Ks]
            z = np.array([K ** (1 / kappa) for K in Ks])
            yy = -np.log(np.clip(ph, 1e-3, 0.999))
            sl = (z @ yy) / (z @ z)                       # through-origin slope
            pred = yy - sl * z
            print(f"  kappa={kappa} B={B:8.0f} phi=" + " ".join(f"{x:.2f}" for x in ph) +
                  f"  1/Kc={sl:.3f} resid_max={np.abs(pred).max():.2f}")

    # ---------- E2: Kc ~ B^(kappa/2) ----------
    print("\n### E2  K_c(B) slope [pred kappa/2]")
    for kappa in [0.5, 1.0, 2.0]:
        Bs = [1e5, 1e6, 1e7, 1e8]
        kcs = [bisect_Kc(3, 0.4, kappa, B) for B in Bs]
        print(f"  kappa={kappa}: K_c={[round(k,1) for k in kcs]} "
              f"slope={np.polyfit(np.log(Bs), np.log(kcs), 1)[0]:.3f}")

    # ---------- E3: retry exponent  Delta ~ (1-p)^R ----------
    print("\n### E3  Delta vs (1-p): slope should equal R  (K=20,H=3,kappa=1)")
    for R in [0, 1, 2]:
        ps, ds = [], []
        for pbar in [0.6, 0.7, 0.8, 0.9, 0.95, 0.98]:
            rng = np.random.default_rng(3)
            g = []
            for _ in range(120):
                p = draw_skills(20, 1 - pbar, 1.0, rng)
                r = run_library(p, 3, 10, 1e12, rho=0.3, R=R, rng=rng)
                if r and r["gap"] > 1e-14:
                    g.append(r["gap"])
            ps.append(pbar); ds.append(np.mean(g))
        sl = np.polyfit(np.log([1 - x for x in ps]), np.log(ds), 1)[0]
        print(f"  R={R}: Delta={[f'{x:.2e}' for x in ds]} slope={sl:+.2f} (pred {R:+d})")

    # ---------- E4: beta wall ----------
    print("\n### E4  beta wall (B=1e7,H=3,kappa=1,Kc_pred=q^2*d0/(2beta))")
    for beta in [0.0, 0.005, 0.01, 0.02]:
        row = [phi_of_K(K, 3, 0.4, 1.0, 1e7, reps=250, seed=K + 3, beta=beta)[0]
               for K in [4, 10, 20, 50, 100]]
        Kb = (0.36 * 0.4 / (2 * beta)) if beta > 0 else float("inf")
        print(f"  beta={beta:.3f} Kbeta_pred={Kb:7.1f} phi=" +
              " ".join(f"{x:.2f}" for x in row))

    # ---------- E5: certification budget table ----------
    print("\n### E5  B* (episodes), H=3, alpha=.05, N=10, q from p")
    for R in [0, 1, 2]:
        for pbar in [0.6, 0.8, 0.95]:
            q = 1 - (1 - pbar) ** (R + 1)
            row = []
            for K in [4, 100, 400]:
                delta = q ** 2 * (R + 1) * (1 - pbar) ** R * 0.05 / K
                m = 2 * math.log(20 / 0.05) / delta ** 2
                row.append(m * 30)
            print(f"  R={R} p={pbar}: q={q:.3f}  B*(K=4,100,400)=" +
                  " ".join(f"{x:.2e}" for x in row))

    # ---------- E6: dependence sign ----------
    print("\n### E6  product lower certificate coverage vs dependence sign")
    for name, a in [("indep", np.zeros(3)),
                    ("pos rho=.3", np.full(3, math.sqrt(0.3))),
                    ("neg (a=+,-,+)", np.array([math.sqrt(.3), -math.sqrt(.3), math.sqrt(.3)]))]:
        pp = np.array([0.7, 0.65, 0.75])
        Strue = plan_success(pp, a)
        rng = np.random.default_rng(5)
        m, cov, ubad = 3000, 0, 0
        for _ in range(500):
            qh = rng.binomial(m, pp) / m
            L = float(np.prod(qh)) - 3 * math.sqrt(math.log(20) / (2 * m))
            U = float(np.prod(qh)) + 3 * math.sqrt(math.log(20) / (2 * m))
            cov += (L <= Strue); ubad += (U < Strue)
        print(f"  {name:14s} S_true={Strue:.4f} prod=0.3413 lower_cov={cov/500:.3f} "
              f"upper_viol={ubad/500:.3f}")
