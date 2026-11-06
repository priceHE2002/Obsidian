#!/usr/bin/env python3
"""
r2-scaling: CPU verification of the "library certification scaling law".

Model
-----
K skills, step success q_g, deficits d_g = 1-q_g ~ G(d)=(d/d0)^kappa on [0,d0].
Plan pi = H distinct skills; retries R: q_i=1-(1-p_i)^(R+1); S(pi)=E prod q_i under a
1-factor Gaussian copula (exact by 1-D Gauss-Hermite quadrature).
Certification of the argmax plan over N queried plans with m joint rollouts each:
    L_i = Shat_i - eps(m) - beta,  U_i = Shat_i + eps(m) + beta
    eps_Hoeffding = sqrt(log(2N/a)/(2m))
    eps_Bernstein = sqrt(2 s(1-s) log(2N/a)/m) + (2/3) log(2N/a)/m
    success iff L_(1) > U_(2).
Budget B = N*H*m skill-executions (a length-H joint rollout costs H).

Laws under test
---------------
L1  Delta(K,H) = (d0/kappa) H^{1/kappa-1} K^{-1/kappa}          (slope -1/kappa in K)
L2  m* = 2 log(2N/a) / (Delta-2beta)^2  =>  K_c ~ B^{kappa/2} (N fixed)
L3  beta wall  K_beta = [d0 H^{1/kappa-1}/(2 kappa beta)]^kappa, independent of B
L4  phi(p) interior optimum at p* ~ (H-1)/(H-1+R)
"""
import math
from itertools import combinations
from statistics import NormalDist

import numpy as np

ND = NormalDist()
PHI = np.vectorize(ND.cdf)
PHI_INV = np.vectorize(ND.inv_cdf)


def gh_nodes(n=120):
    x, w = np.polynomial.hermite_e.hermegauss(n)
    return x, w / math.sqrt(2 * math.pi)


GHX, GHW = gh_nodes(120)


def plan_success(p, rho, R=0):
    """Exact S(pi) under a 1-factor Gaussian copula."""
    q = 1.0 - (1.0 - np.asarray(p, float)) ** (R + 1)
    if rho <= 1e-12:
        return float(np.prod(q))
    if rho >= 1 - 1e-12:
        return float(np.min(q))
    t = PHI_INV(1.0 - q)
    z = GHX[:, None]
    cond = 1.0 - PHI((t[None, :] - math.sqrt(rho) * z) / math.sqrt(1 - rho))
    return float(GHW @ np.prod(cond, axis=1))


def draw_skills(K, d0, kappa, rng):
    d = d0 * rng.random(K) ** (1.0 / kappa)
    return np.clip(1.0 - d, 0.005, 0.9995)


def top_plans(p, H, N):
    """Best-N plans among H-subsets of the 2H+2 most reliable skills (oracle planner)."""
    d = 1.0 - p
    pool = np.argsort(d)[: min(len(p), 2 * H + 2)]
    if len(pool) < H:
        return []
    subs = [tuple(sorted(c)) for c in combinations(pool, H)]
    subs.sort(key=lambda s: sum(d[i] for i in s))
    return subs[:N]


def eps_bound(m, N, alpha, kind="bernstein", s=0.05):
    lt = math.log(2 * N / alpha)
    if kind == "hoeffding":
        return math.sqrt(lt / (2 * m))
    return math.sqrt(2 * s * (1 - s) * lt / m) + (2.0 / 3.0) * lt / m


def run_library(p, H, N, B, alpha, rho, R, rng, beta=0.0, kind="bernstein",
                design_s=None, m_per=None):
    plans = top_plans(p, H, N)
    if len(plans) < 2:
        return None
    S = np.array([plan_success(p[list(pi)], rho, R) for pi in plans])
    n = len(plans)
    m = m_per if m_per is not None else max(4.0, B / (n * H))
    s = design_s if design_s is not None else float(np.clip(S.mean(), 0.02, 0.5))
    eps = eps_bound(m, n, alpha, kind, s)
    Shat = rng.binomial(int(round(m)), np.clip(S, 1e-9, 1 - 1e-9)) / m
    L, U = Shat - eps - beta, Shat + eps + beta
    o = np.argsort(-Shat)
    return dict(ok=bool(L[o[0]] > U[o[1]]), m=m, eps=eps, S=S,
                gap=float(S[o[0]] - S[o[1]]), Lmax=float(L.max()),
                Smax=float(S.max()), n=n)


def phi_of_K(K, H, d0, kappa, B, alpha=0.05, rho=0.3, R=0, N=10, reps=200,
             seed=0, beta=0.0, kind="bernstein"):
    rng = np.random.default_rng(seed)
    ok, gaps, Ls = [], [], []
    for _ in range(reps):
        p = draw_skills(K, d0, kappa, rng)
        r = run_library(p, H, N, B, alpha, rho, R, rng, beta=beta, kind=kind)
        if r is None:
            continue
        ok.append(r["ok"]); gaps.append(r["gap"]); Ls.append(r["Lmax"])
    if not ok:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(ok)), float(np.mean(gaps)), float(np.mean(Ls))


def pred_Kc(H, d0, kappa, B, N=10, alpha=0.05, kind="bernstein", s=0.05):
    m = B / (N * H)
    two_eps = 2 * eps_bound(m, N, alpha, kind, s)
    delta0 = (d0 / kappa) * H ** (1.0 / kappa - 1.0)
    return (delta0 / two_eps) ** kappa


def bisect_Kc(H, d0, kappa, B, N=10, reps=120):
    lo, hi = max(3.0, H + 1.0), 3000.0
    for _ in range(11):
        mid = math.sqrt(lo * hi)
        f, _, _ = phi_of_K(int(round(mid)), H, d0, kappa, B, N=N, reps=reps,
                           seed=int(mid * 7) % 9999)
        if np.isnan(f) or f <= 0.5:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


if __name__ == "__main__":
    print("### sanity: exact copula quadrature")
    p = np.array([0.7, 0.6, 0.8])
    for rho in [0.0, 0.3, 1.0, -0.3]:
        print(f"  rho={rho:+.1f} S={plan_success(p, rho):.6f}")
    print(f"  product={np.prod(p):.6f}  min={p.min():.6f}")

    print("\n### E0  Delta(K) exponent [pred -1/kappa]; Delta(H) [pred 1/kappa-1]")
    for kappa in [0.5, 1.0, 2.0]:
        Ks = np.array([4, 10, 20, 50, 100, 200, 400])
        g = [phi_of_K(int(K), 3, 0.4, kappa, 1e9, reps=40, seed=int(K))[1] for K in Ks]
        sl = np.polyfit(np.log(Ks), np.log(g), 1)[0]
        print(f"  kappa={kappa}: Delta={[round(x,4) for x in g]} slope={sl:+.3f}")
    for kappa in [1.0, 2.0]:
        Hs = np.array([2, 3, 4, 5, 6])
        g = [phi_of_K(60, int(H), 0.4, kappa, 1e9, reps=40, seed=H)[1] for H in Hs]
        sl = np.polyfit(np.log(Hs), np.log(g), 1)[0]
        print(f"  kappa={kappa}: Delta(H)={[round(x,4) for x in g]} slope={sl:+.3f} "
              f"(pred {1/kappa-1:+.3f})")

    print("\n### E1  phi(K) vs B  (H=3,kappa=1,d0=.4,N=10,rho=.3,bernstein)")
    Ks = [4, 7, 10, 20, 50, 100, 400]
    for B in [1e4, 1e5, 1e6, 1e7]:
        row = [phi_of_K(K, 3, 0.4, 1.0, B, reps=200, seed=int(K + B % 1000))[0]
               for K in Ks]
        print(f"  B={B:9.0f} phi=" + " ".join(f"{x:.2f}" for x in row))
    print("  K_c pred:",
          [round(pred_Kc(3, 0.4, 1.0, B, s=0.05), 1) for B in [1e4, 1e5, 1e6, 1e7]])

    print("\n### E2  K_c(B) loglog slope [pred kappa/2]")
    for kappa in [0.5, 1.0, 2.0]:
        Bs2 = [1e4, 1e5, 1e6, 1e7]
        kcs = [bisect_Kc(3, 0.4, kappa, B) for B in Bs2]
        sl = np.polyfit(np.log(Bs2), np.log(kcs), 1)[0]
        print(f"  kappa={kappa}: K_c={[round(k,1) for k in kcs]} slope={sl:.3f}")

    print("\n### E3  phi(p) optimum K=6,H=3,B=3e4 [pred p*=(H-1)/(H-1+R)]")
    for R in [0, 1, 2]:
        ps = np.round(np.arange(0.25, 0.951, 0.05), 3)
        rng0 = np.random.default_rng(11)
        vals = []
        for pbar in ps:
            f = []
            for _ in range(150):
                p = np.clip(pbar + 0.10 * (2 * rng0.random(6) - 1), 0.02, 0.995)
                r = run_library(p, 3, 10, 3e4, 0.05, 0.3, R, rng0)
                if r:
                    f.append(r["ok"])
            vals.append(np.mean(f))
        print(f"  R={R}: " + " ".join(f"{x:.2f}" for x in vals) +
              f"  argmax={ps[int(np.argmax(vals))]} pred={(3-1)/(3-1+R):.2f}")

    print("\n### E4  beta wall (B=1e6,H=3,kappa=1): phi vs K")
    for beta in [0.0, 0.01, 0.02, 0.04]:
        row = [phi_of_K(K, 3, 0.4, 1.0, 1e6, reps=150, seed=K, beta=beta)[0]
               for K in [4, 10, 20, 50, 100]]
        Kb = (0.4 / (2 * beta)) if beta > 0 else float("inf")
        print(f"  beta={beta:.2f} K_beta_pred={Kb:7.1f} phi=" +
              " ".join(f"{x:.2f}" for x in row))

    print("\n### E5  full-library vs queried certification cost (kappa=1,H=3,d0=.4)")
    for K in [4, 20, 50, 100, 400]:
        H, Nq = 3, 10
        delta = 0.4 * K ** -1.0
        m = 2 * math.log(2 * Nq / 0.05) / delta ** 2
        Bq = Nq * H * m
        Nfull = min(K ** H, 1e12)
        print(f"  K={K:4d}: Delta={delta:.4f} m*={m:9.3g} B*(N=10)={Bq:9.3g} "
              f"B*(N=K^H={Nfull:9.3g})={Nfull*H*m:9.3g}")

    print("\n### E6  one-sided lower product certificate: coverage vs sign of rho")
    for rho in [-0.4, -0.2, 0.0, 0.2, 0.4]:
        p3 = np.array([0.7, 0.65, 0.75])
        Strue = plan_success(p3, rho)
        rng = np.random.default_rng(5)
        m, cov = 3000, 0
        for _ in range(400):
            qh = rng.binomial(m, p3) / m
            L = float(np.prod(qh)) - 3 * math.sqrt(math.log(1 / 0.05) / (2 * m))
            cov += (L <= Strue)
        print(f"  rho={rho:+.1f}: S_true={Strue:.4f} prod={np.prod(p3):.4f} "
              f"coverage={cov/400:.3f}")
