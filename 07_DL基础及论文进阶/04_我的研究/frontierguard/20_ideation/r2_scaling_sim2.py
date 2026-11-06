#!/usr/bin/env python3
"""
r2-scaling v2: CPU verification of the library certification scaling law.

Model
-----
K skills, per-step success q_g, deficits d_g = 1-q_g ~ G(d) = (d/d0)^kappa, d in [0,d0].
Plan pi = H distinct skills; retries R: q_i = 1-(1-p_i)^(R+1).
Dependence: 1-factor copula with loadings a_i in [-1,1]:
    U_i = Phi(a_i Z + sqrt(1-a_i^2) e_i),  Z,e_i iid N(0,1)  =>  corr(i,j)=a_i a_j.
    S(pi) = E_Z[ prod_i (1 - Phi((t_i - a_i Z)/sqrt(1-a_i^2))) ]  (exact, 1-D quadrature).
    a_i = +sqrt(rho) all i  -> equicorrelation rho >= 0 (comonotone limit rho=1: S=min q).
    a_i = (-1)^i sqrt(rho)  -> negative cross-correlation.

Certification of the argmax over the N queried plans, m joint rollouts each:
    eps = Bernstein(s) or Hoeffding; L_i = Shat_i - eps - beta; U_i = Shat_i + eps + beta;
    success iff L_(1) > U_(2).   Budget B = N*H*m.

Predictions
-----------
P1  Delta(K,H,q) = q^{H-1} * (d0/kappa) * H^{1/kappa-1} * K^{-1/kappa}
P2  m* = 2 log(2N/a)/(Delta-2beta)^2 ;  K_c ~ B^{kappa/2} (N, alpha fixed)
P3  beta wall: K_beta = [q^{H-1} d0 H^{1/kappa-1}/(2 kappa beta)]^kappa  (B-independent)
P4  phi(p) interior optimum; retries shift argmax p* down (q* in [(H-2)/H,(H-1)/H])
"""
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
    """Exact S(pi) for 1-factor copula with per-skill loadings a."""
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
    d = d0 * rng.random(K) ** (1.0 / kappa)
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
             seed=0, beta=0.0, kind="bernstein"):
    rng = np.random.default_rng(seed)
    ok, gaps = [], []
    for _ in range(reps):
        p = draw_skills(K, d0, kappa, rng)
        r = run_library(p, H, N, B, alpha, rho, R, rng, beta, kind)
        if r:
            ok.append(r["ok"]); gaps.append(r["gap"])
    if not ok:
        return float("nan"), float("nan")
    return float(np.mean(ok)), float(np.mean(gaps))


def delta_pred(K, H, q, d0, kappa):
    return q ** (H - 1) * (d0 / kappa) * H ** (1.0 / kappa - 1.0) * K ** (-1.0 / kappa)


def bisect_Kc(H, d0, kappa, B, N=10, reps=250, hi=3000.0):
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
    print("### S0 exact copula sanity")
    p3 = np.array([0.7, 0.6, 0.8])
    print("  rho=0  ", round(plan_success(p3, rho=0.0), 6), " product", round(float(np.prod(p3)), 6))
    print("  rho=1  ", round(plan_success(p3, rho=0.999999), 6), " min", float(p3.min()))
    print("  rho=.3 ", round(plan_success(p3, rho=0.3), 6))
    alt = np.array([0.3, -0.3])
    print("  alt-loadings (corr=-0.09) H=2", round(plan_success(np.array([0.7, 0.7]), alt), 6),
          " product", 0.49)

    # ---------------- E0: margin law ----------------
    print("\n### E0  Delta law: regress log Delta on (log K, log H, H log q)")
    rows = []
    for kappa in [1.0, 2.0]:
        X, y = [], []
        for H in [2, 3, 4, 5, 6]:
            for K in [10, 20, 50, 100, 200, 400]:
                for pbar in [0.5, 0.7, 0.85]:
                    rng = np.random.default_rng(K * 7 + H * 13 + int(pbar * 100))
                    p = np.clip(pbar + 0.10 * (2 * rng.random(K) - 1), 0.05, 0.99)
                    r = run_library(p, H, 10, 1e12, rho=0.3, R=0, rng=rng)
                    if r is None or r["gap"] <= 1e-12:
                        continue
                    q = float(np.clip(p.mean(), 0.02, 0.99))
                    X.append([1, math.log(K), math.log(H), H * math.log(q)])
                    y.append(math.log(r["gap"]))
        X, y = np.array(X), np.array(y)
        coef, res, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coef
        ss = 1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"  kappa={kappa}: coef(logK)={coef[1]:+.3f} [pred -1/kappa={-1/kappa:+.2f}]"
              f"  coef(logH)={coef[2]:+.3f} [pred {1/kappa-1:+.2f}]"
              f"  coef(Hlogq)={coef[3]:+.3f} [pred +1]  R2={ss:.3f}  n={len(y)}")
        rows.append((kappa, coef.tolist(), ss))

    # ---------------- E1: phi(K) collapse ----------------
    print("\n### E1  phi(K) [H=3,kappa=1,d0=.4,N=10,rho=.3,bernstein]")
    Ks = [4, 7, 10, 20, 50, 100, 400]
    for B in [1e4, 1e5, 1e6, 1e7]:
        row = [phi_of_K(K, 3, 0.4, 1.0, B, reps=240, seed=K * 31 + int(B % 977))[0]
               for K in Ks]
        print(f"  B={B:9.0f} phi=" + " ".join(f"{x:.2f}" for x in row))

    # ---------------- E2: K_c(B) exponent ----------------
    print("\n### E2  K_c(B) loglog slope [pred kappa/2] (H=3,d0=.4)")
    for kappa in [0.5, 1.0, 2.0]:
        Bs = [1e5, 1e6, 1e7, 1e8]
        kcs = [bisect_Kc(3, 0.4, kappa, B) for B in Bs]
        sl = np.polyfit(np.log(Bs), np.log(kcs), 1)[0]
        print(f"  kappa={kappa}: K_c={[round(k,1) for k in kcs]} slope={sl:.3f}")

    # ---------------- E3: phi(p) interior optimum ----------------
    print("\n### E3  phi(p): K=6,H=3,B=1e6,N=10  [pred argmax q* in ((H-2)/H,(H-1)/H)]")
    for R in [0, 1, 2]:
        ps = np.round(np.arange(0.20, 0.951, 0.05), 3)
        rng0 = np.random.default_rng(7)
        vals = []
        for pbar in ps:
            f = []
            for _ in range(200):
                p = np.clip(pbar + 0.08 * (2 * rng0.random(6) - 1), 0.02, 0.995)
                r = run_library(p, 3, 10, 1e6, rho=0.3, R=R, rng=rng0)
                if r:
                    f.append(r["ok"])
            vals.append(np.mean(f))
        imax = int(np.argmax(vals))
        qs = 1 - (1 - ps[imax]) ** (R + 1)
        print(f"  R={R}: " + " ".join(f"{x:.2f}" for x in vals) +
              f"  argmax p={ps[imax]:.2f} q*={qs:.2f}")

    # ---------------- E4: beta wall ----------------
    print("\n### E4  beta wall (B=1e7,H=3,kappa=1): phi vs K")
    for beta in [0.0, 0.005, 0.01, 0.02]:
        row = [phi_of_K(K, 3, 0.4, 1.0, 1e7, reps=200, seed=K + 3)[0]
               for K in [4, 10, 20, 50, 100]]
        Kb = (0.36 * 0.4 / (2 * beta)) if beta > 0 else float("inf")
        print(f"  beta={beta:.3f} K_beta_pred={Kb:7.1f} phi=" +
              " ".join(f"{x:.2f}" for x in row))

    # ---------------- E5: cost table ----------------
    print("\n### E5  certification budget B* (episodes), H=3, alpha=.05")
    print("     q=0.60 -> Delta = q^2*d0/K = .144/K ; N=10 queried vs N=K^H full")
    for K in [4, 20, 50, 100, 400]:
        H = 3
        q = 0.6
        delta = q ** (H - 1) * (1 - q) / K
        for Nq in [10]:
            lt = math.log(2 * Nq / 0.05)
            m = 2 * lt / delta ** 2
            print(f"  K={K:4d}  Delta={delta:.5f}  m*={m:9.3g}  "
                  f"B*(N=10)={Nq*H*m:9.3g}  B*(N=K^H={K**H:,.0f})={K**H*H*m:9.3g}")

    # ---------------- E6: dependence sign ----------------
    print("\n### E6  coverage of the product lower certificate vs sign of dependence")
    print("     (H=3, q=(.7,.65,.75); product lower bound L = prod(qhat) - 3*eps)")
    for name, a in [("indep", np.zeros(3)),
                    ("pos .3 (a=+.55)", np.full(3, math.sqrt(0.3))),
                    ("neg (a=+,-,+ .3)", np.array([math.sqrt(.3), -math.sqrt(.3), math.sqrt(.3)]))]:
        Strue = plan_success(np.array([0.7, 0.65, 0.75]), a)
        rng = np.random.default_rng(5)
        m, cov, ubad = 3000, 0, 0
        for _ in range(500):
            qh = rng.binomial(m, np.array([0.7, 0.65, 0.75])) / m
            L = float(np.prod(qh)) - 3 * math.sqrt(math.log(1 / 0.05) / (2 * m))
            U = float(np.prod(qh)) + 3 * math.sqrt(math.log(1 / 0.05) / (2 * m))
            cov += (L <= Strue); ubad += (U < Strue)
        print(f"  {name:18s}: S_true={Strue:.4f} prod={0.3413:.4f} "
              f"lower-cov={cov/500:.3f}  upper-viol={ubad/500:.3f}")

    # ---------------- E7: self-averaging of Delta ----------------
    print("\n### E7  coefficient of variation of Delta vs K (self-averaging => sharp K_c)")
    for K in [4, 20, 100, 400]:
        rng = np.random.default_rng(99)
        g = [run_library(draw_skills(K, 0.4, 1.0, rng), 3, 10, 1e12, rng=rng)["gap"]
             for _ in range(300)]
        g = np.array([x for x in g if x > 1e-9])
        print(f"  K={K:4d}: mean={g.mean():.5f} cv={g.std()/g.mean():.3f}")
