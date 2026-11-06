"""E0 synthetic oracle for the certification cost-benefit phase diagram.
Units: c = 1 (one skill rollout). A joint plan rollout of horizon H costs H.
C = C_fail / c  (failure cost in rollout units); gamma = C / H.
"""
import numpy as np, math, json
rng = np.random.default_rng(20261102)
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
ALPHA = 0.05

def cost_cert(m, Delta, C, k=2, H=3):
    """expected total cost of certifying k plans with m joint rollouts each."""
    return k * H * m + C * np.minimum(1.0, 4 * np.exp(-m * Delta**2 / 2))

def best_m(Delta, C, k=2, H=3, mgrid=None):
    if mgrid is None:
        mgrid = np.unique(np.r_[np.arange(0, 50, 1.0), np.logspace(0, 5, 400)])
    v = cost_cert(mgrid, Delta, C, k, H)
    i = int(np.argmin(v))
    return float(mgrid[i]), float(v[i])

def point_cost(q0, C):
    return C * q0

def q0_marginal(beta, Delta, sig_m):
    """P(marginal planner ranks wrong) with systematic bias gap beta."""
    return Phi((beta - Delta) / (math.sqrt(2) * sig_m))

# ---------- 1. boundary gamma* Delta^2 for beta=0 ----------
def boundary(Delta, beta=0.0, sig_m=0.02, k=2, H=3, lo=-3, hi=9):
    """smallest log10(gamma) at which certification beats the point policy."""
    q0 = q0_marginal(beta, Delta, sig_m)
    if q0 < 1e-6:
        return np.inf
    for lg in np.arange(lo, hi, 0.05):
        C = 10**lg * H          # C/H = gamma
        _, vc = best_m(Delta, C, k, H)
        if vc < point_cost(q0, C) - 1e-12:
            return 10**lg
    return np.inf

print("=== 1. certify/point boundary (beta=0, sig_m=2pp, k=2, H=3)")
rows = []
for D in [0.02, 0.03, 0.05, 0.08, 0.12, 0.20]:
    g = boundary(D)
    rows.append((D, g, g * D**2 if np.isfinite(g) else np.nan))
    print(f"  Delta={D:.3f}  gamma*={g:9.3g}  gamma**Delta^2={g*D*D:6.3f}")

# ---------- 2. beta axis: does certification ever pay when beta small? ----------
print("=== 2. beta/Delta axis (Delta=0.05, gamma up to 1e6)")
rows2 = []
for r in [0.0, 0.25, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5]:
    beta = r * 0.05
    q0 = q0_marginal(beta, 0.05, 0.02)
    C = 1e6 * 3
    _, vc = best_m(0.05, C, 2, 3)
    vp = point_cost(q0, C)
    rows2.append((r, q0, vc, vp, vc < vp))
    print(f"  beta/Delta={r:4.2f} q0={q0:6.3f} cert_cost={vc:10.3g} point_cost={vp:10.3g} certify={vc<vp}")

# ---------- 3. optimal width vs margin ----------
print("=== 3. w*/Delta (w=2*sqrt(ln(2/alpha)/(2m)))")
rows3 = []
for g in [3, 10, 100, 1e3, 1e4, 1e6]:
    for D in [0.05, 0.10]:
        C = g * 3
        m, _ = best_m(D, C, 2, 3)
        w = 2 * math.sqrt(math.log(2 / ALPHA) / (2 * m)) if m > 0 else np.inf
        rows3.append((g, D, m, w, w / D))
        print(f"  gamma={g:8.3g} Delta={D:.2f} m*={m:8.2f} w*={w:6.3f} w*/Delta={w/D:5.3f}")

# ---------- 4. cube-root regime (continuous commitment / newsvendor) ----------
print("=== 4. cube-root law w*=(2k/A)^(1/3), A=C_fail*sqrt(2 ln(C_opp/C_fail))")
rows4 = []
for Cfail in [1e3, 1e4, 1e5, 1e6, 1e7]:
    Copp = 1e2 * Cfail
    A = Cfail * math.sqrt(2 * math.log(Copp / Cfail))
    kap = 2 * 3 * math.log(2 / ALPHA)      # 2*c*H*ln(2/alpha)
    w = (2 * kap / A)**(1 / 3)
    rows4.append((Cfail, w))
    print(f"  C_fail={Cfail:9.0f}  w*={w:6.4f} ({w*100:5.2f} pp)")

# ---------- 5. k* (how many plans to certify) vs gamma ----------
print("=== 5. k* vs gamma: k = min P s.t. marginal screening tail eta^(k-1) < budget")
def kstar(g, D=0.05, P=100, eta=0.1, H=3, alpha=ALPHA):
    C = g * H
    ks = []
    for k in range(2, P + 1):
        cost = k * H * (2 * math.log(2 * P / alpha) / D**2)
        # risk that true best is outside top-k after screening
        risk = eta**(k - 1)
        ks.append((cost + C * risk, k))
    return min(ks)[1]
rows5 = []
for g in [1e1, 1e2, 1e3, 1e4, 1e6, 1e8]:
    k = kstar(g)
    rows5.append((g, k))
    print(f"  gamma={g:9.3g}  k*={k}")

# ---------- 6. (p,gamma) diagram via latent-class dependence ----------
print("=== 6. skill reliability p: Delta(p) from latent-class, certify region")
def latent(p, H=3, lam=0.5, spread=0.55):
    """two-state latent dependence; returns (product-plug-in, true S, bias)."""
    a = min(0.995, p + spread * p * (1 - p) * 2)
    b = max(0.005, p - spread * p * (1 - p) * 2)
    q = lam * a + (1 - lam) * b
    S = lam * a**H + (1 - lam) * b**H
    return q**H, S, q**H - S
rows6 = []
for p in [0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 0.97]:
    prod, S, beta = latent(p)
    rows6.append((p, prod, S, beta))
    print(f"  p={p:.2f} prod={prod:.3f} S={S:.3f} beta={beta:+.3f}")

# Delta(p): spread of S across plans differing by one skill's reliability
rows6b = []
for p in [0.3, 0.45, 0.6, 0.7, 0.8, 0.9, 0.97]:
    _, S1, _ = latent(p)
    _, S2, _ = latent(p, spread=0.55 * 0.8)
    D = abs(S1 - S2)
    rows6b.append((p, D))
    print(f"  p={p:.2f} Delta(p)={D:.4f}")

# ---------- 7. abstain vs success curve ----------
print("=== 7. abstain/success curve (P=10 plans, S~Beta, threshold tau)")
P, N = 10, 200000
S = rng.beta(5, 2, size=(N, P))            # heterogeneous plan quality
best = S.max(axis=1)
for label, width in [("point(w=0)", 0.0), ("cert(w=0.10)", 0.10), ("cert(w=0.05)", 0.05)]:
    est = best - width / 2 + rng.normal(0, width / 3, N) if width else best + rng.normal(0, 0.05, N)
    est = np.clip(est, 0, 1)
    out = []
    for tau in [0.5, 0.6, 0.7, 0.8, 0.9]:
        act = est >= tau
        out.append((tau, float(1 - act.mean()), float((best * act).sum() / max(act.sum(), 1))))
    print(f"  {label:14s} " + "  ".join(f"tau={t}: abst={a:.2f} succ={s:.3f}" for t, a, s in out))

json.dump({"b1": rows, "b2": rows2, "b3": rows3, "b4": rows4,
           "b5": rows5, "b6": rows6, "b6b": rows6b},
          open("/Users/heyuhang/Desktop/frontierguard/20_ideation/r2_practical_sim.json", "w"), indent=1)
print("saved")
