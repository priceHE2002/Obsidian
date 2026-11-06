"""E0 v2: certify/point band in the margin, w*/Delta scaling, abstain violation curve."""
import numpy as np, math, json
rng = np.random.default_rng(20261102)
Phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
ALPHA, H, K = 0.05, 3, 2
SIG_M = 0.02   # plan-level noise of the marginal-only estimate at n=1000/plan (21's D=0.018)

def cost_cert(m, D, C):
    return K * H * np.asarray(m, float) + C * np.minimum(1.0, 4 * np.exp(-np.asarray(m, float) * D**2 / 2))
MG = np.unique(np.r_[np.arange(0, 40, 1.0), np.logspace(0.5, 6, 600)])
def best(mgrid=None):
    return lambda D, C: (lambda v: (float(MG[int(np.argmin(v))]), float(v.min())))(cost_cert(MG, D, C))

def q0(beta, D, sig=SIG_M):
    return Phi((beta - D) / (math.sqrt(2) * sig))

print("### A. certify band in Delta (beta=0, point-estimate error from noise only)")
rowsA = []
for g in [1e2, 1e3, 1e4, 1e5, 1e6, 1e8]:
    C = g * H
    lo = hi = None
    for D in np.arange(0.002, 0.60, 0.002):
        _, vc = best()(D, C)
        if vc < C * q0(0.0, D):
            lo = D if lo is None else lo; hi = D
    rowsA.append((g, lo, hi))
    print(f"  gamma={g:8.0e}  certify band Delta in [{(lo or 0)*100:5.2f}, {(hi or 0)*100:6.2f}] pp "
          f"(width {(0 if lo is None else (hi-lo)*100):5.2f} pp)")

print("### B. w*/Delta over 6 decades of gamma (D=0.05 and 0.10)")
rowsB = []
for g in [3e4, 1e5, 1e6, 1e7, 1e8, 1e10]:
    for D in [0.03, 0.05, 0.10]:
        C = g * H
        m, v = best()(D, C)
        w = 2 * math.sqrt(math.log(2 / ALPHA) / (2 * m)) if m > 0 else float('inf')
        rowsB.append((g, D, m, w, w / D))
        print(f"  gamma={g:8.0e} D={D:.2f} m*={m:9.1f} w*={w:6.3f} w*/D={w/D:5.3f}")

print("### C. Delta(p) and the reliability band (latent class, H=3, lam=.5)")
def latent(p, spread=0.55):
    a = min(.995, p + spread*p*(1-p)*2); b = max(.005, p - spread*p*(1-p)*2)
    q = .5*a + .5*b
    return q**H, .5*a**H + .5*b**H, q**H - (.5*a**H + .5*b**H)
rowsC = []
for p in [.3,.4,.5,.6,.7,.8,.9,.95]:
    pr, S, be = latent(p); _, S2, _ = latent(p, spread=.44)
    D = abs(S - S2); rowsC.append((p, pr, S, be, D))
    print(f"  p={p:.2f} prod={pr:.3f} S={S:.3f} beta={be:+.3f} Delta={D*100:5.2f}pp "
          f"needed gamma*={1/D**2:9.0f}")

print("### D. abstain / violation curve (P=10 plans, truth S~Beta(5,2))")
N, P = 400000, 10
S = rng.beta(5, 2, size=(N, P)); bestS = S.max(1)
def curve(noise, shift, alpha_cov=0.05, label=""):
    est = np.clip(bestS + shift + rng.normal(0, noise, N), 0, 1)
    out = []
    for tau in [.6,.7,.8,.9,.95]:
        act = est >= tau
        if act.sum() == 0: out.append((tau, 1.0, 0.0)); continue
        out.append((tau, float(1-act.mean()), float((bestS[act] < tau).mean())))
    print(f"  {label:26s} " + " | ".join(f"t={t:.2f} abst={a:.2f} viol={v:.3f}" for t,a,v in out))
    return out
d1 = curve(0.09, 0.0, label="point est (sd 9pp)")
d2 = curve(0.03, 0.0, label="cert L, w=5pp (sd 3pp)")
d3 = curve(0.03, -0.025, label="cert L, w=5pp, shift-.025")
d4 = curve(0.03, -0.05, label="cert L, w=10pp, shift-.05")
json.dump({"A": rowsA, "B": rowsB, "C": rowsC}, open("r2_sim2.json","w"), indent=1)
