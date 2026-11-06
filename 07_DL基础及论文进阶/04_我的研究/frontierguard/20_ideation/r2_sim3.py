"""final numbers: gamma*(beta/Delta), AUC of abstain curves, k* law, channel value."""
import numpy as np, math
rng = np.random.default_rng(7)
Phi = lambda z: 0.5*(1+math.erf(z/math.sqrt(2)))
A, H, K = 0.05, 3, 2
SIG = 0.02
MG = np.unique(np.r_[np.arange(0,40,1.0), np.logspace(0.5,7,900)])
cert = lambda m,D,C: K*H*m + C*np.minimum(1.0, 4*np.exp(-m*D**2/2))
def vcert(D,C): return float(cert(MG,D,C).min())
def mstar(D,C): return float(MG[int(np.argmin(cert(MG,D,C)))])
q0 = lambda b,D: Phi((b-D)/(math.sqrt(2)*SIG))

print("### gamma*(beta/Delta), Delta=5pp  [selection channel, cost=C_fail/(cH)]")
for r in [0.0,0.25,0.5,1.0,1.5,2.0,3.0]:
    C=None
    for lg in np.arange(2,8,0.1):
        C=10**lg*H
        if vcert(0.05,C) < C*q0(r*0.05,0.05): break
    print(f"  beta/Delta={r:4.2f}  q0={q0(r*0.05,0.05):5.3f}  gamma*={C/H:10.3g}")

print("### k* (plans certified jointly) vs gamma, P=100, eta=0.1")
for g in [1e2,1e4,1e6,1e8]:
    C=g*H; best=None
    for k in range(2,101):
        tot = k*H*(2*math.log(2*100/A)/0.05**2) + C*0.1**(k-1)
        if best is None or tot<best[0]: best=(tot,k)
    print(f"  gamma={g:8.0e} k*={best[1]}  log10-fit k*~{1+0.5*math.log10(g):.1f}")

print("### abstain/violation AUC (trapezoid over abstain in [0,1])")
N,P=400000,10
S=rng.beta(5,2,size=(N,P)); b=S.max(1)
def auc(noise,shift):
    est=np.clip(b+shift+rng.normal(0,noise,N),0,1)
    taus=np.linspace(0.3,0.999,400); pts=[]
    for t in taus:
        act=est>=t
        if act.sum()<50: continue
        pts.append((1-act.mean(), float((b[act]<t).mean())))
    pts=sorted(pts); xs=np.array([p[0] for p in pts]); ys=np.array([p[1] for p in pts])
    return float(np.trapezoid(ys,xs)+ys[0]*xs[0])
for lab,no,sh in [("point sd9pp",0.09,0.0),("cert w=5pp",0.03,-0.025),("cert w=10pp",0.03,-0.05)]:
    print(f"  {lab:14s} AUC_violation={auc(no,sh):.4f}")

print("### channel value: gamma* if value comes from abstention instead of ranking")
for dv in [0.1,0.3,0.47]:
    for cab in [0.0,0.01,0.05]:
        # value per decision = dv*(1-cab)*C ; cost = k*H*m*
        m=mstar(0.05,1.0)  # m* depends weakly on C
        cost=K*H*m
        Cstar=cost/(dv*(1-cab)) if dv>0 else np.inf
        print(f"  dViol={dv:.2f} C_ab/C_fail={cab:.2f} -> gamma*={Cstar/H:9.3g}")

print("### w*/Delta closed form check  sqrt(ln(2/a)/ln(2*g*D^2/k))")
for g in [3e4,1e6,1e10]:
    for D in [0.03,0.05,0.10]:
        m=mstar(D,g*H); w=2*math.sqrt(math.log(2/A)/(2*m))
        pred=D*math.sqrt(math.log(2/A)/math.log(2*g*D**2/K))
        print(f"  g={g:8.0e} D={D:.2f} w/D={w/D:.3f} pred={pred/D:.3f}")
