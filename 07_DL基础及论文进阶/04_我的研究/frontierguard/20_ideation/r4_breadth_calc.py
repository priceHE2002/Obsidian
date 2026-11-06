# R4 breadth-lever arithmetic (R4 scout). All S/p mapping uses judge model:
# p = sigma(1.15*(S-3.66)) * exec, CoRL; c_ICLR=3.74
import math

def sig(x): return 1/(1+math.exp(-x))
def p(S, c=3.66, exec_=0.78): return sig(1.15*(S-c))*exec_
def S_for(p_target, c=3.66, exec_=0.78): return c + math.log(p_target/exec_/(1-p_target/exec_))/1.15

print("== probability mapping (exec=0.78 / 0.88) ==")
for S in [3.55,3.60,3.71,3.76,3.80,3.82,3.85,3.90,3.95,4.00,4.05,4.10,4.16,4.20]:
    print(f"S={S:.2f}  p78={p(S,exec_=0.78)*100:5.1f}%  p88={p(S,exec_=0.88)*100:5.1f}%  q={sig(1.15*(S-3.66))*100:5.1f}%")
print("S needed for 50%: exec .78 ->", round(S_for(0.5,exec_=0.78),3), "; exec .85 ->", round(S_for(0.5,exec_=0.85),3), "; exec .88 ->", round(S_for(0.5,exec_=0.88),3))
print("S needed for 45%: exec .78 ->", round(S_for(0.45,exec_=0.78),3))
dpdS = 1.15*sig(1.15*(3.82-3.66))*(1-sig(1.15*(3.82-3.66)))*0.78
print("dp/dS at S=3.82, exec .78:", round(dpdS,4), " => dp per +0.01 S:", round(dpdS*0.01*100,3), "pp")
print("dp/dS at S=3.82, exec .88:", round(1.15*sig(1.15*(3.82-3.66))*(1-sig(1.15*(3.82-3.66)))*0.88,4))

print("\n== MDE (paired, alpha .05 two-sided, power .80; factor 2.802) ==")
def mde(sd_pp, n): return 2.802*sd_pp/math.sqrt(n)
for n in [4,6,8,27,40,80,240]:
    print(f"n={n:4d}  sigma_d=5pp -> {mde(5,n):5.2f}pp ; sigma_d=12.13pp -> {mde(12.13,n):5.2f}pp")
# E5 implied sigma_d
print("E5 implied sigma_d at n=80, MDE 3.8pp:", round(3.8*math.sqrt(80)/2.802,2), "pp")
# 3 families: split 800 eps -> n per family; or full 800/family
print("LLM 1->3, fixed 800 eps: n/family=27, MDE =", round(mde(12.13,27),2), "pp")
print("LLM 3x800 eps: MDE/family =", round(mde(12.13,80),2), "pp ; interaction (3 grp, n=80) ~", round(2.802*12.13*math.sqrt(2/80),1), "pp")

print("\n== per-skill data at fixed B=500M total, K=4/6/8 ==")
for K in [4,6,8]:
    print(f"K={K}: steps/skill={500/K:.1f}M"); 
print("effect scaling assumption -> power for true effect 5pp at K=4 (MDE 5.72, sigma_d 5, n=6):")
for eff in [6,5,4.5,4,3]:
    z = eff/(5/math.sqrt(6))
    from statistics import NormalDist
    print(f"  true effect {eff}pp -> power {NormalDist().cdf(z-1.96):.2f}")

print("\n== E1 precision: kappa_hat = max_u |r(g,u)-r(g,iso)|; r(g,u) marginal over 3 partners ==")
for m,label in [(100,"mini-E1"),(300,"main E1")]:
    se_cell = math.sqrt(0.25/m)          # single (g,u) cell from m rollouts
    se_mar  = math.sqrt(0.25/(3*m))      # averaged over 3 partners
    se_diff = math.sqrt(se_mar**2+se_mar**2)
    print(f"{label} m={m}: SE cell={se_cell*100:.2f}pp, marginal={se_mar*100:.2f}pp, kappa diff 95%CI=+-{1.96*se_diff*100:.1f}pp")
for half in [0.5,1.0,1.3]:
    m = (1.96**2)*2*0.25/3/(half/100)**2
    eps = 16*4*m
    print(f"half-width {half}pp needs m={m:.0f}/cell, total eps={eps:.0f}, GPU-h={eps/16286:.1f}, cost=¥{eps/16286*7:.0f}")

print("\n== E1/G6 scaling with K (m=300, 16.29k eps/h, ¥7/h, x1.12) ==")
for K in [4,6,8]:
    pairs=K*K; cells=pairs*4; eps=cells*300+3600*K/4
    g6_cells=K*4; g6=g6_cells*1000*6*6
    base_e1=16*4*300+3600; base_g6=16*1000*36
    print(f"K={K}: E1 eps={eps:.0f} ({eps/16286:.2f}h)  G6 rollouts={g6:.0f} ({g6/16000:.0f}h)  delta vs K=4: +{(eps-base_e1)/16286+(g6-base_g6)/16000:.1f}h -> ¥{(eps-base_e1)/16286*7*1.12+(g6-base_g6)/16000*7*1.12:.0f}")

print("\n== LLM cost per 800 eps (12.0M tokens, 70/30) ==")
for name,pin,pout in [("4o-mini",0.15,0.60),("Haiku3.5",0.80,4.00),("plan($3/$12)",3.0,12.0),("GPT-4o($2.5/$10)",2.5,10.0)]:
    usd = 8.4*pin + 3.6*pout
    print(f"{name}: ${usd:.1f} = ¥{usd*7.2:.0f} per family (800 eps)")

print("\n== budget stress ==")
base=3313
for name,inc in [("6skills",160),("8skills",310),("LLMlight",250),("LLM3full",800),("Aero",550),("6+LLM",410),("8+Aero",860),("8+LLM",560)]:
    print(f"{name}: {base}+{inc}={base+inc}; +50% price scenario (base 4549)+{inc}={4549+inc}")
print("headroom to hard cap 4600:", 4600-3313, "; to user cap 5000:", 5000-3313)
