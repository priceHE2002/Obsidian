"""r2-online 最小合成验证（CPU, 0 GPU）.
(a) 重试放大偏置地板 a(p,R)=(R+1)(1-p)^R/q
(b) 计划级 product 证书的偏置地板 |S_ind - S_true| 及其随入口敏感度 b 的标度
(c) 闭环抽样：greedy / uniform / quota × (分层 / 混层) 估计 -> 覆盖 / vacuous
(d) 配额 eps 的 覆盖-探索税 Pareto
"""
import numpy as np

rng = np.random.default_rng(20261102)
SK = dict(S1=0.85, S2=0.70, S8=0.65, S7=0.55)   # 4 技能孤立边际
B = 0.9                                          # 入门层敏感度 (logit 位移/层)
R = 1
NMIN = 30                                        # 分层样本下限，不足则 vacuous


def q_of(p, R):
    return 1.0 - (1.0 - p) ** (R + 1)


def a_of(p, R):
    return (R + 1) * (1 - p) ** R / q_of(p, R)


def p_at(p, u):
    lo = np.log(p / (1 - p)) + B * (u - 1)
    return float(np.clip(1 / (1 + np.exp(-lo)), 0.02, 0.98))


print("== (a) 重试放大 a(p,R)=(R+1)(1-p)^R/q ==")
print(f"{'p':>5} " + " ".join(f"{'R='+str(r):>9}" for r in (0, 1, 2)))
for name, p in SK.items():
    print(f"{p:5.2f} " + " ".join(f"{a_of(p,r):9.4f}" for r in (0, 1, 2)))
for r in (0, 1, 2):
    print(f"R={r}: a(S7)/a(S1) = {a_of(0.55,r)/a_of(0.85,r):.3f}")

print("\n== (b) 技能在三种入门层 u∈{0 难/部署, 1 孤立, 2 易} 的 p ==")
for name, p in SK.items():
    print(f"{name}: " + "  ".join(f"u={u}:{p_at(p,u):.3f}" for u in (0, 1, 2)))

PLANS = [("S1", "S7", "S8"), ("S1", "S2", "S7"), ("S2", "S8", "S7"), ("S1", "S2", "S8")]
print("\n== 计划级 product 证书的偏置地板 ==")
print(f"{'plan':<20}{'R':>3}{'S_dep':>8}{'S_ind':>8}{'floor':>9}")
for pl in PLANS:
    for r in (0, 1, 2):
        Sd = float(np.prod([q_of(p_at(SK[g], 0), r) for g in pl]))
        Si = float(np.prod([q_of(p_at(SK[g], 1), r) for g in pl]))
        print(f"{'-'.join(pl):<20}{r:>3}{Sd:>8.3f}{Si:>8.3f}{(Si-Sd)*100:>8.1f}pp")

print("\n== (b2) 地板 vs 入口敏感度 b（S1-S7-S8, R=1; b=0 必须为 0） ==")
for bb in (0.0, 0.3, 0.6, 0.9, 1.2):
    B = bb
    Sd = float(np.prod([q_of(p_at(SK[g], 0), 1) for g in ("S1", "S7", "S8")]))
    Si = float(np.prod([q_of(p_at(SK[g], 1), 1) for g in ("S1", "S7", "S8")]))
    print(f"b={bb:4.1f}  S_dep={Sd:.3f}  S_ind={Si:.3f}  floor={(Si-Sd)*100:6.1f}pp")
B = 0.9

# ---------------- 闭环 ----------------
DEP = [("S1", "S7", "S8"), ("S1", "S2", "S7"), ("S2", "S8", "S7"),
       ("S1", "S2", "S8"), ("S1", "S8", "S7"), ("S2", "S1", "S7")]
ALLP = [(p, 0) for p in DEP] + [(p, 2) for p in DEP]   # 同一计划有两种入口层可选
S_TRUE = [float(np.prod([q_of(p_at(SK[g], 0), R) for g in p])) for p in DEP]


def mean_at(obs, g, u):
    k, n = obs.get((g, u), (0, 0))
    return (k / n, n) if n else (None, 0)


def pooled_mean(obs, g):
    k = sum(v[0] for (gg, _u), v in obs.items() if gg == g)
    n = sum(v[1] for (gg, _u), v in obs.items() if gg == g)
    return (k / n if n else 0.5), n


def score(opt, obs):
    pl, u = opt
    S = 1.0
    for g in pl:
        m, n = mean_at(obs, g, u)
        if m is None:
            m, _ = pooled_mean(obs, g)
        S *= q_of(m, R)
    return S


def est(pl, obs, strat):
    cs = []
    for g in pl:
        if strat:
            m, n = mean_at(obs, g, 0)
            if n < NMIN:
                return None                     # vacuous：部署层数据不足
        else:
            m, n = pooled_mean(obs, g)
            if n < NMIN:
                return None
        cs.append((min(max(m, 1e-3), 1 - 1e-3), n))
    S = float(np.prod([q_of(p, R) for p, _ in cs]))
    var = sum((S * a_of(p, R)) ** 2 * p * (1 - p) / n for p, n in cs)
    h = 1.96 * np.sqrt(var)
    return max(0.0, S - h), min(1.0, S + h)


def pick(pol, obs, eps):
    if pol == "uniform":
        return int(rng.integers(len(ALLP)))
    if pol == "quota" and rng.random() < eps:
        return int(rng.integers(6))            # 配额强制抽部署层
    sc = [score(o, obs) for o in ALLP]
    best = max(sc)
    cand = [i for i, s in enumerate(sc) if s >= best - 1e-12]
    return int(rng.choice(cand))


def trial(pol, T, eps, strat, reps=40):
    cov = n_ok = vac = 0
    wid = 0.0
    hard = tot = 0
    for _ in range(reps):
        obs = {}
        for _t in range(T):
            j = pick(pol, obs, eps)
            pl, u = ALLP[j]
            tot += 1
            hard += int(u == 0)
            for g in pl:
                ok = rng.random() < p_at(SK[g], u)
                k, n = obs.get((g, u), (0, 0))
                obs[(g, u)] = (k + int(ok), n + 1)
        for i in range(len(DEP)):
            r = est(DEP[i], obs, strat)
            if r is None:
                vac += 1
                continue
            lo, hi = r
            n_ok += 1
            cov += int(lo <= S_TRUE[i] <= hi)
            wid += hi - lo
    return cov / max(n_ok, 1), wid / max(n_ok, 1), vac / (reps * len(DEP)), hard / max(tot, 1)


print("\n== (c) 闭环：名义 95% 区间, 6 个部署计划, R=1 ==")
print(f"{'policy':<9}{'est':<8}{'T':>6}{'coverage':>10}{'width':>8}{'vacuous':>9}{'hard%':>7}")
for T in (600, 2400):
    for pol, eps in (("uniform", 0), ("greedy", 0), ("quota", 0.3)):
        for strat in (True, False):
            c, w, v, h = trial(pol, T, eps, strat)
            print(f"{pol:<9}{'strat' if strat else 'pooled':<8}{T:>6}{c:>10.3f}{w:>8.3f}{v:>9.3f}{h:>7.3f}")

print("\n== (d) 配额 eps: 覆盖 vs 探索税（T=1200, strat） ==")
print(f"{'eps':>5}{'coverage':>10}{'hard%':>8}{'width':>8}{'vacuous':>9}")
for eps in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5):
    c, w, v, h = trial("quota", 1200, eps, True)
    print(f"{eps:>5.2f}{c:>10.3f}{h:>8.3f}{w:>8.3f}{v:>9.3f}")
