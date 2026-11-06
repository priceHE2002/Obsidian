# -*- coding: utf-8 -*-
"""Round-4 修复回路侦察 v2：合成实算（A1′ 下的接口/修复代数 + 功效 + 预算 + 概率映射）。
所有数字只用于 r4_repair.md 的"合成实算/估计"标注；不是真实技能库实验。

结构：context(0) -> S1(1)/S2(2) -> S7(3) -> S8(4)，H=3。
计划 L1: (0,1)->(1,3)->(3,4)；L2: (0,2)->(2,3)->(3,4)；各 p=0.5。
S = q1 q2 q3, q1 用 RHO0(u1)=1/4（预处理给定），q2/q3 用 T[pair]。
候选修复动作（E1 后按最高权重层预注册）：(S1,u1a),(S2,u1b),(S7,u2*),(S8,u3*)。
"""
import itertools
import math

import numpy as np

rng = np.random.default_rng(20261102)

U = 4
H = 3
RHO0 = np.ones(U) / U
PLANS = {
    "L1": [(0, 1), (1, 3), (3, 4)],
    "L2": [(0, 2), (2, 3), (3, 4)],
}
P_PLAN = {"L1": 0.5, "L2": 0.5}
# 候选动作在计划实际用到的层上（u2*=1 for S7, u3*=2 for S8, step1 层 0）
CAND = [(1, 0), (2, 0), (3, 1), (4, 2)]
PAIRS = set(p for ps in PLANS.values() for p in ps)


def build_lib(rng, kappa, defect_cells, t_conc):
    """r[g,j,u]=base[g,j]+lay[j,u]; defect_cells 里的 (j,u) 下沉 kappa。
    t_conc: 计划诱导的入口层质量集中在候选层上的比例。"""
    nodes = 5
    base = rng.uniform(0.70, 0.82, size=(nodes, nodes))
    lay = rng.normal(0.0, 0.010, size=(nodes, U))
    for (j, u) in defect_cells:
        lay[j, u] -= kappa
    r = np.clip(base[:, :, None] + lay[None, :, :], 0.05, 0.95)
    T = {}
    # T[(g,j)] 是"执行 j 后的出口层分布"= 下一步的入口层：S1/S2 之后进入 S7 -> 层1；S7 之后进入 S8 -> 层2
    USTAR = {(0, 1): 1, (0, 2): 1, (1, 3): 2, (2, 3): 2, (3, 4): 3}
    for pair in PAIRS:
        row = np.full(U, (1.0 - t_conc) / (U - 1))
        row[USTAR[pair]] = t_conc
        T[pair] = row
    return r, T


def q(r, rho, pair):
    g, j = pair
    return float(sum(rho[u] * r[g, j, u] for u in range(U)))


def plan_value(r, T, plan):
    qs = [q(r, RHO0, plan[0]), q(r, T[plan[0]], plan[1]), q(r, T[plan[1]], plan[2])]
    return float(np.prod(qs)), qs


def value(r, T):
    return float(sum(P_PLAN[n] * plan_value(r, T, p)[0] for n, p in PLANS.items()))


def sens(r, T):
    """w_{(pair,u)} = dV/dr，A1′ 下精确：Σ_π p(π)(S_π/q_k)ρ_k(u)。"""
    w = {}
    for name, plan in PLANS.items():
        s, qs = plan_value(r, T, plan)
        rhos = [RHO0, T[plan[0]], T[plan[1]]]
        for k, pair in enumerate(plan):
            for u in range(U):
                w[(pair, u)] = w.get((pair, u), 0.0) + P_PLAN[name] * (s / qs[k]) * rhos[k][u]
    return w


def action_v(r, T, w, j, u, eta_frac=0.6):
    """v_a = Σ_j w_{(g,j,u)} η_a；η_a = eta_frac × 该 (j,u) 的缺陷深度。"""
    depth = 0.0
    for pair in PAIRS:
        if pair[1] != j:
            continue
        g = pair[0]
        base = float(np.mean([r[g, j, v] for v in range(U)]))
        depth = max(depth, base - r[g, j, u])
    eta = eta_frac * max(depth, 0.0)
    v = sum(w.get((pair, u), 0.0) * eta for pair in PAIRS if pair[1] == j)
    return v, eta, depth


def apply_repair(r, j, u, delta):
    r2 = r.copy()
    for g in range(r.shape[0]):
        r2[g, j, u] = min(0.95, r2[g, j, u] + delta)
    return r2


def gains_for(rng, kappa, t_conc):
    """4 候选、2 槽；G=top2、U=4 摊薄 x=0.161、R=random2、N=0。返回 pp。"""
    defect_cells = [CAND[i] for i in rng.choice(4, 2, replace=False)]
    r, T = build_lib(rng, kappa, defect_cells, t_conc)
    V0 = value(r, T)
    w = sens(r, T)
    v = {a: action_v(r, T, w, *a) for a in CAND}
    order = sorted(CAND, key=lambda a: -v[a][0])
    # G: 2 轮，每轮修 1 个（第二轮的 w 与 η 用修后核重算）
    rG = r
    for a in order[:2]:
        w2 = sens(rG, T)
        _, eta2, _ = action_v(rG, T, w2, *a)
        rG = apply_repair(rG, *a, eta2)
    # U: 同预算摊薄 4 个
    x_u = (2 * 2.80 + 0.4 - 4 * 1.25) / (4 * 1.55)  # =0.161
    rU = r
    for a in CAND:
        w2 = sens(rU, T)
        _, eta2, _ = action_v(rU, T, w2, *a)
        rU = apply_repair(rU, *a, eta2 * x_u)
    # R: 随机 2 个
    rR = r
    for a in [CAND[i] for i in rng.choice(4, 2, replace=False)]:
        w2 = sens(rR, T)
        _, eta2, _ = action_v(rR, T, w2, *a)
        rR = apply_repair(rR, *a, eta2)
    # 固定顺序（列表前 2，代表"同预算均匀列表"）
    rF = r
    for a in CAND[:2]:
        w2 = sens(rF, T)
        _, eta2, _ = action_v(rF, T, w2, *a)
        rF = apply_repair(rF, *a, eta2)
    # oracle 最优两动作
    best = 0.0
    for combo in itertools.combinations(CAND, 2):
        rc = r
        for a in combo:
            w2 = sens(rc, T)
            _, eta2, _ = action_v(rc, T, w2, *a)
            rc = apply_repair(rc, *a, eta2)
        best = max(best, value(rc, T) - V0)
    return (np.array([value(rG, T) - V0, value(rU, T) - V0, value(rR, T) - V0,
                      value(rF, T) - V0, best]) * 100)


def four_arm(kappa, t_conc, ntrial=400):
    rows = np.array([gains_for(rng, kappa, t_conc) for _ in range(ntrial)])
    m = rows.mean(0)
    return dict(G=m[0], U=m[1], R=m[2], F=m[3], O=m[4],
                GminusU=m[0] - m[1], GminusR=m[0] - m[2], greedy=m[0] / m[4])


# ---------- 1) 一阶界 / 价值墙 ----------
def check_bound(ntrial=300):
    mx, mn, ok = [], [], True
    for kappa in (0.06, 0.12, 0.20):
        e = []
        for _ in range(ntrial):
            defect_cells = [CAND[i] for i in rng.choice(4, 2, replace=False)]
            r, T = build_lib(rng, kappa, defect_cells, rng.uniform(0.2, 0.9))
            V0 = value(r, T)
            w = sens(r, T)
            a = CAND[int(rng.integers(0, 4))]
            v, eta, depth = action_v(r, T, w, *a)
            r2 = apply_repair(r, *a, eta)
            e.append(abs((value(r2, T) - V0) - v) * 100)
            ok = ok and ((value(r2, T) - V0) <= H * depth + 1e-9)
        mx.append(max(e))
        mn.append(float(np.mean(e)))
    return mx, mn, ok


# ---------- 2) 规模 K 的选靶优势 ----------
def scaling_gap(K, ntrial=2000, n_defect=2, n_slots=2):
    hits = []
    for _ in range(ntrial):
        idx = set(rng.choice(K, n_defect, replace=False))
        ch = set(rng.choice(K, n_slots, replace=False))
        hits.append(len(idx & ch) / n_defect)
    return float(np.mean(hits))


# ---------- 3) 功效 / 预算 / 概率 ----------
T975 = {2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179}
T80 = {2: 0.885, 3: 0.941, 4: 0.941, 5: 0.920, 6: 0.906, 7: 0.896,
       8: 0.889, 9: 0.883, 10: 0.879, 11: 0.876, 12: 0.873}


def mde(n, sd):
    df = max(n - 1, 2)
    return (T975[df] + T80[df]) * sd / math.sqrt(n)


def budget(n_seeds=6):
    c0, c1, cr = 1.25, 1.55, 0.20
    per_G = 2 * (c0 + c1) + 2 * cr       # 6.0
    per_R = per_G
    per_U = 4 * c0 + 4 * c1 * ((per_G - 4 * c0) / (4 * c1))
    repair = n_seeds * (per_G + per_R + per_U)
    ev = 4 * n_seeds * 0.45
    dose = c0 + 2.0 * c1                 # 2 点剂量校准 1 seed
    raw = repair + ev + dose
    return dict(per_G=per_G, per_R=per_R, per_U=per_U, repair=repair,
                ev=ev, dose=dose, raw=raw, buffered=raw * 1.12, cny=raw * 1.12 * 7)


def prob(S, c, disc, ex):
    q = 1 / (1 + math.exp(-1.15 * (S - c))) * disc
    return q, q * ex


if __name__ == "__main__":
    print("=" * 76)
    print("1) 一阶界 |ΔS-ΣwΔr| 与价值墙 ΔS<=Hκ (合成 300 trial/κ)")
    mx, mn, ok = check_bound()
    for k, a, b in zip((0.06, 0.12, 0.20), mx, mn):
        print(f"  κ={k*100:.0f}pp: max={a:.2f}pp mean={b:.2f}pp; ΔS≤Hκ 全成立={ok}")

    print("=" * 76)
    print("2) 四臂合成（4 候选、2 槽、同预算；单位 pp，400 trial）")
    for kappa in (0.06, 0.12, 0.20):
        for tc in (0.25, 0.6, 0.9):
            o = four_arm(kappa, tc)
            print(f"  κ={kappa*100:2.0f}pp c={tc:.2f}: G={o['G']:5.2f} U={o['U']:4.2f} "
                  f"R={o['R']:4.2f} F={o['F']:4.2f} O={o['O']:5.2f} | "
                  f"G-U={o['GminusU']:4.2f} G-R={o['GminusR']:4.2f} G/O={o['greedy']:.3f}")

    print("=" * 76)
    print("3) 动作空间 K：random 2 槽捕获缺陷质量")
    for K in (4, 8, 16, 50, 100):
        print(f"  K={K:3d}: capture={scaling_gap(K):.2%}")

    print("=" * 76)
    print("4) MDE（双侧 5%/power 80%）")
    for n in (3, 4, 6, 8, 12):
        print(f"  n={n:2d}: " + "  ".join(f"sd={sd}->{mde(n, sd):.2f}pp" for sd in (1.0, 1.5, 2.0)))

    print("=" * 76)
    print("5) 预算（6 seeds）")
    for k, v in budget().items():
        print(f"  {k}: {v:.1f}")

    print("=" * 76)
    print("6) 概率映射（exec=0.78）")
    for S in (3.80, 3.82, 3.85, 3.87, 3.89, 3.92):
        qc, ec = prob(S, 3.66, 0.97, 0.78)
        qi, ei = prob(S, 3.74, 0.9025, 0.78)
        print(f"  S={S:.2f}: CoRL 质量={qc:.2%} 期望={ec:.2%} | ICLR 质量={qi:.2%} 期望={ei:.2%}")
