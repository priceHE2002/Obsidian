# -*- coding: utf-8 -*-
"""26_synthesis_v3 预算复算：A100 ¥7/h，12% 重跑缓冲，电 120 + 存储 80 + LLM 641。"""

TRAIN_H = {8192: 9.51, 2048: 3.14}   # h/run（500M / 125M steps, A100 单卡，含开销）
EVAL_TP = 16_000                      # rollouts/h（评测，G6/E1/E3 统一）

ROWS = [  # (组, runs, raw GPU-h, 说明)
    ("G1a", 18, 18*TRAIN_H[8192],  "3 方法×6 seeds @N=8192"),
    ("G1b", 12, 12*TRAIN_H[2048],  "3 方法×4 seeds @N=2048"),
    ("G6",  0, 576_000/EVAL_TP,    "CRN 576k rollouts"),
    ("E1",  0, 22_800/EVAL_TP,     "16 对×U=4×m=300 + R 指数 3.6k"),
    ("E3",  0, 400_000/EVAL_TP,    "200 计划×1,000×2 任务"),
    ("E4b", 0, 24.0,               "成本-精度 N=10/100/1,000×K=4/20"),
    ("G0",  0, 20.0,               "集成/调试/pilot"),
]
BUF, PRICE = 1.12, 7.0
MISC = 120 + 80 + 641
HARD, TARGET_CAP = 4600, None

raw = sum(r[2] for r in ROWS)
bh  = raw * BUF
rent = bh * PRICE
base = rent + MISC
print(f"runs 合计={sum(r[1] for r in ROWS)}（G1a 18 + G1b 12）")
for name,_,h,note in ROWS:
    print(f"  {name:4s} {h:8.2f}h  {note}")
print(f"raw={raw:.2f}h | 含12%缓冲={bh:.2f}h | 租卡=¥{rent:.0f} | 基准=¥{base:.0f} (硬顶¥{HARD}, 余 {HARD-base:.0f})")
E_rent = sum(h for n,_,h,_ in ROWS if n in ("E1","E3","E4b"))
print(f"测量类 E1+E3+E4b 占租卡 raw={E_rent/raw*100:.1f}% | 含 G6={ (E_rent+36)/raw*100:.1f}%")
print(f"应急金 ¥1,000 另列（不动用）；对 5,000 余 ¥{5000-base:.0f}")

def case(pm=1.0, cut=()):
    h = sum(r[2] for r in ROWS if r[0] not in cut) * BUF * pm
    return h*PRICE + MISC
C0 = ("G1a",)  # 近似：整组去重难；C0 是 6→5 seeds，删 3 runs
c0h = (18-3)*TRAIN_H[8192] + sum(r[2] for r in ROWS if r[0] not in ("G1a",))
def case_c0(pm=1.0, cut=()):
    h = ((18-3)*TRAIN_H[8192] + sum(r[2] for r in ROWS if r[0] not in ("G1a",) and r[0] not in cut)) * BUF * pm
    return h*PRICE + MISC
print("\n敏感性：")
print(f"  单价+50% 原样        ≈¥{case(1.5):.0f}")
print(f"  单价+50% +C0(6→5 seeds) ≈¥{case_c0(1.5):.0f}  (≤4600 {'OK' if case_c0(1.5)<=HARD else 'FAIL'})")
print(f"  单价+50% +C0+C1(砍G1b)  ≈¥{case_c0(1.5, ('G1b',)):.0f}")
d0 = case_c0(2.25); d1 = case_c0(2.25, ('G1b',)); d2 = case_c0(2.25, ('G1b','G6','E3'))
print(f"  双因素 原样           ≈¥{case(2.25):.0f}")
print(f"  双因素 C0            ≈¥{d0:.0f}")
print(f"  双因素 C0+C1(砍G1b)   ≈¥{d1:.0f}")
print(f"  双因素 C0+C1+C2(砍G6,E3) ≈¥{d2:.0f} (≤4900 {'OK' if d2<=4900 else 'FAIL'})")
# C3: G1a 6→4 seeds（再删 2 seeds×3 方法=6 runs）
c3 = ((18-3-6)*TRAIN_H[8192] + sum(r[2] for r in ROWS if r[0] not in ("G1a","G1b","G6","E3")))*BUF*2.25
print(f"  双因素 +C3(G1a 4 seeds) ≈¥{c3*PRICE+MISC:.0f}")
# FLOOR
fh = ((9*TRAIN_H[8192]) + 16 + 8 + 2 + 2) * BUF
print(f"\nFLOOR = {fh:.1f}h×7 + 750 = ¥{fh*PRICE+750:.0f}（9 训练 run + G6 16h + G0 8h + E1/E2 4h；MDE≈9.5pp）")
# Q* 交叉点
print("\n Q* = 4·K²·U·H·L² (U=4, L=1；括号为 U=1 标量核版本):")
for K in (4,20,100):
    for H in (3,4):
        q=4*K*K*4*H; q1=4*K*K*1*H
        print(f"   K={K:3d} H={H}: {q:>9,d}  ({q1:,d})")
# 计数自洽
print(f"\n计数：E1=16×4×300={16*4*300:,} + R 3,600 = 22,800；E5=2×50×4×2=800；G6=16×1000×6×6=576,000 列 {576_000/EVAL_TP:.0f}h")
