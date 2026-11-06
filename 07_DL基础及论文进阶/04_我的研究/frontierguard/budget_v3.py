# -*- coding: utf-8 -*-
"""v3.2 budget audit: every number printed here is the number quoted in §6."""
RENT = 7.0          # Y/GPU-h (A100 80GB, assumption; verify W4)
BUF = 1.12
MISC = 220 + 80 + 641   # electricity + storage + LLM(v3.2)
CAP = 5000

rows = [
    ("G1a N=8192 (4 methods x 8 seeds)", 32, 9.51),
    ("G1b N=2048 (4 methods x 8 seeds)", 32, 3.14),
    ("G2  ablations (9 variants x 2 seeds)", 18, 3.14),
    ("G6  CRN batch (4 methods x 16 cells x 8 seeds x 1000)", 1, 32.0),
]
print("=== rent rows (v3.2) ===")
hr = 0.0
for name, runs, h in rows:
    gh = runs * h
    hr += gh
    print(f"{name:52s} runs={runs:3d} h/run={h:5.2f} GPU-h={gh:7.1f} cost=Y{gh*RENT:7.0f}")
print(f"subtotal: GPU-h={hr:.1f} cost=Y{hr*RENT:.0f}")
buf = hr * BUF
print(f"+12%: GPU-h={buf:.1f} cost=Y{buf*RENT:.0f}")
print(f"misc: electricity 220 + storage 80 + LLM 641 = Y{MISC}")
base = buf * RENT + MISC
print(f"BASE TOTAL = Y{base:.0f}  (cap {CAP}, margin Y{CAP-base:.0f})\n")

def cost(hours, price=RENT, misc=MISC):
    return hours * BUF * price + misc

print("=== scenarios ===")
print(f"price+50% (Y10.5/h)   GPU-h={hr*BUF:7.1f} total=Y{cost(hr, RENT*1.5):7.0f}")
print(f"duration+50%          GPU-h={hr*1.5*BUF:7.1f} total=Y{cost(hr*1.5, RENT):7.0f}")
seq = [("C1 cut G2", hr-56.52), ("C2 G1a seeds 8->6", hr-56.52-76.08),
       ("C3 CRN 32->16h", hr-56.52-76.08-16.0), ("C4 G1b seeds 8->6", hr-56.52-76.08-16.0-25.12)]
for name, h in seq:
    c = cost(h, RENT*1.5)
    print(f"  {name:28s} GPU-h={h*BUF:7.1f} total=Y{c:7.0f} {'OVER' if c>CAP else 'OK'}")
print()
seq2 = [("C1 cut G2", 436.80), ("C2 G1a 8->6", 360.72), ("C3 CRN 32->16h", 344.72),
        ("C4 G1b 8->6", 319.60)]
for name, h in seq2:
    c = h * BUF * RENT * 2.25 + MISC
    print(f"dual {name:28s} GPU-h={h*1.12*2.25:7.1f} total=Y{c:7.0f} {'OVER' if c>CAP else 'OK'}")
floor_h = 15*9.51 + 15*3.14 + 8*3.14 + 8.0
print(f"FLOOR GPU-h={floor_h:.1f}: dual Y{round(floor_h*2.25*BUF*RENT+MISC)}, price+50% Y{round(floor_h*1.5*BUF*RENT+MISC)}")
print(f"FLOOR with 32h CRN: Y{(floor_h+24)*2.25*BUF*RENT+MISC:.0f}  (must NOT be used)\n")

print("=== consistency checks ===")
K, L = 1221, 50
for N, thru in [(8192, 15000), (2048, 12000)]:
    B = K*N*L
    print(f"N={N}: B={B/1e6:.1f}M  h={B/thru/3600+0.25:.2f}")
llm_tok = 840*6.5*2200
print(f"LLM: 840 eps*6.5 calls*2200 tok = {llm_tok/1e6:.1f}M tok")
usd = 0.7*llm_tok/1e6*3 + 0.3*llm_tok/1e6*12
print(f"     input 70%@$3 + output 30%@$12 = ${usd:.1f} = Y{usd*7.2:.0f}; +30% buffer = Y{usd*7.2*1.3:.0f} -> quote Y641")
print(f"local GPU-h = 13.3+68.7+30.0+20.0 = {13.3+68.7+30.0+20.0:.1f} (31 runs, Y0)")
