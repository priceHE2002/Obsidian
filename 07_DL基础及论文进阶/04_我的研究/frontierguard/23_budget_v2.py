# -*- coding: utf-8 -*-
"""23_synthesis_v2 预算复算（单价 A100 ¥7/h，已含 12% 重跑缓冲）。"""
ROWS=[("E0",0.0,0),("E1",7.0,56),("E2",2.0,16),("E3",4.0,31),("E4",1.0,8),
      ("E5",2.0,16),("G0",20.0,157),("G1a",285.3,2237),("G1b",75.4,591),("G6",30.0,235)]
PRICE=7.0; BUF=1.12; MISC=120+80+641; TARGET=5000
h=sum(r[1] for r in ROWS); y=sum(r[2] for r in ROWS)
hb=h*BUF
print(f"runs 合计=54 | 训练/评测 GPU-h={h:.1f} | 含 12% 缓冲={hb:.1f} | 行取整 ¥={y} | 计算 ¥={hb*PRICE:.0f}")
print(f"基准 = {y} + 杂项 {MISC} = ¥{y+MISC} | 对 {TARGET} 余 ¥{TARGET-(y+MISC)} | 对 4200 余 ¥{4200-(y+MISC)}")
def close(price_mult=1.0, hour_mult=1.0, cut=None):
    cut=cut or []
    hh=sum(r[1] for r in ROWS if r[0] not in cut)*BUF*hour_mult
    yy=sum(r[2] for r in ROWS if r[0] not in cut)*hour_mult
    rent=hh*PRICE*price_mult
    return rent+MISC, hh
for label,pm,hm,cut in [
    ("单价+50% 原样",1.5,1.0,[]),
    ("单价+50% C1 砍G1b",1.5,1.0,["G1b"]),
    ("双因素 原样",2.25,1.0,[]),
    ("双因素 C1",2.25,1.0,["G1b"]),
    ("双因素 C2 +砍G6,E组",2.25,1.0,["G1b","G6","E1","E2","E3","E4","E5"]),
    ("双因素 C3 +G1a 6→4 seeds(-10runs,-95.1h)",2.25,1.0,["G1b","G6","E1","E2","E3","E4","E5"]),
]:
    if "C3" in label:
        hh=(sum(r[1] for r in ROWS)-75.4-30.0-16.0-95.1)*BUF*hm
        rent=hh*PRICE*pm
        print(label, f"≈¥{rent+MISC:.0f}  (含缓冲 {hh:.1f}h)", "<=5000 OK" if rent+MISC<=5000 else ">5000")
    else:
        c,hh=close(pm,hm,cut); print(label, f"≈¥{c:.0f} ({'<=5000 OK' if c<=5000 else '>5000'})")
# FLOOR
floor_h=(114.1+24+32)*BUF; print(f"FLOOR = {floor_h:.1f}h×7 + 750 = ¥{floor_h*7+750:.0f} (12 runs, MDE≈9.5pp)")
