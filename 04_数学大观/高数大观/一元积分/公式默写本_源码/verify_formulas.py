# -*- coding: utf-8 -*-
"""数值验证《一元积分大观》核心公式区的每一条结论。"""
import sympy as sp

x, a, t, n, p, q, s, th = sp.symbols('x a t n p q s theta', positive=True)
X = sp.Symbol('x', real=True)

ok, bad = [], []

def chk(label, lhs, rhs, **subs):
    """数值比较 lhs 与 rhs（lhs 为积分，rhs 为答案）"""
    try:
        d = sp.simplify(lhs - rhs)
        vals = {}
        for k, v in subs.items():
            vals[sp.Symbol(k, real=True)] = v
        num = complex(d.subs(vals).evalf()) if d.free_symbols else complex(d)
        if abs(num) < 1e-9:
            ok.append(label)
        else:
            bad.append((label, num))
    except Exception as e:
        bad.append((label, f'ERR {e}'))

# ---------- 不定积分表（对答案求导，与被积函数比较） ----------
def dchk(label, f, F, **subs):
    chk(label, sp.diff(F, X), f, **subs)

dchk('∫x^α = x^(α+1)/(α+1)',       X**sp.Rational(5,2), X**sp.Rational(7,2)/sp.Rational(7,2))
dchk('∫1/x = ln|x|',                1/X, sp.log(sp.Abs(X)), x=2)
dchk('∫e^x',                        sp.exp(X), sp.exp(X))
dchk('∫a^x = a^x/ln a',             sp.Rational(3,2)**X, sp.Rational(3,2)**X/sp.log(sp.Rational(3,2)))
dchk('∫ln x = x ln x - x',          sp.log(X), X*sp.log(X)-X, x=2)
dchk('∫1/(1+x^2) = arctan x',       1/(1+X**2), sp.atan(X))
dchk('∫1/√(1-x^2) = arcsin x',      1/sp.sqrt(1-X**2), sp.asin(X), x=sp.Rational(1,2))
dchk('∫sin = -cos',                 sp.sin(X), -sp.cos(X))
dchk('∫cos = sin',                  sp.cos(X), sp.sin(X))
dchk('∫tan = -ln|cos|',             sp.tan(X), -sp.log(sp.Abs(sp.cos(X))), x=1)
dchk('∫cot = ln|sin|',              1/sp.tan(X), sp.log(sp.Abs(sp.sin(X))), x=1)
dchk('∫sec = ln|sec+tan|',          1/sp.cos(X), sp.log(sp.Abs(1/sp.cos(X)+sp.tan(X))), x=1)
dchk('∫csc = ln|csc-cot|',          1/sp.sin(X), sp.log(sp.Abs(1/sp.sin(X)-1/sp.tan(X))), x=1)
dchk('∫sec² = tan',                 1/sp.cos(X)**2, sp.tan(X), x=1)
dchk('∫csc² = -cot',                1/sp.sin(X)**2, -1/sp.tan(X), x=1)
dchk('∫sec tan = sec',              sp.tan(X)/sp.cos(X), 1/sp.cos(X), x=1)
dchk('∫csc cot = -csc',             sp.cos(X)/sp.sin(X)**2, -1/sp.sin(X), x=1)

a0 = sp.Rational(3,2)
dchk('∫1/(x²+a²)', 1/(X**2+a0**2), sp.atan(X/a0)/a0)
dchk('∫1/√(a²-x²)', 1/sp.sqrt(a0**2-X**2), sp.asin(X/a0), x=1)
dchk('∫1/(x²-a²)', 1/(X**2-a0**2), sp.log(sp.Abs((X-a0)/(X+a0)))/(2*a0), x=3)
dchk('∫1/√(x²+a²)', 1/sp.sqrt(X**2+a0**2), sp.log(X+sp.sqrt(X**2+a0**2)))
dchk('∫1/√(x²-a²)', 1/sp.sqrt(X**2-a0**2), sp.log(X+sp.sqrt(X**2-a0**2)), x=3)
dchk('∫√(a²-x²)', sp.sqrt(a0**2-X**2),
     X*sp.sqrt(a0**2-X**2)/2 + a0**2/2*sp.asin(X/a0), x=1)
dchk('∫√(x²+a²)', sp.sqrt(X**2+a0**2),
     X*sp.sqrt(X**2+a0**2)/2 + a0**2/2*sp.log(X+sp.sqrt(X**2+a0**2)), x=1)
dchk('∫√(x²-a²)', sp.sqrt(X**2-a0**2),
     X*sp.sqrt(X**2-a0**2)/2 - a0**2/2*sp.log(X+sp.sqrt(X**2-a0**2)), x=3)
dchk('∫sinh = cosh', sp.sinh(X), sp.cosh(X))
dchk('∫cosh = sinh', sp.cosh(X), sp.sinh(X))

# ---------- 定积分 ----------
chk('∫₀^∞ x^n e^-x = n!', sp.integrate(x**4*sp.exp(-x), (x, 0, sp.oo)), sp.factorial(4))
chk('∫₀^∞ x^n e^-ax = n!/a^(n+1)',
    sp.integrate(x**3*sp.exp(-2*x), (x, 0, sp.oo)), sp.factorial(3)/2**4)
chk('∫₀^∞ e^-ax = 1/a', sp.integrate(sp.exp(-3*x), (x, 0, sp.oo)), sp.Rational(1,3))
chk('∫₀^∞ x e^-ax = 1/a²', sp.integrate(x*sp.exp(-3*x), (x, 0, sp.oo)), sp.Rational(1,9))
chk('∫₀^∞ x² e^-ax = 2/a³', sp.integrate(x**2*sp.exp(-3*x), (x, 0, sp.oo)), sp.Rational(2,27))
chk('∫₀^∞ x e^-x² = 1/2', sp.integrate(x*sp.exp(-x**2), (x, 0, sp.oo)), sp.Rational(1,2))
chk('∫₀^∞ x³ e^-x² = 1/2', sp.integrate(x**3*sp.exp(-x**2), (x, 0, sp.oo)), sp.Rational(1,2))

chk('泊松 ∫₀^∞ e^-x² = √π/2', sp.integrate(sp.exp(-x**2), (x, 0, sp.oo)), sp.sqrt(sp.pi)/2)
chk('∫_{-∞}^∞ e^-x² = √π', sp.integrate(sp.exp(-x**2), (x, -sp.oo, sp.oo)), sp.sqrt(sp.pi))
chk('∫_{-∞}^∞ e^-x²/2 = √(2π)',
    sp.integrate(sp.exp(-x**2/2), (x, -sp.oo, sp.oo)), sp.sqrt(2*sp.pi))
chk('∫₀^∞ e^-ax² = ½√(π/a)',
    sp.integrate(sp.exp(-3*x**2), (x, 0, sp.oo)), sp.sqrt(sp.pi/3)/2)

chk('Γ(1)=1', sp.gamma(1), 1)
chk('Γ(1/2)=√π', sp.gamma(sp.Rational(1,2)), sp.sqrt(sp.pi))
chk('Γ(n+1)=n!', sp.gamma(6), sp.factorial(5))
chk('Γ(s+1)=sΓ(s)', sp.gamma(sp.Rational(7,2)), sp.Rational(5,2)*sp.gamma(sp.Rational(5,2)))
chk('∫₀^∞ √x e^-x = √π/2',
    sp.integrate(sp.sqrt(x)*sp.exp(-x), (x, 0, sp.oo)), sp.sqrt(sp.pi)/2)
chk('∫₀^∞ x^-1/2 e^-x = √π',
    sp.integrate(x**sp.Rational(-1,2)*sp.exp(-x), (x, 0, sp.oo)), sp.sqrt(sp.pi))

chk('B(1,1)=1', sp.beta(1,1), 1)
chk('B(1/2,1/2)=π', sp.beta(sp.Rational(1,2), sp.Rational(1,2)), sp.pi)
chk('B(p,q)=Γ(p)Γ(q)/Γ(p+q)', sp.beta(3,4), sp.gamma(3)*sp.gamma(4)/sp.gamma(7))
chk('B 换元形式', sp.beta(3,4),
    sp.integrate(t**(3-1)/(1+t)**(3+4), (t, 0, sp.oo)))
chk('∫₀^1 dx/√(x(1-x)) = π',
    sp.integrate(1/sp.sqrt(x*(1-x)), (x, 0, 1)), sp.pi)
chk('∫₀^{π/2} sin^{2p-1}cos^{2q-1} = ½B(p,q)',
    sp.integrate(sp.sin(th)**(2*2-1)*sp.cos(th)**(2*3-1), (th, 0, sp.pi/2)),
    sp.beta(2,3)/2)

# ---------- Wallis ----------
for nn, val in [(2, sp.pi/4), (3, sp.Rational(2,3)), (4, 3*sp.pi/16), (6, 5*sp.pi/32)]:
    chk(f'∫₀^(π/2) sin^{nn} = {val}',
        sp.integrate(sp.sin(x)**nn, (x, 0, sp.pi/2)), val)
chk('∫₀^π cos^6 = 5π/16', sp.integrate(sp.cos(x)**6, (x, 0, sp.pi)), 5*sp.pi/16)
chk('∫₀^{2π} sin^6 = 5π/8', sp.integrate(sp.sin(x)**6, (x, 0, 2*sp.pi)), 5*sp.pi/8)
chk('Wallis 递推 I_n=(n-1)/n I_{n-2}',
    sp.integrate(sp.sin(x)**8, (x, 0, sp.pi/2)),
    sp.Rational(7,8)*sp.integrate(sp.sin(x)**6, (x, 0, sp.pi/2)))
# 双阶乘形式
for nn in [2,3,4,5,6,7]:
    k = nn//2
    if nn % 2 == 0:
        val = sp.factorial2(nn-1)/sp.factorial2(nn)*sp.pi/2
    else:
        val = sp.factorial2(nn-1)/sp.factorial2(nn)
    chk(f'Wallis 双阶乘 n={nn}', sp.integrate(sp.sin(x)**nn, (x, 0, sp.pi/2)), val)

# ---------- 区间再现推论 ----------
chk('推论A ∫₀^{π/2} f(sin)=∫₀^{π/2} f(cos)',
    sp.integrate(sp.sin(x)**4/(1+sp.sin(x)**2), (x, 0, sp.pi/2)),
    sp.integrate(sp.cos(x)**4/(1+sp.cos(x)**2), (x, 0, sp.pi/2)))
chk('推论B ∫₀^π x f(sin) = π/2 ∫₀^π f(sin)',
    sp.integrate(x*sp.sin(x)**2, (x, 0, sp.pi)),
    sp.pi/2*sp.integrate(sp.sin(x)**2, (x, 0, sp.pi)))
chk('∫₀^π x sin²x = π²/4', sp.integrate(x*sp.sin(x)**2, (x, 0, sp.pi)), sp.pi**2/4)
chk('∫₋a^a √(a²-x²) = πa²/2',
    sp.integrate(sp.sqrt(9-x**2), (x, -3, 3)), 9*sp.pi/2)
chk('∫₀^a √(a²-x²) = πa²/4',
    sp.integrate(sp.sqrt(9-x**2), (x, 0, 3)), 9*sp.pi/4)
chk('周期 ∫_a^{a+T} = ∫₀^T',
    sp.integrate(sp.sin(x)**2, (x, sp.Rational(7,3), sp.Rational(7,3)+sp.pi)),
    sp.integrate(sp.sin(x)**2, (x, 0, sp.pi)))

print(f'通过 {len(ok)} 条')
if bad:
    print('!! 未通过 / 出错：')
    for lbl, v in bad:
        print('   -', lbl, v)
else:
    print('全部通过 ✓')
