# -*- coding: utf-8 -*-
"""由《一元积分大观》核心公式区生成：默写本 + 答案册（LaTeX -> PDF）。"""
import os

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 内容数据
# 每条: (pre, left, ans, lines)  lines=0 => 同行点线留白; n>0 => 另起 n 行点线
SECTIONS = [
("一、不定积分基本公式表", [
    ("1.1　幂 · 指数 · 对数", [
        ("", r"\int x^{\alpha}\,\mathrm{d}x=", r"\dfrac{x^{\alpha+1}}{\alpha+1}+C\quad(\alpha\neq-1)", 0),
        ("", r"\int \dfrac{\mathrm{d}x}{x}=", r"\ln\lvert x\rvert+C", 0),
        ("", r"\int e^{x}\,\mathrm{d}x=", r"e^{x}+C", 0),
        ("", r"\int a^{x}\,\mathrm{d}x=", r"\dfrac{a^{x}}{\ln a}+C", 0),
        ("", r"\int \ln x\,\mathrm{d}x=", r"x\ln x-x+C", 0),
        ("", r"\int \dfrac{\mathrm{d}x}{1+x^{2}}=", r"\arctan x+C", 0),
        ("", r"\int \dfrac{\mathrm{d}x}{\sqrt{1-x^{2}}}=", r"\arcsin x+C", 0),
    ]),
    ("1.2　三角函数", [
        ("", r"\int \sin x\,\mathrm{d}x=", r"-\cos x+C", 0),
        ("", r"\int \cos x\,\mathrm{d}x=", r"\sin x+C", 0),
        ("", r"\int \tan x\,\mathrm{d}x=", r"-\ln\lvert\cos x\rvert+C", 0),
        ("", r"\int \cot x\,\mathrm{d}x=", r"\ln\lvert\sin x\rvert+C", 0),
        ("", r"\int \sec x\,\mathrm{d}x=", r"\ln\lvert\sec x+\tan x\rvert+C", 0),
        ("", r"\int \csc x\,\mathrm{d}x=", r"\ln\lvert\csc x-\cot x\rvert+C", 0),
        ("", r"\int \sec^{2}x\,\mathrm{d}x=", r"\tan x+C", 0),
        ("", r"\int \csc^{2}x\,\mathrm{d}x=", r"-\cot x+C", 0),
        ("", r"\int \sec x\tan x\,\mathrm{d}x=", r"\sec x+C", 0),
        ("", r"\int \csc x\cot x\,\mathrm{d}x=", r"-\csc x+C", 0),
    ]),
    ("1.3　含参数推广", [
        ("", r"\int \dfrac{\mathrm{d}x}{x^{2}+a^{2}}=", r"\dfrac{1}{a}\arctan\dfrac{x}{a}+C", 0),
        ("", r"\int \dfrac{\mathrm{d}x}{\sqrt{a^{2}-x^{2}}}=", r"\arcsin\dfrac{x}{a}+C", 0),
        ("", r"\int \dfrac{\mathrm{d}x}{x^{2}-a^{2}}=", r"\dfrac{1}{2a}\ln\left\lvert\dfrac{x-a}{x+a}\right\rvert+C", 0),
        ("", r"\int \dfrac{\mathrm{d}x}{\sqrt{x^{2}+a^{2}}}=", r"\ln\left\lvert x+\sqrt{x^{2}+a^{2}}\right\rvert+C", 0),
        ("", r"\int \dfrac{\mathrm{d}x}{\sqrt{x^{2}-a^{2}}}=", r"\ln\left\lvert x+\sqrt{x^{2}-a^{2}}\right\rvert+C", 0),
        ("", r"\int \sqrt{a^{2}-x^{2}}\,\mathrm{d}x=",
             r"\dfrac{x}{2}\sqrt{a^{2}-x^{2}}+\dfrac{a^{2}}{2}\arcsin\dfrac{x}{a}+C", 0),
        ("", r"\int \sqrt{x^{2}+a^{2}}\,\mathrm{d}x=",
             r"\dfrac{x}{2}\sqrt{x^{2}+a^{2}}+\dfrac{a^{2}}{2}\ln\left\lvert x+\sqrt{x^{2}+a^{2}}\right\rvert+C", 0),
        ("", r"\int \sqrt{x^{2}-a^{2}}\,\mathrm{d}x=",
             r"\dfrac{x}{2}\sqrt{x^{2}-a^{2}}-\dfrac{a^{2}}{2}\ln\left\lvert x+\sqrt{x^{2}-a^{2}}\right\rvert+C", 0),
        ("", r"\int \sinh x\,\mathrm{d}x=", r"\cosh x+C", 0),
        ("", r"\int \cosh x\,\mathrm{d}x=", r"\sinh x+C", 0),
    ]),
]),
("二、定积分核心结论", [
    ("2.1　对称 · 分解 · 周期 · 平移", [
        ("", r"\int_{-a}^{a}f(x)\,\mathrm{d}x=",
             r"\begin{cases}2\displaystyle\int_{0}^{a}f(x)\,\mathrm{d}x, & f\ \text{为偶函数}\\[7pt] 0, & f\ \text{为奇函数}\end{cases}", 2),
        ("偶 + 奇分解：", r"f(x)=",
             r"\tfrac{1}{2}\left[f(x)+f(-x)\right]+\tfrac{1}{2}\left[f(x)-f(-x)\right]", 1),
        ("", r"\int_{a}^{a+T}f(x)\,\mathrm{d}x=", r"\int_{0}^{T}f(x)\,\mathrm{d}x\quad(T\ \text{为周期})", 0),
        ("", r"[\,a,b\,]\ \xrightarrow{\ x=\frac{a+b}{2}+t\ }",
             r"\left[-\dfrac{b-a}{2},\ \dfrac{b-a}{2}\right]", 0),
    ]),
    ("2.2　区间再现", [
        ("", r"\int_{a}^{b}f(x)\,\mathrm{d}x\ \xrightarrow{\ x=a+b-t\ }", r"\int_{a}^{b}f(a+b-t)\,\mathrm{d}t", 0),
        ("推论 A：", r"\int_{0}^{\pi/2}f(\sin x)\,\mathrm{d}x=", r"\int_{0}^{\pi/2}f(\cos x)\,\mathrm{d}x", 0),
        ("推论 B：", r"\int_{0}^{\pi}xf(\sin x)\,\mathrm{d}x=", r"\dfrac{\pi}{2}\int_{0}^{\pi}f(\sin x)\,\mathrm{d}x", 0),
        ("推论 B 应用：", r"\int_{0}^{\pi}x\sin^{2}x\,\mathrm{d}x=", r"\dfrac{\pi^{2}}{4}", 0),
    ]),
    ("2.3　Wallis 公式", [
        ("", r"\int_{0}^{\pi/2}\sin^{n}x\,\mathrm{d}x=\int_{0}^{\pi/2}\cos^{n}x\,\mathrm{d}x=",
             r"\begin{cases}\dfrac{(2k-1)!!}{(2k)!!}\cdot\dfrac{\pi}{2}, & n=2k\ \text{（偶次，带 }\frac{\pi}{2}\text{）}\\[8pt] \dfrac{(2k)!!}{(2k+1)!!}, & n=2k+1\ \text{（奇次）}\end{cases}", 2),
        ("递推：", r"I_{n}=", r"\dfrac{n-1}{n}I_{n-2}\qquad\left(I_{0}=\dfrac{\pi}{2},\ I_{1}=1\right)", 0),
        ("", r"\int_{0}^{\pi/2}\sin^{2}x\,\mathrm{d}x=", r"\dfrac{\pi}{4}", 0),
        ("", r"\int_{0}^{\pi/2}\sin^{3}x\,\mathrm{d}x=", r"\dfrac{2}{3}", 0),
        ("", r"\int_{0}^{\pi/2}\sin^{4}x\,\mathrm{d}x=", r"\dfrac{3\pi}{16}", 0),
        ("", r"\int_{0}^{\pi/2}\sin^{6}x\,\mathrm{d}x=", r"\dfrac{5\pi}{32}", 0),
        ("", r"\int_{0}^{\pi}\cos^{6}\theta\,\mathrm{d}\theta=", r"\dfrac{5\pi}{16}", 0),
        ("", r"\int_{0}^{2\pi}\sin^{6}x\,\mathrm{d}x=", r"\dfrac{5\pi}{8}", 0),
    ]),
    ("2.4　几何意义 · 面积", [
        ("", r"\int_{-a}^{a}\sqrt{a^{2}-x^{2}}\,\mathrm{d}x=", r"\dfrac{\pi a^{2}}{2}\quad\text{（上半圆）}", 0),
        ("", r"\int_{0}^{a}\sqrt{a^{2}-x^{2}}\,\mathrm{d}x=", r"\dfrac{\pi a^{2}}{4}\quad\text{（四分之一圆）}", 0),
        ("两曲线间的面积：", r"S=", r"\int_{a}^{b}\lvert f(x)-g(x)\rvert\,\mathrm{d}x", 0),
    ]),
]),
("三、特殊反常积分", [
    ("3.1　指数型（结果是阶乘）", [
        ("", r"\int_{0}^{\infty}x^{n}e^{-x}\,\mathrm{d}x=", r"n!", 0),
        ("", r"\int_{0}^{\infty}x^{n}e^{-ax}\,\mathrm{d}x=", r"\dfrac{n!}{a^{n+1}}\quad(a>0)", 0),
        ("", r"\int_{0}^{\infty}e^{-ax}\,\mathrm{d}x=", r"\dfrac{1}{a}\quad(a>0)", 0),
        ("", r"\int_{0}^{\infty}xe^{-ax}\,\mathrm{d}x=", r"\dfrac{1}{a^{2}}", 0),
        ("", r"\int_{0}^{\infty}x^{2}e^{-ax}\,\mathrm{d}x=", r"\dfrac{2}{a^{3}}", 0),
        ("", r"\int_{0}^{\infty}xe^{-x^{2}}\,\mathrm{d}x=", r"\dfrac{1}{2}", 0),
        ("", r"\int_{0}^{\infty}x^{3}e^{-x^{2}}\,\mathrm{d}x=", r"\dfrac{1}{2}", 0),
    ]),
    ("3.2　泊松积分", [
        ("", r"\int_{0}^{\infty}e^{-x^{2}}\,\mathrm{d}x=", r"\dfrac{\sqrt{\pi}}{2}", 0),
        ("", r"\int_{-\infty}^{+\infty}e^{-x^{2}}\,\mathrm{d}x=", r"\sqrt{\pi}", 0),
        ("", r"\int_{-\infty}^{+\infty}e^{-x^{2}/2}\,\mathrm{d}x=", r"\sqrt{2\pi}", 0),
        ("", r"\int_{0}^{\infty}e^{-ax^{2}}\,\mathrm{d}x=", r"\dfrac{1}{2}\sqrt{\dfrac{\pi}{a}}\quad(a>0)", 0),
    ]),
    ("3.3　伽马函数 $\Gamma$", [
        ("定义：", r"\Gamma(s)=", r"\int_{0}^{\infty}x^{s-1}e^{-x}\,\mathrm{d}x\quad(s>0)", 1),
        ("递推：", r"\Gamma(s+1)=", r"s\,\Gamma(s)", 0),
        ("", r"\Gamma(1)=", r"1", 0),
        ("", r"\Gamma\!\left(\dfrac{1}{2}\right)=", r"\sqrt{\pi}", 0),
        ("", r"\Gamma(n+1)=", r"n!\quad(n\ \text{为非负整数})", 0),
        ("", r"\int_{0}^{\infty}\sqrt{x}\,e^{-x}\,\mathrm{d}x=", r"\Gamma\!\left(\dfrac{3}{2}\right)=\dfrac{\sqrt{\pi}}{2}", 0),
        ("", r"\int_{0}^{\infty}x^{-1/2}e^{-x}\,\mathrm{d}x=", r"\Gamma\!\left(\dfrac{1}{2}\right)=\sqrt{\pi}", 0),
    ]),
    ("3.4　贝塔函数 $B$", [
        ("定义：", r"B(p,q)=", r"\int_{0}^{1}x^{p-1}(1-x)^{q-1}\,\mathrm{d}x\quad(p>0,\ q>0)", 1),
        ("对称性：", r"B(p,q)=", r"B(q,p)", 0),
        ("与 $\Gamma$ 的关系：", r"B(p,q)=", r"\dfrac{\Gamma(p)\Gamma(q)}{\Gamma(p+q)}", 0),
        ("", r"B(1,1)=", r"1", 0),
        ("", r"B\!\left(\dfrac{1}{2},\dfrac{1}{2}\right)=", r"\pi", 0),
        ("", r"B(p,q)\ \xrightarrow{\ x=\frac{t}{1+t}\ }", r"\int_{0}^{\infty}\dfrac{t^{\,p-1}}{(1+t)^{\,p+q}}\,\mathrm{d}t", 1),
        ("", r"\int_{0}^{1}\dfrac{\mathrm{d}x}{\sqrt{x(1-x)}}=", r"B\!\left(\dfrac{1}{2},\dfrac{1}{2}\right)=\pi", 0),
        ("", r"\int_{0}^{\pi/2}\sin^{2p-1}\theta\cos^{2q-1}\theta\,\mathrm{d}\theta=",
             r"\dfrac{1}{2}B(p,q)\quad\left(\text{令 }x=\sin^{2}\theta\right)", 0),
    ]),
]),
]

def flatten():
    items, n = [], 0
    for sec, subs in SECTIONS:
        items.append(("sec", sec))
        for sub, its in subs:
            items.append(("sub", sub))
            for pre, left, ans, lines in its:
                n += 1
                items.append(("item", n, pre, left, ans, lines))
    return items, n

ITEMS, TOTAL = flatten()

# ---------------------------------------------------------------- 公共导言
PREAMBLE = r"""
\documentclass[11pt,a4paper]{article}
\usepackage[left=13mm,right=13mm,top=15mm,bottom=13mm,includeheadfoot]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{multicol}
\usepackage{xeCJK}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{array}

\setCJKmainfont{Noto Serif CJK SC}
\setCJKsansfont{Noto Sans CJK SC}
\setlength{\parindent}{0pt}

\definecolor{secc}{RGB}{22,40,72}     % 章标题
\definecolor{subc}{RGB}{58,88,128}    % 节标题
\definecolor{ansc}{RGB}{10,86,140}    % 答案
\definecolor{leftc}{RGB}{74,84,100}   % 答案册中的题面（灰）
\definecolor{mute}{RGB}{132,132,132}  % 编号
\definecolor{dotc}{RGB}{178,178,178}  % 点线
\definecolor{ruleg}{RGB}{120,140,170}

\newcounter{fcnt}
\makeatletter
% 点导线：用 1fill（高于 \parfillskip 的 fil 阶），保证铺满整栏
\newcommand{\dtf}{\leaders\hbox{$\m@th \mkern \@dotsep mu.\mkern \@dotsep mu$}\hskip 0pt plus 1fill\relax}
\makeatother
\newcommand{\dotlines}[1]{%
  \ifnum#1>0 \noindent{\color{dotc}\dtf}\par\vspace{0.95em}\fi
  \ifnum#1>1 \noindent{\color{dotc}\dtf}\par\vspace{0.95em}\fi
  \ifnum#1>2 \noindent{\color{dotc}\dtf}\par\vspace{0.95em}\fi
  \ifnum#1>3 \noindent{\color{dotc}\dtf}\par\vspace{0.95em}\fi
}
\newcommand{\ans}[1]{{\color{ansc}#1}}

\newcommand{\Fsec}[1]{%
  \par\vspace{0.85em}\nobreak
  \noindent{\sffamily\bfseries\large\color{secc}#1}\par
  \vspace{0.18em}\noindent{\color{ruleg}\rule{\linewidth}{0.9pt}}\par
  \vspace{0.55em}\nobreak}
\newcommand{\Fsub}[1]{%
  \par\vspace{0.45em}\nobreak
  \noindent{\sffamily\bfseries\color{subc}#1}\par\vspace{0.35em}\nobreak}
"""

# ---------------------------------------------------------------- 默写本
def build_workbook():
    body = []
    for it in ITEMS:
        if it[0] == "sec":
            body.append(r"\Fsec{%s}" % it[1])
        elif it[0] == "sub":
            body.append(r"\Fsub{%s}" % it[1])
        else:
            _, n, pre, left, ans, lines = it
            body.append(r"\WBitem{%s}{%s}{%d}" % (pre, left, lines))
    doc = PREAMBLE + r"""
\newcommand{\WBitem}[3]{%
  \stepcounter{fcnt}%
  \par\noindent\hangindent=2.0em\hangafter=1
  \makebox[2.0em][l]{\sffamily\footnotesize\color{mute}\thefcnt.}%
  #1$\displaystyle #2$\ %
  \ifnum#3=0 {\color{dotc}\dtf}\par\vspace{1.9em}%
  \else\par\vspace{0.1em}\dotlines{#3}\fi}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\sffamily\color{mute}一元积分 · 考研核心公式默写本}
\fancyhead[R]{\small\sffamily\color{mute}第 \thepage\ 页}
\renewcommand{\headrulewidth}{0.4pt}\renewcommand{\footrulewidth}{0pt}
\begin{document}
{\sffamily\bfseries\LARGE\color{secc}一元积分 · 考研核心公式默写本}\par
\vspace{0.35em}
{\small\color{mute}来源《一元积分大观》核心公式区 · 共 \textbf{@@N@@} 条公式}\par
\vspace{0.6em}
{\small 姓名\underline{\hspace{2.1cm}}\quad 日期\underline{\hspace{2.1cm}}\quad
 用时\underline{\hspace{1.5cm}}\quad 正确率\underline{\hspace{1.5cm}}}\par
\vspace{0.4em}
{\small\color{mute}用法：先默写、后对照《答案册》。做错的在编号上画圈，次日只默错项。}\par
\vspace{0.5em}
{\color{ruleg}\rule{\linewidth}{1.1pt}}\par
\vspace{0.9em}
\begin{multicols}{2}
""" .replace("@@N@@", str(TOTAL)) + "\n".join(body) + r"""
\end{multicols}
\vspace{0.6em}
{\color{ruleg}\rule{\linewidth}{0.7pt}}\par
\vspace{0.7em}
{\sffamily\bfseries\color{secc}错题记录 · 反复默错的那几条}\par
\vspace{0.4em}
\dotlines{4}
\end{document}
"""
    open(os.path.join(OUT, "workbook.tex"), "w").write(doc)

# ---------------------------------------------------------------- 答案册
def build_answers():
    body = []
    for it in ITEMS:
        if it[0] == "sec":
            body.append(r"\Fsec{%s}" % it[1])
        elif it[0] == "sub":
            body.append(r"\Fsub{%s}" % it[1])
        else:
            _, n, pre, left, ans, lines = it
            body.append(r"\ANitem{%s}{%s}{%s}" % (pre, left, ans))
    doc = PREAMBLE + r"""
\newcommand{\ANitem}[3]{%
  \stepcounter{fcnt}%
  \par\noindent\hangindent=2.0em\hangafter=1
  \makebox[2.0em][l]{\sffamily\footnotesize\color{mute}\thefcnt.}%
  #1$\displaystyle {\color{leftc}#2}\,\ans{#3}$\par\vspace{0.5em}}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\sffamily\color{mute}一元积分 · 考研核心公式默写本 · 答案册}
\fancyhead[R]{\small\sffamily\color{mute}第 \thepage\ 页}
\renewcommand{\headrulewidth}{0.4pt}\renewcommand{\footrulewidth}{0pt}
\begin{document}
{\sffamily\bfseries\LARGE\color{secc}一元积分 · 考研核心公式默写本 · 答案册}\par
\vspace{0.35em}
{\small\color{mute}编号与默写本一一对应 · 共 \textbf{@@N@@} 条 ·
\textcolor{ansc}{蓝色}为待默写部分}\par
\vspace{0.5em}
{\color{ruleg}\rule{\linewidth}{1.1pt}}\par
\vspace{0.8em}
""" .replace("@@N@@", str(TOTAL)) + "\n".join(body) + r"""
\end{document}
"""
    open(os.path.join(OUT, "answers.tex"), "w").write(doc)

build_workbook()
build_answers()
print("total items:", TOTAL)
