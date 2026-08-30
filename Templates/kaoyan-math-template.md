# 考研数学笔记模板

> 推荐：正文解释尽量简洁，重要结论独立成行；复杂推导统一用 `aligned`。

---

## 1. 行内公式

函数：

`$f(x)=x^2+1$`

极限：

`$\lim_{x\to0}f(x)=A$`

简单分数：

`$\frac{1}{1+x^2}$`

如果行内公式太小，可强制教材式大小：

`$\displaystyle \frac{1}{1+x^2}$`

或只放大分数：

`$\dfrac{1}{1+x^2}$`

---

## 2. 行间公式

```latex
$$
\int_a^b f(x)\,\mathrm{d}x
$$
```

效果：

$$
\int_a^b f(x)\,\mathrm{d}x
$$

---

## 3. 多步推导：aligned

```latex
$$
\begin{aligned}
f(x)
&= x^2+2x+1\\
&=(x+1)^2\\
&\ge 0
\end{aligned}
$$
```

推荐所有考试推导尽量用 `&=` 对齐等号。

---

## 4. 分段函数：cases

```latex
$$
f(x)=
\begin{cases}
x^2, & x\ge 0,\\
-x, & x<0.
\end{cases}
$$
```

---

# 高等数学常用模板

## 5. 极限

```latex
$$
\lim_{x\to x_0} f(x)=A
$$
```

左右极限：

```latex
$$
\lim_{x\to x_0^-}f(x)
\qquad
\lim_{x\to x_0^+}f(x)
$$
```

无穷远：

```latex
$$
\lim_{x\to+\infty}f(x)=A
$$
```

---

## 6. 导数

```latex
$$
f'(x)
=
\lim_{\Delta x\to0}
\frac{f(x+\Delta x)-f(x)}{\Delta x}
$$
```

高阶导数：

```latex
$$
f^{(n)}(x)
$$
```

在某点：

```latex
$$
f^{(n)}(x_0)
$$
```

---

## 7. 微分

```latex
$$
\mathrm{d}y=f'(x)\,\mathrm{d}x
$$
```

建议把微分符号 `d` 写成正体：

```latex
\mathrm{d}x
```

而不是：

```latex
dx
```

---

## 8. 不定积分

```latex
$$
\int f(x)\,\mathrm{d}x
=
F(x)+C
$$
```

---

## 9. 定积分

```latex
$$
\int_a^b f(x)\,\mathrm{d}x
$$
```

牛顿—莱布尼茨公式：

```latex
$$
\int_a^b f(x)\,\mathrm{d}x
=
F(b)-F(a)
$$
```

---

## 10. 换元积分

```latex
$$
\begin{aligned}
u&=\varphi(x),\\
\mathrm{d}u&=\varphi'(x)\,\mathrm{d}x
\end{aligned}
$$
```

---

## 11. 分部积分

```latex
$$
\int u\,\mathrm{d}v
=
uv-\int v\,\mathrm{d}u
$$
```

---

## 12. 泰勒公式

Maclaurin：

```latex
$$
f(x)
=
f(0)
+f'(0)x
+\frac{f''(0)}{2!}x^2
+\cdots
+\frac{f^{(n)}(0)}{n!}x^n
+o(x^n)
$$
```

一般点：

```latex
$$
f(x)
=
\sum_{k=0}^{n}
\frac{f^{(k)}(x_0)}{k!}
(x-x_0)^k
+
o\!\left((x-x_0)^n\right)
$$
```

---

## 13. 常见等价无穷小

```latex
$$
x\to0,\qquad
\sin x\sim x,\qquad
\tan x\sim x,\qquad
1-\cos x\sim\frac{x^2}{2}
$$
```

---

## 14. 求和

```latex
$$
\sum_{k=1}^{n} a_k
$$
```

---

## 15. 二重积分

```latex
$$
\iint_D f(x,y)\,\mathrm{d}\sigma
$$
```

若写成直角坐标：

```latex
$$
\iint_D f(x,y)\,\mathrm{d}x\,\mathrm{d}y
$$
```

---

# 线性代数常用模板

## 16. 圆括号矩阵：pmatrix

```latex
$$
A=
\begin{pmatrix}
a_{11} & a_{12}\\
a_{21} & a_{22}
\end{pmatrix}
$$
```

---

## 17. 方括号矩阵：bmatrix

```latex
$$
A=
\begin{bmatrix}
1 & 2\\
3 & 4
\end{bmatrix}
$$
```

---

## 18. 行列式：vmatrix

```latex
$$
|A|
=
\begin{vmatrix}
a & b\\
c & d
\end{vmatrix}
$$
```

---

## 19. 增广矩阵

```latex
$$
\left(
\begin{array}{ccc|c}
a_{11} & a_{12} & a_{13} & b_1\\
a_{21} & a_{22} & a_{23} & b_2
\end{array}
\right)
$$
```

---

## 20. 线性方程组

```latex
$$
\begin{cases}
a_{11}x_1+a_{12}x_2=b_1,\\
a_{21}x_1+a_{22}x_2=b_2.
\end{cases}
$$
```

---

## 21. 特征值

```latex
$$
|A-\lambda E|=0
$$
```

或：

```latex
$$
\det(A-\lambda I)=0
$$
```

---

## 22. 特征向量

```latex
$$
(A-\lambda I)x=0
$$
```

---

## 23. 相似对角化

```latex
$$
P^{-1}AP=\Lambda
$$
```

其中：

```latex
$$
\Lambda=
\operatorname{diag}
(\lambda_1,\lambda_2,\ldots,\lambda_n)
$$
```

---

## 24. 二次型

```latex
$$
f(x_1,\ldots,x_n)
=
x^{\mathrm T}Ax
$$
```

---

# 推荐的规范写法

## 25. 三角函数

推荐：

```latex
\sin x
\cos x
\tan x
\cot x
\sec x
\csc x
```

不要直接写：

```latex
sin x
cos x
tan x
```

---

## 26. 对数与指数

```latex
\ln x
\log_a x
e^x
```

---

## 27. 自动伸缩括号

```latex
$$
\left(
\frac{x+1}{x-1}
\right)^2
$$
```

---

## 28. 间距

常用数学间距：

```latex
\,
\quad
\qquad
```

例如：

```latex
$$
x\to0,\qquad \sin x\sim x
$$
```

积分中：

```latex
$$
\int f(x)\,\mathrm{d}x
$$
```

---

# Obsidian Callout 推荐

## 29. 定理

```markdown
> [!theorem] 定理
> 若……
> 则……
```

## 30. 结论

```markdown
> [!conclusion] 结论
> 这里写需要记忆的最终结论。
```

## 31. 易错点

```markdown
> [!mistake] 易错点
> 注意条件、定义域、端点、正负号。
```

---

# 一道题的推荐完整结构

```markdown
## 题目

求：
$$
\int \frac{1}{1+x^2}\,\mathrm{d}x
$$

## 思路

识别基本积分公式。

## 解

$$
\begin{aligned}
\int \frac{1}{1+x^2}\,\mathrm{d}x
&=\arctan x+C
\end{aligned}
$$

> [!conclusion] 结论
> $$
> \int \frac{1}{1+x^2}\,\mathrm{d}x
> =
> \arctan x+C
> $$
```