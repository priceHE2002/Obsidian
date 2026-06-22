
---

### 一、基本定义 (Basic Definitions)

设有一个角度为 $\theta$ 的直角三角形，其对边 (opposite) 为 $y$，邻边 (adjacent) 为 $x$，斜边 (hypotenuse) 为 $r$。

$$
\sin \theta = \frac{y}{r} \quad (\text{正弦})
$$
$$
\cos \theta = \frac{x}{r} \quad (\text{余弦})
$$
$$
\tan \theta = \frac{y}{x} \quad (\text{正切})
$$
$$
\cot \theta = \frac{x}{y} \quad (\text{余切})
$$
$$
\sec \theta = \frac{r}{x} \quad (\text{正割})
$$
$$
\csc \theta = \frac{r}{y} \quad (\text{余割})
$$

---

### 二、基本恒等式 (Fundamental Identities)

#### 1. 倒数关系 (Reciprocal Identities)
$$
\sin \theta = \frac{1}{\csc \theta}
$$
$$
\cos \theta = \frac{1}{\sec \theta}
$$
$$
\tan \theta = \frac{1}{\cot \theta}
$$

#### 2. 商数关系 (Quotient Identities)
$$
\tan \theta = \frac{\sin \theta}{\cos \theta}
$$
$$
\cot \theta = \frac{\cos \theta}{\sin \theta}
$$

#### 3. 平方关系 (Pythagorean Identities)‼️
$$
\sin^2 \theta + \cos^2 \theta = 1
$$
$$
1 + \tan^2 \theta = \sec^2 \theta
$$
$$
1 + \cot^2 \theta = \csc^2 \theta
$$

---

### 三、诱导公式 (Reduction Formulas)

(其中 $k \in \mathbb{Z}$)

| 角度 | $\sin$ | $\cos$ | $\tan$ | $\cot$ |
| :--- | :--- | :--- | :--- | :--- |
| $-\alpha$ | $-\sin \alpha$ | $\cos \alpha$ | $-\tan \alpha$ | $-\cot \alpha$ |
| $\pi - \alpha$ | $\sin \alpha$ | $-\cos \alpha$ | $-\tan \alpha$ | $-\cot \alpha$ |
| $\pi + \alpha$ | $-\sin \alpha$ | $-\cos \alpha$ | $\tan \alpha$ | $\cot \alpha$ |
| $2\pi - \alpha$ | $-\sin \alpha$ | $\cos \alpha$ | $-\tan \alpha$ | $-\cot \alpha$ |
| $2k\pi + \alpha$ | $\sin \alpha$ | $\cos \alpha$ | $\tan \alpha$ | $\cot \alpha$ |
| $\frac{\pi}{2} - \alpha$ | $\cos \alpha$ | $\sin \alpha$ | $\cot \alpha$ | $\tan \alpha$ |
| $\frac{\pi}{2} + \alpha$ | $\cos \alpha$ | $-\sin \alpha$ | $-\cot \alpha$ | $-\tan \alpha$ |
| $\frac{3\pi}{2} - \alpha$ | $-\cos \alpha$ | $-\sin \alpha$ | $\cot \alpha$ | $\tan \alpha$ |
| $\frac{3\pi}{2} + \alpha$ | $-\cos \alpha$ | $\sin \alpha$ | $-\cot \alpha$ | $-\tan \alpha$ |

**口诀**: 奇变偶不变，符号看象限。(奇偶指 $\frac{\pi}{2}$ 的奇数倍或偶数倍，符号看象限指将 $\alpha$ 视为锐角时，原函数值的符号。)

---

### 四、和差角公式 (Sum and Difference Formulas)

$$
\sin(\alpha \pm \beta) = \sin \alpha \cos \beta \pm \cos \alpha \sin \beta
$$
$$
\cos(\alpha \pm \beta) = \cos \alpha \cos \beta \mp \sin \alpha \sin \beta
$$
$$
\tan(\alpha \pm \beta) = \frac{\tan \alpha \pm \tan \beta}{1 \mp \tan \alpha \tan \beta}
$$

---

### 五、倍角公式 (Double-Angle & Multiple-Angle Formulas)

#### 1. 二倍角公式 (Double-Angle)
$$
\sin(2\theta) = 2 \sin \theta \cos \theta
$$
$$
\cos(2\theta) = \cos^2 \theta - \sin^2 \theta = 2 \cos^2 \theta - 1 = 1 - 2 \sin^2 \theta
$$
$$
\tan(2\theta) = \frac{2 \tan \theta}{1 - \tan^2 \theta}
$$

#### 2. 三倍角公式 (Triple-Angle)
$$
\sin(3\theta) = 3 \sin \theta - 4 \sin^3 \theta
$$
$$
\cos(3\theta) = 4 \cos^3 \theta - 3 \cos \theta
$$

---

### 六、半角公式 (Half-Angle Formulas)

$$
\sin\left(\frac{\theta}{2}\right) = \pm \sqrt{\frac{1 - \cos \theta}{2}}
$$
$$
\cos\left(\frac{\theta}{2}\right) = \pm \sqrt{\frac{1 + \cos \theta}{2}}
$$
$$
\tan\left(\frac{\theta}{2}\right) = \pm \sqrt{\frac{1 - \cos \theta}{1 + \cos \theta}} = \frac{\sin \theta}{1 + \cos \theta} = \frac{1 - \cos \theta}{\sin \theta}
$$
(正负号取决于 $\frac{\theta}{2}$ 所在的象限)

---

### 七、积化和差公式 (Product-to-Sum Formulas)

$$
\sin \alpha \cos \beta = \frac{1}{2} [\sin(\alpha + \beta) + \sin(\alpha - \beta)]
$$
$$
\cos \alpha \sin \beta = \frac{1}{2} [\sin(\alpha + \beta) - \sin(\alpha - \beta)]
$$
$$
\cos \alpha \cos \beta = \frac{1}{2} [\cos(\alpha + \beta) + \cos(\alpha - \beta)]
$$
$$
\sin \alpha \sin \beta = -\frac{1}{2} [\cos(\alpha + \beta) - \cos(\alpha - \beta)]
$$

---

### 八、和差化积公式 (Sum-to-Product Formulas)

$$
\sin \alpha + \sin \beta = 2 \sin\left(\frac{\alpha + \beta}{2}\right) \cos\left(\frac{\alpha - \beta}{2}\right)
$$
$$
\sin \alpha - \sin \beta = 2 \cos\left(\frac{\alpha + \beta}{2}\right) \sin\left(\frac{\alpha - \beta}{2}\right)
$$
$$
\cos \alpha + \cos \beta = 2 \cos\left(\frac{\alpha + \beta}{2}\right) \cos\left(\frac{\alpha - \beta}{2}\right)
$$
$$
\cos \alpha - \cos \beta = -2 \sin\left(\frac{\alpha + \beta}{2}\right) \sin\left(\frac{\alpha - \beta}{2}\right)
$$

---

### 九、万能公式 (Universal Substitution Formulas)

令 $t = \tan\left(\frac{\theta}{2}\right)$, 则：
$$
\sin \theta = \frac{2t}{1+t^2}
$$
$$
\cos \theta = \frac{1-t^2}{1+t^2}
$$
$$
\tan \theta = \frac{2t}{1-t^2}
$$

---

### 十、降幂公式 (Power-Reduction Formulas)

$$
\sin^2 \theta = \frac{1 - \cos(2\theta)}{2}
$$
$$
\cos^2 \theta = \frac{1 + \cos(2\theta)}{2}
$$
$$
\tan^2 \theta = \frac{1 - \cos(2\theta)}{1 + \cos(2\theta)}
$$