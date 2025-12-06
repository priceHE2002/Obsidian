```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Mac 环境字体配置 ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 信号处理逻辑：从 H_L(s) 到 H(z) ---

# 已知参数
# s_k * T 的值
s1T = -0.5080 + 0.5080j
s2T = -0.5080 - 0.5080j

# 留数 A_k
A1 = -0.5080j
A2 = 0.5080j

# 1. 映射极点到 z 域: p_k = exp(s_k * T)
p1 = np.exp(s1T)
p2 = np.exp(s2T)

print(f"z域极点 p1: {p1:.4f}")
print(f"z域极点 p2: {p2:.4f}")

# 2. 计算分母系数 (1 - p1*z^-1)(1 - p2*z^-1)
# = 1 - (p1 + p2)z^-1 + (p1 * p2)z^-2
# 系数分别是 1, -(p1+p2), p1*p2
den_a1 = -(p1 + p2).real
den_a2 = (p1 * p2).real

# 3. 计算分子系数
# H(z) = A1/(1 - p1*z^-1) + A2/(1 - p2*z^-1)
# 通分后分子 = A1(1 - p2*z^-1) + A2(1 - p1*z^-1)
#           = (A1 + A2) - (A1*p2 + A2*p1)z^-1
num_b0 = (A1 + A2).real
num_b1 = -(A1 * p2 + A2 * p1).real

# --- 3. 结果展示与验证 ---
print("\n【最终 H(z) 系数验证】")
print(f"分子 b0 (z^0):  {num_b0:.4f} (理论应为0)")
print(f"分子 b1 (z^-1): {num_b1:.4f} (图中为 0.3104)")
print("-" * 20)
print(f"分母 a0 (z^0):  1.0000")
print(f"分母 a1 (z^-1): {den_a1:.4f} (图中为 -1.0514)")
print(f"分母 a2 (z^-2): {den_a2:.4f} (图中为 0.3620)")

# 简单的文本图，用于 Obsidian 显示验证结果
filename = "Hz系数计算验证.png"
vault_path = "/Users/heyuhang/Documents/Obsidian Vault/DSP/第四章 IIR"
full_path = os.path.join(vault_path, filename)

try:
    plt.figure(figsize=(6, 4))
    plt.text(0.1, 0.8, "H(z) 系数计算结果：", fontsize=14, fontweight='bold')
    plt.text(0.1, 0.6, f"分子 b1 (z^-1): {num_b1:.4f}", fontsize=12, color='blue')
    plt.text(0.1, 0.4, f"分母 a1 (z^-1): {den_a1:.4f}", fontsize=12, color='red')
    plt.text(0.1, 0.2, f"分母 a2 (z^-2): {den_a2:.4f}", fontsize=12, color='red')
    plt.axis('off')
    
    if os.path.exists(full_path):
        os.remove(full_path)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"![[{filename}]]")

except Exception as e:
    print(f"Error: {e}")
```