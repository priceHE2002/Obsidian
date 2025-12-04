```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Mac 环境字体配置 (陈教授专用配置) ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 信号处理逻辑 ---

# A. 生成测试信号 (N=12)
N = 12
n = np.arange(N)
# 构造一个由直流、基波和高频分量组成的信号
x = 2 + 3 * np.cos(2 * np.pi * n / N) + 1 * np.sin(2 * np.pi * 3 * n / N)

# B. 算法实现：时间抽取分解为 3 组 4 点序列

# 第一步：时间抽取 (Decimation)
# x0: x[0], x[3], x[6], x[9]
# x1: x[1], x[4], x[7], x[10]
# x2: x[2], x[5], x[8], x[11]
x0 = x[0::3]
x1 = x[1::3]
x2 = x[2::3]

# 第二步：分别计算 3 个 4 点 DFT
G0 = np.fft.fft(x0) # 4点 DFT
G1 = np.fft.fft(x1) # 4点 DFT
G2 = np.fft.fft(x2) # 4点 DFT

# 第三步：基-3 合成 (Synthesis)
# X[k] = G0[k%4] + W_12^k * G1[k%4] + W_12^2k * G2[k%4]
X_custom = np.zeros(N, dtype=complex)

for k in range(N):
    # 旋转因子 W_N^nk = exp(-j * 2pi * n * k / N)
    W1 = np.exp(-1j * 2 * np.pi * k / 12)
    W2 = np.exp(-1j * 2 * np.pi * 2 * k / 12)
    
    # 利用周期性 k mod 4 获取短序列 DFT 的值
    idx = k % 4
    X_custom[k] = G0[idx] + W1 * G1[idx] + W2 * G2[idx]

# C. 对照组：直接使用 numpy 的 12 点 FFT
X_std = np.fft.fft(x)

# --- 3. 绘图与结果展示 ---
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# 子图 1: 时域序列及其分解
ax[0].stem(n, x, linefmt='b-', markerfmt='bo', basefmt='r-', label='原始序列 x[n]')
ax[0].stem(n[0::3], x0, linefmt='g--', markerfmt='go', basefmt=' ', label='第0组 (x[3r])')
ax[0].stem(n[1::3], x1, linefmt='y--', markerfmt='yo', basefmt=' ', label='第1组 (x[3r+1])')
ax[0].stem(n[2::3], x2, linefmt='m--', markerfmt='mo', basefmt=' ', label='第2组 (x[3r+2])')
ax[0].set_title(f'12点序列的时间抽取分解 (3组 $\\times$ 4点)', fontsize=14)
ax[0].set_xlabel('n')
ax[0].set_ylabel('幅度')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

# 子图 2: 频域幅度谱对比
k_axis = np.arange(N)
# --- 修正点：移除了 use_line_collection=True 参数 ---
ax[1].stem(k_axis, np.abs(X_std), linefmt='k-', markerfmt='ko', basefmt='k-', label='标准FFT结果')
# 为了显示区别，自定义算法的结果稍微错位一点点或者用红色叉号表示
ax[1].plot(k_axis, np.abs(X_custom), 'rx', markersize=10, markeredgewidth=2, label='分解合成算法结果')
ax[1].set_title('DFT幅度谱对比验证', fontsize=14)
ax[1].set_xlabel('k (频率索引)')
ax[1].set_ylabel('|X[k]|')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()

# --- 4. 自动保存与显示 (固定配置) ---
filename = "混合基FFT分解演示.png"
vault_path = "/Users/heyuhang/Documents/Obsidian Vault"
full_path = os.path.join(vault_path, filename)

try:
    if os.path.exists(full_path):
        os.remove(full_path) # 覆盖旧图
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close() # 释放内存
    print(f"![[{filename}]]") # 触发 Obsidian 显示
except Exception as e:
    print(f"Error: {e}")
```