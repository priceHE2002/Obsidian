```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Mac 环境字体配置 (陈教授专用配置) ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 信号处理逻辑 ---

# 参数设置
Fs = 1000.0           # 采样频率 (Hz), 对应 omega_sam = 2*pi*Fs
T = 1.0 / Fs          # 采样间隔 (s)
N = 512               # DFT 点数
L = 512               # 信号长度

# 生成时间序列
t = np.arange(L) * T

# 生成模拟信号：包含 50Hz 和 120Hz 两个分量
# x(t) = 0.7*sin(2*pi*50*t) + sin(2*pi*120*t)
f1 = 50
f2 = 120
x = 0.7 * np.sin(2 * np.pi * f1 * t) + 1.0 * np.sin(2 * np.pi * f2 * t)

# 计算 DFT (使用 FFT 算法)
Y = np.fft.fft(x, N)

# 计算双边幅度谱 P2 (对应 PPT 中的 |X[m]|)
P2 = np.abs(Y) / N  # 归一化，方便观察真实幅度

# --- 关键：手动实现 PPT 中的频率映射逻辑 ---
# 我们生成一个频率轴向量，对应 PPT 红色框中的公式
f_axis = np.zeros(N)
for i in range(N):
    if i <= N/2:
        # 正频率部分: f = (Fs / N) * i
        f_axis[i] = (Fs / N) * i
    else:
        # 负频率部分: f = (Fs / N) * (i - N)
        f_axis[i] = (Fs / N) * (i - N)

# 为了画图符合习惯（负频率在左，正频率在右），我们需要对数据进行“移位” (fftshift)
# 将数组中的负频率部分移到前面
f_shifted = np.fft.fftshift(f_axis)
P2_shifted = np.fft.fftshift(P2)

# --- 3. 绘图 ---
plt.figure(figsize=(10, 6))

# 子图1：时域波形（部分）
plt.subplot(2, 1, 1)
plt.plot(t[0:50], x[0:50], 'b.-')
plt.title(f'时域抽样信号 $x[k]$ (前50个点, $f_{{sam}}={int(Fs)}$Hz)')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.grid(True)

# 子图2：DFT 频谱分析（验证 PPT 原理）
plt.subplot(2, 1, 2)
# 使用 stem 绘制离散谱线
plt.stem(f_shifted, P2_shifted, basefmt=" ")
plt.title('利用 DFT 分析得到的真实频谱 (经移位处理)')
plt.xlabel('频率 (Hz) \n (对应 PPT 中的 $\omega$ 轴)')
plt.ylabel('幅度 $|X(j\omega)|$')
plt.grid(True)
plt.xlim([-200, 200]) # 仅展示 -200Hz 到 200Hz 范围，看清谱线
plt.text(50, 0.35, f'f1={f1}Hz', color='red', ha='center')
plt.text(120, 0.5, f'f2={f2}Hz', color='red', ha='center')
plt.text(-50, 0.35, f'-{f1}Hz', color='red', ha='center')
plt.text(-120, 0.5, f'-{f2}Hz', color='red', ha='center')

plt.tight_layout()

# --- 4. 自动保存与显示 (固定配置) ---
filename = "DFT连续信号频谱分析原理.png"
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