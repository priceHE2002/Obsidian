```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Mac 环境字体配置 (陈教授专用配置) ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 信号处理逻辑 ---

# 题目参数
f1 = 100          # 频率 1 (Hz)
f2 = 120          # 频率 2 (Hz)
fs = 600          # 抽样频率 (Hz)
T = 1/fs          # 抽样间隔

# 定义绘图函数：生成信号并计算高密度频谱(DTFT近似)
# 这里的 N_sample 是题目中的 N (截断长度)
def analyze_resolution(N_sample, ax, color):
    # 生成 N 个点的时域序列 (相当于加了矩形窗)
    n = np.arange(N_sample)
    x = np.cos(2 * np.pi * f1 * n * T) + np.cos(2 * np.pi * f2 * n * T)
    
    # 为了观察频谱的连续形状(主瓣宽度)，我们做高密度补零 FFT
    # 注意：物理分辨率由 N_sample 决定，补零只是为了让曲线平滑，方便观察是否"粘"在一起
    fft_size = 2048 
    X = np.fft.fft(x, fft_size)
    freqs = np.fft.fftfreq(fft_size, T)
    
    # 移位并归一化
    X_shift = np.fft.fftshift(X)
    freqs_shift = np.fft.fftshift(freqs)
    mag = np.abs(X_shift)
    mag = mag / np.max(mag) # 归一化幅度
    
    # 计算理论分辨率
    res = fs / N_sample
    
    # 绘图
    ax.plot(freqs_shift, mag, color=color, linewidth=2)
    ax.set_title(f'样本数 N={N_sample} (分辨率 $\Delta f \\approx {res:.1f}$ Hz)', fontsize=11)
    ax.set_xlim(50, 170) # 聚焦观察 100Hz 和 120Hz 附近
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3)
    
    # 标记真实频率
    ax.axvline(f1, color='k', linestyle='--', alpha=0.3)
    ax.axvline(f2, color='k', linestyle='--', alpha=0.3)
    
    # 状态判断标注
    if res > 20:
        ax.text(110, 0.8, "无法分辨\n(单峰)", ha='center', color='red', fontweight='bold')
    elif res == 20:
        ax.text(110, 0.5, "临界分辨\n(刚好分开)", ha='center', color='orange', fontweight='bold')
    else:
        ax.text(110, 0.5, "清晰分辨\n(双峰)", ha='center', color='green', fontweight='bold')

# 创建画布
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# 情况 1: N=20 (小于30，理论上分不开)
analyze_resolution(20, ax1, 'red')

# 情况 2: N=30 (题目计算的临界值)
analyze_resolution(30, ax2, 'orange')

# 情况 3: N=60 (远大于30，清晰)
analyze_resolution(60, ax3, 'green')

plt.tight_layout()

# --- 3. 自动保存与显示 (固定配置) ---
filename = "DFT频率分辨率N值影响演示.png"
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