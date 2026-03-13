```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Mac 环境字体配置 (陈教授专用配置) ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 信号处理逻辑 ---

# 参数
f1 = 100
f2 = 120
fs = 600
T = 1/fs
# 频率差 20Hz

def plot_spectrum(N, ax, title, color):
    # 生成信号
    n = np.arange(N)
    x = np.cos(2*np.pi*f1*n*T) + np.cos(2*np.pi*f2*n*T)
    
    # 补零做高密度 FFT 以观察主瓣形状
    fft_size = 2048
    X = np.fft.fft(x, fft_size)
    X_mag = np.abs(np.fft.fftshift(X))
    X_mag = X_mag / np.max(X_mag) # 归一化
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, T))
    
    # 绘图
    ax.plot(freqs, X_mag, color=color, linewidth=2)
    ax.set_xlim(50, 170)
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 标注两个真实频率
    ax.axvline(f1, color='k', linestyle='--', alpha=0.3)
    ax.axvline(f2, color='k', linestyle='--', alpha=0.3)

# 创建画布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 情况 1: N = 30 (上一题的标准：瑞利判据)
# 公式: N = fs / delta_f
plot_spectrum(30, ax1, 
              title='N=30 (瑞利判据)\n峰值落在零点上，中间有下凹', 
              color='orange')

# 情况 2: N = 60 (这张PPT的标准：主瓣不重叠)
# 公式: N = 2 * fs / delta_f
plot_spectrum(60, ax2, 
              title='N=60 (严格不重叠)\n两个主瓣完全分开，山脚不相连', 
              color='green')

plt.tight_layout()

# --- 3. 自动保存与显示 (固定配置) ---
filename = "DFT分辨率_瑞利判据vs严格不重叠.png"
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