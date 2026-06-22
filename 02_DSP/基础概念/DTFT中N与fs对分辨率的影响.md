```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Mac 环境字体配置 (陈教授专用配置) ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 信号处理逻辑 ---

# 模拟一个由两个靠得很近的频率组成的信号
f1 = 50  # Hz
f2 = 55  # Hz (间隔 5Hz)

def plot_dtft(N, fs, ax, color, title):
    # 1. 生成时域采样序列
    T = 1/fs
    n = np.arange(N)
    x = np.cos(2*np.pi*f1*n*T) + np.cos(2*np.pi*f2*n*T)
    
    # 2. 计算 DTFT (使用高密度补零 FFT 模拟连续曲线)
    # 补零到 4096 点，是为了让曲线看起来光滑连续，模拟 DTFT 的连续性质
    # 注意：物理分辨率只由原始 N 和 fs 决定，补零不改变分辨率
    fft_size = 4096
    X = np.fft.fft(x, fft_size)
    X_mag = np.abs(np.fft.fftshift(X))
    X_mag = X_mag / np.max(X_mag) # 归一化幅度
    
    # 3. 映射到物理频率轴
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, T))
    
    # 4. 计算观测时长和理论分辨率
    Tp = N / fs
    res = 1 / Tp
    
    # 绘图
    ax.plot(freqs, X_mag, color=color, linewidth=2)
    ax.set_title(f'{title}\n$N={N}, f_{{sam}}={fs}$Hz $\\rightarrow$ 时长 $T_p={Tp:.2f}$s', fontsize=11)
    ax.set_xlabel('物理频率 (Hz)')
    ax.set_ylabel('归一化幅度')
    ax.set_xlim(30, 75) # 聚焦观察 50Hz 和 55Hz 附近
    ax.grid(True, alpha=0.3)
    
    # 标注真实频率
    ax.axvline(f1, color='k', linestyle='--', alpha=0.3)
    ax.axvline(f2, color='k', linestyle='--', alpha=0.3)
    
    # 标注分辨率状态
    if res > abs(f2-f1):
        ax.text(65, 0.5, "无法分辨\n(主瓣太宽)", color='red', ha='center')
    else:
        ax.text(65, 0.5, "清晰分辨\n(双峰)", color='green', ha='center')

# 创建画布
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# 情况 1: 基准 (N=100, fs=200) -> Tp = 0.5s -> Res = 2Hz (足够分辨 5Hz)
plot_dtft(100, 200, ax1, 'blue', '基准情况')

# 情况 2: 减少 N (N=20, fs=200) -> Tp = 0.1s -> Res = 10Hz (无法分辨 5Hz)
plot_dtft(20, 200, ax2, 'red', '减少采样点 N (分辨率下降)')

# 情况 3: 增加 fs (N=100, fs=1000) -> Tp = 0.1s -> Res = 10Hz (无法分辨 5Hz)
# 注意！虽然 N 和基准一样，但因为 fs 变大了，导致总观测时间变短了，分辨率反而下降了！
plot_dtft(100, 1000, ax3, 'orange', '增加采样率 $f_{sam}$ (点数不变)\n(导致时长变短，分辨率下降！)')

plt.tight_layout()

# --- 3. 自动保存与显示 (固定配置) ---
filename = "DTFT中N与fs对分辨率的影响.png"
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