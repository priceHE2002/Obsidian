```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Mac 环境字体配置 (陈教授专用配置) ---
# 确保中文能够正常显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 信号处理逻辑 ---

# 参数设置
Fs = 1000            # 采样频率 1000 Hz
T = 1.0 / Fs         # 采样间隔
L = 1024             # 信号长度 (N点)，选取2的幂次以配合FFT原理
t = np.arange(0, L) * T  # 时间向量

# 生成合成信号
# 信号包含：50Hz 正弦波 + 120Hz 正弦波 + 随机噪声
S = 0.7 * np.sin(2 * np.pi * 50 * t) + \
    1.0 * np.sin(2 * np.pi * 120 * t)
    
# 添加高斯白噪声
X = S + 2.0 * np.random.randn(len(t)) # 噪声较大，时域几乎看不清

# --- 核心：FFT 计算 ---
Y = np.fft.fft(X)            # 计算 FFT (得到复数结果)
P2 = np.abs(Y) / L           # 计算双边频谱的幅值 (归一化)
P1 = P2[0:int(L/2)+1]        # 取单边频谱 (0 到 Fs/2)
P1[1:-1] = 2 * P1[1:-1]      # 能量补偿 (将负频率能量加到正频率)

f = Fs * np.arange(0, int(L/2)+1) / L  # 频率轴向量

# --- 绘图 ---
plt.figure(figsize=(10, 6))

# 绘制时域波形（取前50个点，否则太密）
plt.subplot(2, 1, 1)
plt.plot(1000*t[0:100], X[0:100], linewidth=1.5, color='#1f77b4')
plt.title('含噪声信号的时域波形 (局部)', fontsize=12)
plt.xlabel('时间 (ms)')
plt.ylabel('幅值')
plt.grid(True, linestyle='--', alpha=0.6)

# 绘制频域波形 (单边频谱)
plt.subplot(2, 1, 2)
plt.plot(f, P1, linewidth=1.5, color='#d62728')
plt.title('信号的单边幅值谱 (FFT分析结果)', fontsize=12)
plt.xlabel('频率 (Hz)')
plt.ylabel('幅值 |P1(f)|')
plt.grid(True, linestyle='--', alpha=0.6)

# 标注出主要频率分量
# 理论上应该在 50Hz 和 120Hz 处有峰值
plt.annotate('50Hz 分量', xy=(50, np.max(P1[40:60])), xytext=(80, np.max(P1)*0.8),
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.annotate('120Hz 分量', xy=(120, np.max(P1[110:130])), xytext=(150, np.max(P1)*0.6),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.tight_layout()

# --- 3. 自动保存与显示 (固定配置) ---
# 文件名请根据内容自动命名，务必使用中文
filename = "FFT频谱分析演示.png" 
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