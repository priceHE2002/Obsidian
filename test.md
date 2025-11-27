```python
import numpy as np
import matplotlib.pyplot as plt

# --- 1. 环境与字体设置 (适配 Mac) ---
# Mac 上没有 SimHei，使用 Arial Unicode MS 或 Heiti TC 来显示中文
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# --- 2. 函数定义 ---

def get_ideal_lp(wc, N):
    """
    生成理想低通滤波器的脉冲响应 (Sinc函数)
    参数:
        wc: 截止频率 (rad)
        N:  滤波器阶数参数 (总长度为 2N+1)
    返回:
        n: 时域序列索引
        h: 脉冲响应序列
    """
    # 生成从 -N 到 N 的序列
    n = np.arange(-N, N + 1)
    
    # 初始化 h 数组
    h = np.zeros_like(n, dtype=float)
    
    # 计算 Sinc 函数: sin(wc * n) / (pi * n)
    # 针对 n != 0 的情况
    mask = (n != 0)
    h[mask] = np.sin(wc * n[mask]) / (np.pi * n[mask])
    
    # 针对 n = 0 的情况 (利用极限 wc/pi)
    h[n == 0] = wc / np.pi
    
    return n, h

def plot_response(ax_time, ax_freq, n, h, title):
    """辅助绘图函数：同时绘制时域和频域"""
    # 1. 时域绘制 (Stem plot)
    ax_time.stem(n, h)
    ax_time.set_title(f"{title} - 时域 h[n]")
    ax_time.grid(True, alpha=0.3)
    ax_time.set_xlabel('n')

    # 2. 频域绘制 (DTFT幅度谱)
    # 使用 FFT 计算频谱，补零到 1024 点使曲线平滑
    fft_size = 1024
    H = np.fft.fft(h, fft_size)
    w = np.fft.fftfreq(fft_size) * 2 * np.pi # 频率轴转换到 -pi 到 pi
    
    # 移动零频分量到中心
    H_shifted = np.fft.fftshift(H)
    w_shifted = np.fft.fftshift(w)
    
    ax_freq.plot(w_shifted, np.abs(H_shifted))
    ax_freq.set_title(f"{title} - 幅频响应 |H(w)|")
    ax_freq.set_xlim(-np.pi, np.pi)
    ax_freq.grid(True, alpha=0.3)
    ax_freq.set_xlabel('Frequency (rad)')

# --- 3. 主程序逻辑 ---

# 参数设置
N = 20          # 滤波器长度参数
wc1 = 0.3 * np.pi  # 低通1 截止频率
wc2 = 0.7 * np.pi  # 低通2 截止频率 (用于构建带通)

# 创建画布 (4行 x 2列)
fig, axes = plt.subplots(4, 2, figsize=(12, 16))
plt.subplots_adjust(hspace=0.6) # 调整子图间距

# === A. 低通滤波器 1 (Low Pass 1) ===
n, h_lp1 = get_ideal_lp(wc1, N)
plot_response(axes[0, 0], axes[0, 1], n, h_lp1, "低通滤波器 1 (LP1)")

# === B. 低通滤波器 2 (Low Pass 2) ===
_, h_lp2 = get_ideal_lp(wc2, N)
plot_response(axes[1, 0], axes[1, 1], n, h_lp2, "低通滤波器 2 (LP2)")

# === C. 带通滤波器 (Band Pass) ===
# 原理：利用频谱相减 (高截止 LP - 低截止 LP)
h_bp = h_lp2 - h_lp1
plot_response(axes[2, 0], axes[2, 1], n, h_bp, "带通滤波器 (BPF)")

# === D. 带阻滤波器 (Band Stop) ===
# 原理：全通(Delta) - 带通
# 构造 Delta 函数 (单位脉冲)
delta = np.zeros_like(n, dtype=float)
delta[n == 0] = 1.0

h_bs = delta - h_bp
plot_response(axes[3, 0], axes[3, 1], n, h_bs, "带阻滤波器 (BSF)")

# 显示总标题和图像
plt.suptitle("四种理想数字滤波器的时域与频域特性 (截断效应演示)", fontsize=16, y=0.95)
plt.show()
```
