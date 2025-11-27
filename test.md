```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. 环境与字体设置 (适配 Mac) ---
# Mac 上没有 SimHei，使用 Arial Unicode MS 或 Heiti TC 防止中文乱码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False # 解决负号显示问题

# --- 2. 函数定义 ---

def get_ideal_lp(wc, N):
    """
    生成理想低通滤波器的脉冲响应 (Sinc函数)
    """
    n = np.arange(-N, N + 1)
    h = np.zeros_like(n, dtype=float)
    
    # 计算 Sinc: sin(wc * n) / (pi * n)
    mask = (n != 0)
    h[mask] = np.sin(wc * n[mask]) / (np.pi * n[mask])
    
    # 处理 n=0 的极限情况
    h[n == 0] = wc / np.pi
    
    return n, h

def plot_response(ax_time, ax_freq, n, h, title):
    """辅助绘图函数：同时绘制时域(Stem)和频域(DTFT)"""
    # 时域绘制
    ax_time.stem(n, h)
    ax_time.set_title(f"{title} - 时域 h[n]")
    ax_time.grid(True, alpha=0.3)
    ax_time.set_xlabel('n')

    # 频域绘制 (FFT模拟DTFT)
    fft_size = 1024
    H = np.fft.fft(h, fft_size)
    w = np.fft.fftfreq(fft_size) * 2 * np.pi
    
    # 移频到中心
    H_shifted = np.fft.fftshift(H)
    w_shifted = np.fft.fftshift(w)
    
    ax_freq.plot(w_shifted, np.abs(H_shifted))
    ax_freq.set_title(f"{title} - 幅频响应 |H(w)|")
    ax_freq.set_xlim(-np.pi, np.pi)
    ax_freq.grid(True, alpha=0.3)
    ax_freq.set_xlabel('Frequency (rad)')

# --- 3. 主程序逻辑 ---

# 参数设置
N = 20
wc1 = 0.3 * np.pi
wc2 = 0.7 * np.pi

# 创建画布
fig, axes = plt.subplots(4, 2, figsize=(12, 16))
plt.subplots_adjust(hspace=0.6, wspace=0.3) # 调整间距防止文字重叠

# A. 低通滤波器 1 (LP1)
n, h_lp1 = get_ideal_lp(wc1, N)
plot_response(axes[0, 0], axes[0, 1], n, h_lp1, "低通滤波器 1 (LP1)")

# B. 低通滤波器 2 (LP2)
_, h_lp2 = get_ideal_lp(wc2, N)
plot_response(axes[1, 0], axes[1, 1], n, h_lp2, "低通滤波器 2 (LP2)")

# C. 带通滤波器 (BPF) = LP2 - LP1
h_bp = h_lp2 - h_lp1
plot_response(axes[2, 0], axes[2, 1], n, h_bp, "带通滤波器 (BPF)")

# D. 带阻滤波器 (BSF) = Delta - BPF
delta = np.zeros_like(n, dtype=float)
delta[n == 0] = 1.0
h_bs = delta - h_bp
plot_response(axes[3, 0], axes[3, 1], n, h_bs, "带阻滤波器 (BSF)")

# 总标题
plt.suptitle("四种理想数字滤波器的时域与频域特性", fontsize=16, y=0.92)

# --- 4. 保存并显示图片 (修正版) ---

# 1. 设置你的 Obsidian 库的绝对路径 (根据你的环境填写的)
# 注意：Obsidian Vault 中间有空格，这没问题
vault_path = "/Users/heyuhang/Documents/Obsidian Vault"
filename = "digital_filter_response.png"

# 2. 拼接出完整的保存路径
# 结果会变成: /Users/heyuhang/Documents/Obsidian Vault/digital_filter_response.png
full_path = os.path.join(vault_path, filename)

# 3. 保存图片到指定路径
# 只要路径是对的，就不会报 Read-only 错误
try:
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 打印 Obsidian 引用链接
    # 注意：Obsidian 只需要文件名就能识别，不需要完整路径
    print(f"图表已生成！")
    print(f"![[{filename}]]")
    
except Exception as e:
    print(f"保存失败，请检查路径: {e}")
```
