```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Mac 环境字体配置 (陈教授专用配置) ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 信号处理逻辑 ---

# 1. 设定参数 (根据题目计算结果)
fm = 1000.0           # 最高频率
fs = 2000.0           # 采样频率 (f_sam)
T = 1.0 / fs          # 采样间隔
Tp = 0.5              # 采集时长
N_actual = int(Tp*fs) # 实际采样点数 (1000)
L_dft = 4096          # DFT点数 (2的12次方)

# 2. 生成测试信号
# 为了验证分辨率，我们生成两个频率差刚好为 2Hz 的信号
# f1 = 500Hz, f2 = 502Hz
f1 = 500.0
f2 = 502.0
t = np.arange(N_actual) * T
x = np.cos(2 * np.pi * f1 * t) + np.cos(2 * np.pi * f2 * t)

# 3. 计算 DFT (FFT)
# 补零是 FFT 函数自动完成的，当第二个参数 L_dft > len(x) 时
X_k = np.fft.fft(x, L_dft)

# 4. 计算频率轴
# 频率分辨率(谱线间隔) = fs / L
df_display = fs / L_dft 
freqs = np.fft.fftfreq(L_dft, T)

# 移位方便观察
X_shift = np.fft.fftshift(X_k)
freqs_shift = np.fft.fftshift(freqs)
mag = np.abs(X_shift)
mag = mag / np.max(mag) # 归一化

# --- 3. 绘图 ---
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(freqs_shift, mag, 'b-', linewidth=1.5)
ax.set_title(f'DFT参数设计验证\n采样点N={N_actual}, DFT点数L={L_dft} (补零)', fontsize=12)
ax.set_xlabel('频率 (Hz)')
ax.set_ylabel('归一化幅度')
ax.grid(True, alpha=0.3)

# 放大观察 500Hz 附近，验证能否分辨 2Hz
ax.set_xlim(490, 512)

# 标注关键指标
ax.text(492, 0.9, f'谱线间隔 $\Delta f_d = {df_display:.4f}$ Hz', color='purple', fontweight='bold')
ax.text(492, 0.8, f'物理分辨率 $\Delta f_c = {1/Tp:.1f}$ Hz', color='green', fontweight='bold')

# 标注信号位置
ax.axvline(f1, color='r', linestyle='--', alpha=0.5)
ax.axvline(f2, color='r', linestyle='--', alpha=0.5)
ax.text(f1, 0.5, '500Hz', rotation=90, color='red')
ax.text(f2, 0.5, '502Hz', rotation=90, color='red')

plt.tight_layout()

# --- 4. 自动保存与显示 (固定配置) ---
filename = "DFT参数设计与补零验证.png"
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