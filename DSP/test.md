```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Mac 环境字体配置 (陈教授专用配置) ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 信号处理逻辑 ---
# 定义频率轴，范围取 -2pi 到 2pi 以展示周期性
omega = np.linspace(-2 * np.pi, 2 * np.pi, 1000)

# 定义原频谱函数 X(e^jΩ)
# 利用周期性：计算角度相位映射到 [-pi, pi]，判断是否在 [-0.4pi, 0.4pi] 内
def get_X(w):
    # 将频率 wrap 到 [-pi, pi] 区间
    w_wrapped = np.angle(np.exp(1j * w))
    # 定义矩形窗：宽度为 0.8pi (即 -0.4pi 到 0.4pi)
    magnitude = np.where(np.abs(w_wrapped) <= 0.4 * np.pi, 1.0, 0.0)
    return magnitude

# 计算原频谱
X_mag = get_X(omega)

# 计算调制后的频谱 Y(e^jΩ)
# Y = 0.5 * [X(w - 0.8pi) + X(w + 0.8pi)]
shift = 0.8 * np.pi
Y_mag = 0.5 * (get_X(omega - shift) + get_X(omega + shift))

# --- 3. 绘图逻辑 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.5)

# 绘制 X(e^jΩ)
ax1.plot(omega, X_mag, 'b-', linewidth=2)
ax1.set_title(r'原信号频谱 $X(e^{j\Omega})$', fontsize=14)
ax1.set_ylabel('幅度', fontsize=12)
ax1.set_xlabel(r'数字频率 $\Omega$ (rad)', fontsize=12)
ax1.set_ylim(0, 1.5)
# 设置关键刻度
ticks = [-2*np.pi, -0.4*np.pi, 0, 0.4*np.pi, 2*np.pi]
tick_labels = [r'$-2\pi$', r'$-0.4\pi$', r'$0$', r'$0.4\pi$', r'$2\pi$']
ax1.set_xticks(ticks)
ax1.set_xticklabels(tick_labels)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.fill_between(omega, 0, X_mag, color='blue', alpha=0.1)

# 绘制 Y(e^jΩ)
ax2.plot(omega, Y_mag, 'r-', linewidth=2)
ax2.set_title(r'调制后频谱 $Y(e^{j\Omega}) = \frac{1}{2}[X(e^{j(\Omega-0.8\pi)}) + X(e^{j(\Omega+0.8\pi)})]$', fontsize=14)
ax2.set_ylabel('幅度', fontsize=12)
ax2.set_xlabel(r'数字频率 $\Omega$ (rad)', fontsize=12)
ax2.set_ylim(0, 1.5)
# 设置关键刻度：展示频移后的中心点和边界
ticks_y = [-1.2*np.pi, -0.8*np.pi, -0.4*np.pi, 0, 0.4*np.pi, 0.8*np.pi, 1.2*np.pi]
tick_labels_y = [r'$-1.2\pi$', r'$-0.8\pi$', r'$-0.4\pi$', r'$0$', r'$0.4\pi$', r'$0.8\pi$', r'$1.2\pi$']
ax2.set_xticks(ticks_y)
ax2.set_xticklabels(tick_labels_y)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.fill_between(omega, 0, Y_mag, color='red', alpha=0.1)

# --- 4. 自动保存与显示 (固定配置) ---
filename = "exercise_1_11_spectrum.png" 
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
