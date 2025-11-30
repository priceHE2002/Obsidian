```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Mac 环境字体配置 (陈教授专用配置) ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 信号处理逻辑 ---

# 设定采样频率
fs = 1000.0 # Hz
T = 1.0 / fs

# 设定信号频率为奈奎斯特频率 (fs/2)
f_nyquist = fs / 2  # 500 Hz

# 生成连续时间轴 (为了画出平滑的背景曲线)
t_cont = np.linspace(0, 0.01, 1000)
x_cont = np.cos(2 * np.pi * f_nyquist * t_cont)

# 生成离散采样点
n = np.arange(0, 11) # 采11个点
t_disc = n * T
# x[n] = cos(2*pi * (fs/2) * (n/fs)) = cos(pi * n)
x_disc = np.cos(2 * np.pi * f_nyquist * t_disc)

# --- 3. 绘图 ---
plt.figure(figsize=(10, 5))

# 绘制连续信号背景
plt.plot(t_cont, x_cont, 'b--', alpha=0.4, label='模拟信号 (500Hz)')

# 绘制采样点
plt.stem(t_disc, x_disc, linefmt='r-', markerfmt='ro', basefmt=" ", label='采样点')

plt.title(f'为何 $\Omega=\pi$ 对应奈奎斯特频率 ($f_{{sam}}={int(fs)}$Hz, $f_{{sig}}={int(f_nyquist)}$Hz)')
plt.xlabel('时间 (s)')
plt.ylabel('幅度')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper right')

# 添加标注解释
plt.text(0.002, 1.2, r'相邻采样点相位差 = $\pi$ (180°)', color='red', fontsize=12, ha='center')
plt.text(0.002, -1.3, r'序列值: $+1, -1, +1, -1 \dots$', color='red', fontsize=12, ha='center')

plt.tight_layout()

# --- 4. 自动保存与显示 (固定配置) ---
filename = "数字角频率Pi的物理意义.png"
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