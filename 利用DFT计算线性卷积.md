```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Mac 环境字体配置 (陈教授专用配置) ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 信号处理逻辑 ---

# 定义原序列 (参考教材 例2-5)
x = np.array([1, 2, 3])
h = np.array([5, -2, 4])

N1 = len(x)
N2 = len(h)

# 1. 确定线性卷积后的长度 L >= N1 + N2 - 1
L = N1 + N2 - 1 
# 在实际工程中，为了利用基2-FFT的高效性，通常会取 L 为 2 的幂次，例如：
# L_fft = 2**np.ceil(np.log2(L)) 
# 这里为了演示原理，我们就取理论最小长度
L_fft = L

print(f"序列 x 长度: {N1}, 序列 h 长度: {N2}")
print(f"线性卷积所需最小长度 L: {L}")

# 2. 补零并进行 DFT (FFT)
# np.fft.fft 的第二个参数会自动进行补零操作
X_k = np.fft.fft(x, L_fft)
H_k = np.fft.fft(h, L_fft)

# 3. 频域相乘
Y_k = X_k * H_k

# 4. IDFT 得到时域结果
y_linear_by_dft = np.fft.ifft(Y_k)

# 取实部 (因为输入是实数，计算误差可能导致微小的虚部)
y_linear_by_dft = np.real(y_linear_by_dft)

# 为了对比，计算直接时域线性卷积
y_direct = np.convolve(x, h)

print(f"DFT方法计算结果: {y_linear_by_dft}")
print(f"直接卷积计算结果: {y_direct}")

# --- 3. 绘图 ---
fig, ax = plt.subplots(3, 1, figsize=(8, 10))

# 绘制输入信号 x[k]
ax[0].stem(np.arange(N1), x, basefmt=" ", label='x[k]')
ax[0].set_title('输入序列 x[k]')
ax[0].set_xlim(-1, L)
ax[0].grid(True, alpha=0.3)

# 绘制输入信号 h[k]
ax[1].stem(np.arange(N2), h, basefmt=" ", linefmt='C1-', markerfmt='C1o', label='h[k]')
ax[1].set_title('系统响应 h[k]')
ax[1].set_xlim(-1, L)
ax[1].grid(True, alpha=0.3)

# 绘制卷积结果
ax[2].stem(np.arange(L), y_linear_by_dft, basefmt=" ", linefmt='C2-', markerfmt='C2D', label='DFT计算结果')
# 叠加直接卷积的结果以示验证
ax[2].plot(np.arange(L), y_direct, 'rx', markersize=10, label='时域直接卷积')
ax[2].set_title(f'线性卷积结果 (L={L})')
ax[2].set_xlabel('k')
ax[2].set_xlim(-1, L)
ax[2].legend()
ax[2].grid(True, alpha=0.3)

plt.tight_layout()

# --- 4. 自动保存与显示 (固定配置) ---
filename = "DFT计算线性卷积演示.png" 
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