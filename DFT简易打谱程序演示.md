```python
import numpy as np
import matplotlib.pyplot as plt
import os

# --- 1. Mac 环境字体配置 ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 基础数据准备 ---

# 标准音高频率表 (Hz) - 简谱 1=C
NOTE_FREQS = {
    'Rest': 0,      # 休止符
    'C4 (Do)': 261.63,
    'D4 (Re)': 293.66,
    'E4 (Mi)': 329.63,
    'F4 (Fa)': 349.23,
    'G4 (Sol)': 392.00,
    'A4 (La)': 440.00,
    'B4 (Si)': 493.88
}

def freq_to_note_name(freq):
    """查找最接近的音名"""
    if freq < 50: return "Rest" # 噪音阈值
    min_dist = float('inf')
    best_note = "?"
    for name, target_freq in NOTE_FREQS.items():
        if name == 'Rest': continue
        dist = abs(freq - target_freq)
        if dist < min_dist:
            min_dist = dist
            best_note = name
    return best_note

# --- 3. 造琴：生成“小星星”音频信号 ---
fs = 8000       # 采样率
T_note = 0.5    # 每个音符持续 0.5秒
t = np.linspace(0, T_note, int(T_note*fs), endpoint=False)

# 小星星简谱: 1 1 5 5 6 6 5
score = ['C4 (Do)', 'C4 (Do)', 'G4 (Sol)', 'G4 (Sol)', 'A4 (La)', 'A4 (La)', 'G4 (Sol)']
full_signal = np.array([])

print("--- 正在生成音频 ---")
for note in score:
    f = NOTE_FREQS[note]
    # 生成正弦波
    wave = 0.5 * np.cos(2 * np.pi * f * t)
    # 简单的包络（避免咔哒声）
    wave = wave * np.hanning(len(wave)) 
    full_signal = np.concatenate([full_signal, wave])

# --- 4. 听音：利用 DFT 进行打谱分析 ---
print("\n--- 开始 DFT 打谱分析 ---")
detected_notes = []
segment_len = len(t) # 我们已知每个音符的长度，模拟理想分帧
N_fft = 4096         # DFT点数，补零以提高栅栏密度，看得更准

# 遍历每一个切片
num_segments = len(full_signal) // segment_len
freq_history = []

for i in range(num_segments):
    # 1. 截取一段信号 (对应 PPT 中的 m1 = m(1:N))
    segment = full_signal[i*segment_len : (i+1)*segment_len]
    
    # 2. 计算 DFT (FFT)
    X = np.fft.fft(segment, N_fft)
    
    # 3. 寻找幅度最大的谱线位置 k (只看正频率部分)
    mag = np.abs(X[:N_fft//2])
    k_max = np.argmax(mag)
    
    # 4. 换算为物理频率 (利用核心公式)
    f_detected = k_max * fs / N_fft
    freq_history.append(f_detected)
    
    # 5. 匹配音名
    note_name = freq_to_note_name(f_detected)
    detected_notes.append(note_name)
    
    print(f"第 {i+1} 个音: 频率={f_detected:.1f} Hz --> 识别为: {note_name}")

print(f"\n最终识别乐谱: {detected_notes}")

# --- 5. 可视化 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 时域波形
time_axis = np.arange(len(full_signal)) / fs
ax1.plot(time_axis, full_signal)
ax1.set_title('“小星星”时域波形')
ax1.set_xlabel('时间 (s)')
ax1.set_ylabel('幅度')
ax1.grid(True, alpha=0.3)

# 识别结果（频域分析）
ax2.stem(range(1, num_segments+1), freq_history, basefmt=" ", linefmt='b-', markerfmt='bo')
ax2.set_title('基于 DFT 的音高识别结果')
ax2.set_xlabel('音符序号')
ax2.set_ylabel('识别到的频率 (Hz)')
ax2.set_yticks(list(NOTE_FREQS.values())[1:]) # 设置Y轴刻度为标准音高
ax2.set_yticklabels(list(NOTE_FREQS.keys())[1:])
ax2.grid(True, alpha=0.3)

for i, txt in enumerate(detected_notes):
    ax2.text(i+1, freq_history[i]+10, txt, ha='center', fontsize=9, color='red')

plt.tight_layout()

# --- 6. 保存图片 ---
filename = "DFT简易打谱程序演示.png"
vault_path = "/Users/heyuhang/Documents/Obsidian Vault"
full_path = os.path.join(vault_path, filename)

try:
    if os.path.exists(full_path):
        os.remove(full_path)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"![[{filename}]]")
except Exception as e:
    print(f"Error: {e}")
```