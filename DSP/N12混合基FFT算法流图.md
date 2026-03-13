```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- 1. Mac 环境字体配置 (陈教授专用配置) ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 绘制混合基 FFT 流图 ---
def draw_mixed_radix_flowchart():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 设置绘图区域
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, 13)
    ax.axis('off')
    
    # 标题
    ax.text(5, 12.5, "N=12 混合基FFT算法流图 (4 x 3)", fontsize=16, ha='center', weight='bold')
    
    # 定义位置
    input_x = 1
    stage1_x = 3.5
    twiddle_x = 5.5
    stage2_x = 7.5
    output_x = 9
    
    # --- 第一级：3个 4点DFT ---
    # 我们将输入视为 3组，每组4个点
    # x[n]按顺序排列，我们将其映射到 4点DFT 的输入
    
    # 绘制输入节点
    for n in range(12):
        ax.text(input_x - 0.5, 11 - n, f"x[{n}]", va='center', ha='right', fontsize=10)
        ax.plot(input_x, 11 - n, 'o', color='black', markersize=4)

    # 绘制 3个 4点DFT 盒子
    # 每一组跨度是连续的4个点 (假设输入已经按行排列)
    # Group 0: x[0], x[1], x[2], x[3] -> 4点DFT
    # Group 1: x[4], x[5], x[6], x[7] -> 4点DFT
    # Group 2: x[8], x[9], x[10], x[11] -> 4点DFT
    
    colors = ['#FFDDDD', '#DDFFDD', '#DDDDFF']
    
    for i in range(3): # 3组
        # 盒子的位置
        y_top = 11 - (i * 4) + 0.5
        y_bottom = 11 - (i * 4 + 3) - 0.5
        rect = patches.Rectangle((stage1_x - 1, y_bottom), 2, y_top - y_bottom, 
                                 linewidth=1, edgecolor='black', facecolor=colors[i], alpha=0.5)
        ax.add_patch(rect)
        ax.text(stage1_x, (y_top + y_bottom)/2, f"4点 DFT\n(组 {i+1})", ha='center', va='center', fontsize=10)
        
        # 连接线：输入 -> 4点DFT
        for j in range(4):
            idx = i * 4 + j
            ax.plot([input_x, stage1_x - 1], [11 - idx, 11 - idx], 'k-', lw=1)
            # 输出线
            ax.plot([stage1_x + 1, twiddle_x], [11 - idx, 11 - idx], 'k-', lw=1)

    # --- 旋转因子层 ---
    ax.text(twiddle_x, 12, "乘旋转因子\n$W_{12}^{nk}$", ha='center', fontsize=12, color='red')
    for n in range(12):
        ax.plot(twiddle_x, 11 - n, 'x', color='red', markersize=6)

    # --- 第二级：4个 3点DFT ---
    # 输入来源：来自第一级的输出，进行跨组连接
    # DFT 0: 取自第0, 4, 8条线
    # DFT 1: 取自第1, 5, 9条线
    # ...
    
    for k in range(4): # 4组
        # 这里的绘制逻辑比较抽象，我们画大框表示逻辑
        # 实际上3点DFT跨越了整个纵向空间，为了图示清晰，我们用不同颜色的线表示汇聚
        
        center_y = 11 - k * 3 - 1 # 只是为了排版，不完全对应物理位置
        
        # 绘制4个 3点DFT 盒子在右侧
        # 为了不让线条太乱，我们在右侧画4个盒子，并示意性连接
        box_y_center = 11 - (k * 3 + 1)
        
        # 实际的3点DFT是通过抽取不同组的同序数点组成的
        # 我们用颜色编码连接线
        line_color = ['r', 'g', 'b', 'm'][k]
        
        # 绘制右侧的3点DFT盒子
        # 位置重新排布一下以便美观：均匀分布在垂直方向
        y_pos_list = [11 - x for x in range(12)]
        
        # 定义3点DFT的输入索引 (0,4,8), (1,5,9), (2,6,10), (3,7,11)
        indices = [k, k+4, k+8]
        
        # 绘制聚合线
        # 这里画示意图：从旋转因子层汇聚到一个点
        target_y = 11 - (k * 3 + 1) # 目标中心
        
        # 绘制盒子
        rect = patches.Rectangle((stage2_x, target_y - 1.2), 1.5, 2.4, 
                                 linewidth=1, edgecolor='black', facecolor='white')
        ax.add_patch(rect)
        ax.text(stage2_x + 0.75, target_y, f"3点 DFT\n(组 {k+1})", ha='center', va='center', fontsize=9)
        
        # 画线
        for src_idx in indices:
            src_y = 11 - src_idx
            ax.plot([twiddle_x, stage2_x], [src_y, target_y + (1 - indices.index(src_idx))*0.5], 
                    color=line_color, lw=0.8, alpha=0.6)

        # 输出线
        for j in range(3):
            out_y = target_y + (1 - j)*0.5
            ax.plot([stage2_x + 1.5, output_x], [out_y, out_y], color=line_color, lw=1)
            ax.text(output_x + 0.2, out_y, f"X[{k + j*4}]", va='center', fontsize=10) # 输出索引是乱序的，这里示意

    # 添加说明
    ax.text(5, -0.5, "注：第一级进行3个4点DFT，第二级进行4个3点DFT，中间包含旋转因子乘法", 
            ha='center', fontsize=12, style='italic')

    # --- 3. 自动保存与显示 (固定配置) ---
    filename = "N12混合基FFT算法流图.png"
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

if __name__ == "__main__":
    draw_mixed_radix_flowchart()
```