```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# --- 1. Mac 环境字体配置 ---
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. 绘图逻辑：手绘教科书风格的 直接II型 结构 ---
def draw_direct_form_ii(filepath):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 设置画布范围，隐藏坐标轴
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # --- 定义关键坐标 ---
    input_pos = (1, 6)
    sum1_pos = (3, 6)  # 输入侧加法器
    node_top = (5, 6)  # 中间节点 w(n)
    node_mid = (5, 4)  # 中间延时后 w(n-1)
    node_bot = (5, 2)  # 底部延时后 w(n-2)
    sum2_pos = (7, 6)  # 输出侧加法器
    output_pos = (9, 6)
    
    # --- 绘制组件函数 ---
    def draw_adder(center):
        # 画加法器圆圈
        circle = patches.Circle(center, 0.3, edgecolor='black', facecolor='white', zorder=10)
        ax.add_patch(circle)
        # 画里面的加号
        ax.text(center[0], center[1], '+', ha='center', va='center', fontsize=12, fontweight='bold')

    def draw_delay(center):
        # 画延时单元方框
        rect = patches.Rectangle((center[0]-0.4, center[1]-0.4), 0.8, 0.8, 
                                 edgecolor='black', facecolor='white', zorder=10)
        ax.add_patch(rect)
        ax.text(center[0], center[1], r'$z^{-1}$', ha='center', va='center', fontsize=10)

    def draw_arrow(start, end, label=None, label_offset=(0, 0.2)):
        # 画带箭头的直线
        ax.annotate('', xy=end, xytext=start, 
                    arrowprops=dict(arrowstyle="->", lw=1.5, color='black'))
        if label:
            mid_x = (start[0] + end[0]) / 2 + label_offset[0]
            mid_y = (start[1] + end[1]) / 2 + label_offset[1]
            ax.text(mid_x, mid_y, label, fontsize=11, color='blue', ha='center')

    # --- 绘制主干通路 (前向) ---
    # x(n) -> Sum1
    draw_arrow(input_pos, (sum1_pos[0]-0.3, sum1_pos[1]), label='x(n)')
    
    # Sum1 -> w(n)
    draw_arrow((sum1_pos[0]+0.3, sum1_pos[1]), node_top)
    ax.text(node_top[0], node_top[1]+0.3, 'w(n)', ha='center')
    
    # w(n) -> Sum2 (b0)
    draw_arrow(node_top, (sum2_pos[0]-0.3, sum2_pos[1]), label=r'$b_0$')
    
    # Sum2 -> y(n)
    draw_arrow((sum2_pos[0]+0.3, sum2_pos[1]), output_pos)
    ax.text(output_pos[0]+0.2, output_pos[1], 'y(n)', ha='left', va='center')

    # --- 绘制延时链 (垂直向下) ---
    # w(n) -> z^-1 (Top)
    draw_arrow(node_top, (5, 5)) # 连接线
    draw_delay((5, 5))
    draw_arrow((5, 4.6), node_mid) # 连接线
    
    # w(n-1) -> z^-1 (Bottom)
    draw_arrow(node_mid, (5, 3)) # 连接线
    draw_delay((5, 3))
    draw_arrow((5, 2.6), node_bot) # 连接线

    # --- 绘制反馈通路 (左侧，a系数) ---
    # w(n-1) -> Sum1 (a1)
    # 画折线: (5,4) -> (3,4) -> (3, 5.7)
    ax.plot([5, 3], [4, 4], 'k-', lw=1.5) # 水平
    draw_arrow((3, 4), (3, 5.7), label=r'$-a_1$', label_offset=(-0.3, -1)) 
    
    # w(n-2) -> Sum1 (a2)
    # 画折线: (5,2) -> (3,2) -> (3, 4) (箭头接着上面的线，或者独立画)
    # 为了清晰，我们画独立的线汇入加法器下方
    ax.plot([5, 3], [2, 2], 'k-', lw=1.5)
    draw_arrow((3, 2), (3, 5.7), label=r'$-a_2$', label_offset=(-0.3, -2))

    # --- 绘制前馈通路 (右侧，b系数) ---
    # w(n-1) -> Sum2 (b1)
    # 画折线: (5,4) -> (7,4) -> (7, 5.7)
    ax.plot([5, 7], [4, 4], 'k-', lw=1.5)
    draw_arrow((7, 4), (7, 5.7), label=r'$b_1$', label_offset=(0.3, -1))
    
    # w(n-2) -> Sum2 (b2)
    # 画折线: (5,2) -> (7,2) -> (7, 5.7)
    ax.plot([5, 7], [2, 2], 'k-', lw=1.5)
    draw_arrow((7, 2), (7, 5.7), label=r'$b_2$', label_offset=(0.3, -2))

    # --- 绘制节点和符号 ---
    # 绘制实心小圆点（节点）
    for pt in [input_pos, node_top, node_mid, node_bot, output_pos]:
        circle = patches.Circle(pt, 0.08, color='black', zorder=15)
        ax.add_patch(circle)
    
    # 绘制加法器
    draw_adder(sum1_pos)
    draw_adder(sum2_pos)

    plt.title("教科书风格：二阶IIR直接II型结构 (典范型)", fontsize=14, y=0.95)
    
    # 保存
    if os.path.exists(filepath):
        os.remove(filepath)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# --- 3. 自动保存与显示 (固定配置) ---
filename = "二阶IIR直接II型_教科书版.png"
vault_path = "/Users/heyuhang/Documents/Obsidian Vault/DSP/第六章 IIR数字滤波器的基本结构"
full_path = os.path.join(vault_path, filename)

try:
    draw_direct_form_ii(full_path)
    print(f"![[{filename}]]") 
except Exception as e:
    print(f"Error: {e}")
```