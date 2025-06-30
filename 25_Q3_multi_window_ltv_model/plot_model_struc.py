import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_box(ax, xy, width, height, text, boxstyle="round,pad=0.3", facecolor="#87CEEB"):
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle=boxstyle,
        linewidth=1.5,
        edgecolor="black",
        facecolor=facecolor,
        mutation_aspect=1.0
    )
    ax.add_patch(box)
    rx, ry = xy
    cx = rx + width / 2.0
    cy = ry + height / 2.0
    ax.text(cx, cy, text, ha='center', va='center', fontsize=12, weight='bold')

def draw_arrow(ax, start, end):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->',
        connectionstyle='arc3',
        mutation_scale=15,
        linewidth=1.5,
        color='black'
    )
    ax.add_patch(arrow)

fig, ax = plt.subplots(figsize=(12, 14))
ax.set_xlim(0, 12)
ax.set_ylim(0, 16)
ax.axis('off')

# ========== 第一层：输入 ==========
draw_box(ax, (2, 14), 2, 1, 'Dense Features\n(input)')
draw_box(ax, (8, 14), 2, 1, 'Sparse Features\n(input)')

# ========== 第二层：预处理 ==========
draw_box(ax, (2, 12), 2, 1, 'Dense_Process\n(log + BN)')
draw_box(ax, (8, 12), 2, 1, 'Sparse_Process\n(Embedding)')

# ========== 第三层：拼接 ==========
draw_box(ax, (5, 10), 2, 1, 'Concatenate\n(Dense + Sparse)')

# ========== 第四层：共享底层 ==========
draw_box(ax, (3.5, 8), 5, 1, 'Shared Bottom DNN')

# ========== 第五层：多个 Head ==========
head_y = 6
for i in range(3):
    draw_box(ax, (2 + i * 3, head_y), 2, 1, f'Head_DNN_{i+1}', facecolor="#FFA07A")

# ========== 第六层：选择 ==========
draw_box(ax, (5, 3.5), 2, 1, 'Select Head Output\nby hour_idx', facecolor="#90EE90")

# ========== 第七层：最终输出 ==========
draw_box(ax, (5, 1), 2, 1, 'Final Output', facecolor="#FFD700")

# ========== 箭头 ==========
draw_arrow(ax, (3, 14), (3, 13))      # Dense -> Dense_Process
draw_arrow(ax, (9, 14), (9, 13))      # Sparse -> Sparse_Process
draw_arrow(ax, (3, 12), (6, 11))      # Dense_Process -> Concat
draw_arrow(ax, (9, 12), (6, 11))      # Sparse_Process -> Concat

draw_arrow(ax, (6, 10), (6, 9))       # Concat -> Shared Bottom

draw_arrow(ax, (6, 8), (3, 7))        # -> Head 1
draw_arrow(ax, (6, 8), (6, 7))        # -> Head 2
draw_arrow(ax, (6, 8), (9, 7))        # -> Head 3

draw_arrow(ax, (6, 6), (6, 4.5))      # Head 2 -> Select
draw_arrow(ax, (3, 6), (6, 4.5))      # Head 2 -> Select
draw_arrow(ax, (9, 6), (6, 4.5))      # Head 2 -> Select

draw_arrow(ax, (6, 3.5), (6, 2))    # Select -> Output

plt.tight_layout()
plt.show()
