import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

df = pd.read_csv('daily_HKO_RF_ALL.csv', na_values=["*** 沒有數據/unavailable"], header=None)
df.columns = ['year', 'month', 'day', 'value', 'flag']
print('列名:', df.columns.tolist())
print(df.head())
df = df.dropna(subset=['year', 'month', 'day', 'value'])
df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
df = df.dropna(subset=['date'])
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df = df[['date', 'value']].sort_values('date').reset_index(drop=True)
df = df.tail(100)  # 只取最后100个数据


n_init = 3  # 初始边长

# 预先计算所有点的坐标
x, y = [0], [0]
angle = 0
n = n_init
for i in range(len(df)):
    angle += 91
    rad = np.deg2rad(angle)
    new_x = x[-1] + n * np.cos(rad)
    new_y = y[-1] + n * np.sin(rad)
    x.append(new_x)
    y.append(new_y)
    n += 3

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.axis('off')
lines = []

def init():
    ax.clear()
    ax.set_aspect('equal')
    ax.axis('off')
    return lines

def update(frame):
    ax.clear()
    ax.set_aspect('equal')
    ax.axis('off')
    n = n_init
    angle = 0
    x_, y_ = [0], [0]
    # 计算所有折线段和色块的总数
    seg_num = 200  # 每条折线分成更多小线段，表现更细腻
    total_blocks = len(df) * seg_num
    # 当前帧要显示的色块数
    # 正向/反向循环
    cycle_len = total_blocks
    if frame < cycle_len:
        num_blocks = min(frame + 1, total_blocks)
    else:
        num_blocks = max(total_blocks - (frame - cycle_len + 1), 0)
    block_idx = 0
    for i in range(len(df)):
        color_start, color_end = color_table[i]
        angle += 91
        rad = np.deg2rad(angle)
        new_x = x_[-1] + n * np.cos(rad)
        new_y = y_[-1] + n * np.sin(rad)
        linewidth = 24 + 8 * df['value'].iloc[i] / max(df['value'])
        for seg in range(seg_num):
            if block_idx >= num_blocks:
                break
            t0 = seg / seg_num
            t1 = (seg + 1) / seg_num
            px0 = x_[-1] + t0 * (new_x - x_[-1])
            py0 = y_[-1] + t0 * (new_y - y_[-1])
            px1 = x_[-1] + t1 * (new_x - x_[-1])
            py1 = y_[-1] + t1 * (new_y - y_[-1])
            color = color_start * (1 - t0) + color_end * t0
            # 每个小线段宽度与该折线的数据值成比例
            min_w, max_w = 20, 120
            v = df['value'].iloc[i]
            vmin, vmax = df['value'].min(), df['value'].max()
            width = min_w + (max_w - min_w) * (v - vmin) / (vmax - vmin + 1e-6)
            ax.plot([px0, px1], [py0, py1], color=color, linewidth=width, alpha=0.8)
            block_idx += 1
        x_.append(new_x)
        y_.append(new_y)
        n += 3
        if block_idx >= num_blocks:
            break
    return []


seg_num = 200  # 每条折线分成更多小线段，表现更细腻
total_blocks = len(df) * seg_num

# 颜色表生成函数
def gen_color_table():
    color_table = []
    import colorsys
    # 先生成第一个起点色
    hue = np.random.rand()
    s = 0.08 + 0.14 * np.random.rand()
    v1 = 0.5 + 0.3 * np.random.rand()
    color_start = np.array(colorsys.hsv_to_rgb(hue, s, v1))
    for i in range(len(df)):
        if i == 0:
            this_start = color_start
        else:
            this_start = color_table[-1][1]
        hue = np.random.rand()
        s = 0.08 + 0.14 * np.random.rand()  # 低饱和度
        v2 = 0.85 + 0.13 * np.random.rand()  # 明度高一些，范围0.85~0.98
        this_end = np.array(colorsys.hsv_to_rgb(hue, s, v2))
        color_table.append((this_start, this_end))
    return color_table

color_table = gen_color_table()

def update(frame):
    ax.clear()
    ax.set_aspect('equal')
    ax.axis('off')
    n = n_init
    angle = 0
    x_, y_ = [0], [0]
    # 正向/反向循环
    cycle_len = total_blocks
    if frame < cycle_len:
        num_blocks = min(frame + 1, total_blocks)
    else:
        num_blocks = max(total_blocks - (frame - cycle_len + 1), 0)
    block_idx = 0
    for i in range(len(df)):
        color_start, color_end = color_table[i]
        angle += 91
        rad = np.deg2rad(angle)
        new_x = x_[-1] + n * np.cos(rad)
        new_y = y_[-1] + n * np.sin(rad)
        for seg in range(seg_num):
            if block_idx >= num_blocks:
                break
            t0 = seg / seg_num
            t1 = (seg + 1) / seg_num
            px0 = x_[-1] + t0 * (new_x - x_[-1])
            py0 = y_[-1] + t0 * (new_y - y_[-1])
            px1 = x_[-1] + t1 * (new_x - x_[-1])
            py1 = y_[-1] + t1 * (new_y - y_[-1])
            color = color_start * (1 - t0) + color_end * t0
            rand_width = np.random.uniform(12, 60)
            ax.plot([px0, px1], [py0, py1], color=color, linewidth=rand_width, alpha=0.8)
            block_idx += 1
        x_.append(new_x)
        y_.append(new_y)
        n += 3
        if block_idx >= num_blocks:
            break
    return []

def on_repeat(*args):
    global color_table
    color_table = gen_color_table()


# 让动画帧数等于色块总数，循环时每组重新生成颜色
seg_num = 20
total_blocks = len(df) * seg_num

# 动画帧数为正向+反向

# 记录当前帧数
last_cycle = {'cycle': -1}
def update_with_color_refresh(frame):
    cycle_len = 2 * total_blocks
    cycle = frame // cycle_len
    if cycle != last_cycle['cycle']:
        on_repeat()
        last_cycle['cycle'] = cycle
    f = frame % cycle_len
    if f < total_blocks:
        return update(f)
    else:
        return update(f - total_blocks)

from itertools import count
# 全局存储每个小线段的宽度
rand_width_table = None
def refresh_rand_width_table():
    global rand_width_table
    rand_width_table = [np.random.uniform(20, 120, size=seg_num) for _ in range(len(df))]
refresh_rand_width_table()

def update(frame):
    ax.clear()
    ax.set_aspect('equal')
    ax.axis('off')
    n = n_init
    angle = 0
    x_, y_ = [0], [0]
    cycle_len = total_blocks
    if frame < cycle_len:
        num_blocks = min(frame + 1, total_blocks)
    else:
        num_blocks = max(total_blocks - (frame - cycle_len + 1), 0)
    block_idx = 0
    # 每隔5帧刷新一次宽度表
    if frame % 5 == 0:
        refresh_rand_width_table()
    for i in range(len(df)):
        color_start, color_end = color_table[i]
        angle += 91
        rad = np.deg2rad(angle)
        new_x = x_[-1] + n * np.cos(rad)
        new_y = y_[-1] + n * np.sin(rad)
        for seg in range(seg_num):
            if block_idx >= num_blocks:
                break
            t0 = seg / seg_num
            t1 = (seg + 1) / seg_num
            px0 = x_[-1] + t0 * (new_x - x_[-1])
            py0 = y_[-1] + t0 * (new_y - y_[-1])
            px1 = x_[-1] + t1 * (new_x - x_[-1])
            py1 = y_[-1] + t1 * (new_y - y_[-1])
            color = color_start * (1 - t0) + color_end * t0
            rand_width = rand_width_table[i][seg]
            ax.plot([px0, px1], [py0, py1], color=color, linewidth=rand_width, alpha=0.8)
            block_idx += 1
        x_.append(new_x)
        y_.append(new_y)
        n += 3
        if block_idx >= num_blocks:
            break
    return []

ani = animation.FuncAnimation(
    fig, update_with_color_refresh, frames=count(0), init_func=init, blit=True, interval=30, repeat=True
)

plt.show()
# ani.save('spiral_rainfall.gif', writer='imagemagick')
# ani.save('spiral_rainfall.mp4', writer='ffmpeg')
# plt.show()  # <--- 加上这一行
# 生成颜色表和透明度相位表
def get_color_table():
    pass  # TODO: 实现或删除此函数体