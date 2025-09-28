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
df = df.tail(30)  # 只取最后30个数据


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
            ax.plot([px0, px1], [py0, py1], color=color, linewidth=linewidth, alpha=0.8)
            block_idx += 1
        x_.append(new_x)
        y_.append(new_y)
        n += 3
        if block_idx >= num_blocks:
            break
    return []


# 生成帧序列：正向+反向
frames_seq = list(range(len(df))) + list(reversed(range(len(df))))

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
        # 每条折线的色系独立，但起点色承接上一个终点色
        if i == 0:
            this_start = color_start
        else:
            this_start = color_table[-1][1]
        # 新的色相和亮度
        hue = np.random.rand()
        s = 0.08 + 0.14 * np.random.rand()
        v2 = 0.9
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
    # 取正向或反向帧
    if frame < len(df):
        idx = frame
    else:
        idx = 2*len(df)-1 - frame
    for i in range(min(idx, len(df))):
        color_start, color_end = color_table[i]
        angle += 91
        rad = np.deg2rad(angle)
        new_x = x_[-1] + n * np.cos(rad)
        new_y = y_[-1] + n * np.sin(rad)
        linewidth = 24 + 8 * df['value'].iloc[i] / max(df['value'])
        seg_num = 20
        for seg in range(seg_num):
            t0 = seg / seg_num
            t1 = (seg + 1) / seg_num
            px0 = x_[-1] + t0 * (new_x - x_[-1])
            py0 = y_[-1] + t0 * (new_y - y_[-1])
            px1 = x_[-1] + t1 * (new_x - x_[-1])
            py1 = y_[-1] + t1 * (new_y - y_[-1])
            color = color_start * (1 - t0) + color_end * t0
            ax.plot([px0, px1], [py0, py1], color=color, linewidth=linewidth, alpha=0.8)
        x_.append(new_x)
        y_.append(new_y)
        n += 3
    return []

def on_repeat(*args):
    global color_table
    color_table = gen_color_table()


# 让动画帧数等于色块总数，循环时每组重新生成颜色
seg_num = 20
total_blocks = len(df) * seg_num

# 动画帧数为正向+反向
ani = animation.FuncAnimation(
    fig, update, frames=2*total_blocks, init_func=init, blit=True, interval=180, repeat=True
)
ani._repeat_callback = on_repeat

plt.show()
# ani.save('spiral_rainfall.gif', writer='imagemagick')
# ani.save('spiral_rainfall.mp4', writer='ffmpeg')
# plt.show()  # <--- 加上这一行
# 生成颜色表和透明度相位表
def get_color_table():
    pass  # TODO: 实现或删除此函数体