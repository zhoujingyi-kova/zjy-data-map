
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# 数据读取与处理（只取最后35个）
df = pd.read_csv('daily_HKO_RF_ALL.csv', na_values=["*** 沒有數據/unavailable"])
df = df.rename(columns={'年/Year': 'year', '月/Month': 'month', '日/Day': 'day', '數值/Value': 'value'})
df = df.dropna(subset=['year', 'month', 'day', 'value'])
df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
df = df.dropna(subset=['date'])
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df = df[['date', 'value']].sort_values('date').reset_index(drop=True)
df = df.tail(35)


n_init = 3  # 初始边长






# 随机生成立方体位置
np.random.seed(42)  # 保证每次动画一致，如需每次不同可注释掉
x = np.random.uniform(-10, 10, len(df))
y = np.random.uniform(-10, 10, len(df))
z = np.random.uniform(-10, 10, len(df))



# 设定颜色阈值：低值为蓝色(0,0,1)，高值为红色(1,0,0)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 使用matplotlib colormap实现连续颜色映射
vmin, vmax = df['value'].min(), df['value'].max()
import matplotlib.cm as cm
norm = [(v-vmin)/(vmax-vmin+1e-6) for v in df['value']]
# 极致区分：HSV色相全区间映射，最大饱和度和明度
# 色相随机，饱和度体现数据大小，明度为1
import colorsys
np.random.seed(123)
random_hues = np.random.rand(len(df))
def norm_to_rgb(norm, hue):
    # hue随机，saturation区间0.6~1，整体更鲜艳
    sat = 0.6 + 0.4 * norm  # 数据最小s=0.6，最大s=1
    return colorsys.hsv_to_rgb(hue, sat, 1)
colors = [norm_to_rgb(n, h) for n, h in zip(norm, random_hues)]
# 正方形大小：数据越大越大
min_size, max_size = 60, 300
sizes = [min_size + (max_size-min_size)*(v-vmin)/(vmax-vmin+1e-6) for v in df['value']]

# 生成每帧的索引序列，数据越大越慢
frame_indices = []
for i, v in enumerate(df['value']):
    # 速度：数据越大越慢（帧数越多）
    repeat = int(2 + 8 * (v-vmin)/(vmax-vmin+1e-6))  # 2~10帧
    frame_indices.extend([i]*repeat)


fig = plt.figure(figsize=(9, 9))
ax = fig.add_subplot(111, projection='3d')
ax.set_axis_off()
lines = []

def init():
    ax.cla()
    # 深色立体场景
    ax.set_facecolor('#181c24')
    ax.xaxis.set_pane_color((0.10, 0.12, 0.16, 1.0))
    ax.yaxis.set_pane_color((0.10, 0.12, 0.16, 1.0))
    ax.zaxis.set_pane_color((0.10, 0.12, 0.16, 1.0))
    ax.set_axis_off()
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(-12, 12)
    return lines

def update(frame):
    ax.cla()
    # 深色立体场景
    ax.set_facecolor('#181c24')
    ax.xaxis.set_pane_color((0.10, 0.12, 0.16, 1.0))
    ax.yaxis.set_pane_color((0.10, 0.12, 0.16, 1.0))
    ax.zaxis.set_pane_color((0.10, 0.12, 0.16, 1.0))
    ax.set_axis_off()
    ax.set_xlim(-12, 12)
    ax.set_ylim(-12, 12)
    ax.set_zlim(-12, 12)
    idx = min(frame+1, len(x)) if frame < len(frame_indices) else len(x)
    # 灰色虚线连线
    if idx > 1:
        ax.plot(x[:idx], y[:idx], z[:idx], color='gray', linestyle='dashed', linewidth=1.2, alpha=0.7)
    for i in range(idx):
        # 立方体参数
        side = (sizes[i] ** 0.5) * 0.18
        cx, cy, cz = x[i], y[i], z[i]
        s = side/2
        vertices = np.array([
            [cx-s, cy-s, cz-s],
            [cx+s, cy-s, cz-s],
            [cx+s, cy+s, cz-s],
            [cx-s, cy+s, cz-s],
            [cx-s, cy-s, cz+s],
            [cx+s, cy-s, cz+s],
            [cx+s, cy+s, cz+s],
            [cx-s, cy+s, cz+s],
        ])
        faces = [
            [vertices[j] for j in [0,1,2,3]],   # bottom
            [vertices[j] for j in [4,5,6,7]],   # top
            [vertices[j] for j in [0,1,5,4]],   # front
            [vertices[j] for j in [2,3,7,6]],   # back
            [vertices[j] for j in [1,2,6,5]],   # right
            [vertices[j] for j in [0,3,7,4]],   # left
        ]
        # 6面不同明暗，模拟光照
        base = np.array(colors[i])
        # 自发光：整体加亮
        base_glow = np.clip(base*1.5+0.3, 0, 1)
        face_shades = [0.55, 1.0, 0.8, 0.7, 0.65, 0.6]  # top最亮，bottom最暗
        facecolors = [tuple(np.clip(base_glow*shade, 0, 1)) for shade in face_shades]
        cube = Poly3DCollection(faces, facecolors=facecolors, edgecolors=None, linewidths=0, alpha=0.93)
        ax.add_collection3d(cube)
        # 日期标签白色
        date_str = df['date'].iloc[i].strftime('%Y-%m-%d')
        ax.text(cx, cy, cz-s-1.2, date_str, fontsize=7, color='white', ha='center', va='top')
    return []



# 动画帧序列，速度随数据大小变化
ani = animation.FuncAnimation(
    fig, update, frames=len(frame_indices), init_func=init, blit=True, interval=120, repeat=True
)
plt.show()