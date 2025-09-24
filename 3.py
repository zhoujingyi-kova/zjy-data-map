import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# 数据读取与处理
df = pd.read_csv('daily_HKO_RF_ALL.csv', skiprows=2, na_values=["*** 沒有數據/unavailable"])
df = df.rename(columns={
    '年/Year': 'year',
    '月/Month': 'month',
    '日/Day': 'day',
    '數值/Value': 'value'
})
df = df.dropna(subset=['year', 'month', 'day', 'value'])
df['date'] = pd.to_datetime(df[['year', 'month', 'day']], errors='coerce')
df = df.dropna(subset=['date'])
df['value'] = pd.to_numeric(df['value'], errors='coerce')
df = df[['date', 'value']].sort_values('date').reset_index(drop=True)
df = df.tail(20)  # 取倒数后20个数据


n_init = 3  # 初始边长（恢复原值）

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
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')
    return lines

def update(frame):
    ax.clear()
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.axis('off')
    n = n_init
    # 整体旋转角度，随帧数缓慢增加
    global_angle_offset = frame * 0.03  # 控制旋转速度
    angle = 0
    x_, y_ = [0], [0]
    lines = []
    for i in range(min(frame, len(df))):
        angle += 91
        rad = np.deg2rad(angle) + global_angle_offset
        new_x = x_[-1] + n * np.cos(rad)
        new_y = y_[-1] + n * np.sin(rad)
        ax.plot([x_[-1], new_x], [y_[-1], new_y], color='gray', linewidth=1, alpha=0)
        # 在折线上均匀生成3个球
        for k in range(3):
            t = (k + 1) / 4  # 取1/4, 2/4, 3/4位置
            px = x_[-1] + t * (new_x - x_[-1])
            py = y_[-1] + t * (new_y - y_[-1])
            # 呼吸动画：点大小随帧数周期性变化
            base_size = 80 + 300 * (df['value'].iloc[i] / (df['value'].max() if df['value'].max() > 0 else 1))
            breath = 0.5 + 0.5 * np.sin(frame * 0.18 + i + k)
            size = base_size * (0.4 + 1.2 * breath)
            # alpha也随呼吸变化，制造闪烁感
            alpha = 0.5 + 0.5 * breath
            # 渐变颜色，随时间变化
            color = np.array([
                0.5 + 0.5 * np.sin(frame * 0.12 + i + k),
                0.5 + 0.5 * np.sin(frame * 0.15 + i + 2 + k),
                0.5 + 0.5 * np.sin(frame * 0.18 + i + 4 + k)
            ])
            color = np.clip(color, 0, 1)
            # 发光效果：先画一层大而淡的白色光晕，再画主球
            ax.scatter(px, py, s=size*2.5, c=[[1,1,1]], alpha=0.18*alpha, linewidths=0)
            ax.scatter(px, py, s=size, c=[color], alpha=alpha, edgecolors='none', linewidths=0)
        x_.append(new_x)
        y_.append(new_y)
        n += 3
    return []
    return []


frames_seq = list(range(len(df))) + list(reversed(range(len(df))))

# 为每一组生成一组新的随机颜色和初始透明度
def get_color_table():
    return np.random.rand(len(df), 5, 3)  # [折线数, 5球, RGB]
def get_alpha_phase_table():
    return np.random.uniform(0, 2*np.pi, size=(len(df), 5))  # [折线数, 5球]，每个球的相位

color_table = get_color_table()
alpha_phase_table = get_alpha_phase_table()

def update(frame):
    ax.clear()
    ax.set_facecolor('black')
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
        angle += 91
        rad = np.deg2rad(angle)
        new_x = x_[-1] + n * np.cos(rad)
        new_y = y_[-1] + n * np.sin(rad)
        # 折线中点
        cx = (x_[-1] + new_x) / 2
        cy = (y_[-1] + new_y) / 2
        # 生成5个球，均匀分布在圆环上，半径与折线长度相关
        line_len = np.hypot(new_x - x_[-1], new_y - y_[-1])
        ring_r = 0.25 * line_len
        for k in range(5):
            # 轨迹长度
            trail_len = 25  # 降低渐变球数量，提升流畅度
            # 让渐变球逐步出现，速度平均分配到两个球生成时间之间
            # 计算当前允许显示的渐变球数量
            # 每个球的出现帧数区间
            balls_per_trail = trail_len
            # 计算每个球的出现帧数区间
            # 让每个球的出现时间分配在两个主球之间
            # 总帧数/主球数 = 每个主球的帧区间
            main_ball_count = len(df) * 5
            frames_per_main_ball = max(len(frames_seq) // main_ball_count, 1)
            # 当前主球编号
            main_ball_idx = i * 5 + k
            # 当前主球的起始帧
            start_frame = main_ball_idx * frames_per_main_ball
            # 当前主球的结束帧
            end_frame = (main_ball_idx + 1) * frames_per_main_ball
            # 当前帧在主球区间内的进度
            if frame < start_frame:
                continue
            progress = min(max((frame - start_frame) / (end_frame - start_frame), 0), 1)
            show_trail = int(progress * balls_per_trail)
            for t_idx in range(show_trail):
                trail_frame = max(idx - t_idx, 0)
                theta = 2 * np.pi * k / 5 + trail_frame * 0.12
                px = cx + ring_r * np.cos(theta)
                py = cy + ring_r * np.sin(theta)
                rand_scale = np.random.uniform(0.7, 1.3)
                base_size = (120 + 600 * (df['value'].iloc[i] / (df['value'].max() if df['value'].max() > 0 else 1))) * rand_scale
                breath = 0.5 + 0.5 * np.sin(trail_frame * 0.18 + i + k)
                size = base_size * (0.4 + 1.2 * breath)
                color_head = color_table[i, k]
                color = color_head * (1 - t_idx / trail_len) + np.ones(3) * (t_idx / trail_len)
                color = np.clip(color, 0, 1)
                alpha = 0.8 * (1 - t_idx / trail_len) + 0.2
                # 只在头部加发光
                if t_idx == 0:
                    ax.scatter(px, py, s=size*2.5, c=[[1,1,1]], alpha=0.18*alpha, linewidths=0)
                ax.scatter(px, py, s=size, c=[color], alpha=alpha, edgecolors='none', linewidths=0)
        x_.append(new_x)
        y_.append(new_y)
        n += 3
    return []

def on_repeat(*args):
    global color_table, alpha_phase_table
    color_table = get_color_table()
    alpha_phase_table = get_alpha_phase_table()

ani = animation.FuncAnimation(
    fig, update, frames=frames_seq, init_func=init, blit=True, interval=420, repeat=True
)
ani._repeat_callback = on_repeat

plt.show()
# ani.save('spiral_rainfall.gif', writer='imagemagick')
# ani.save('spiral_rainfall.mp4', writer='ffmpeg')
# plt.show()  # <--- 加上这一行