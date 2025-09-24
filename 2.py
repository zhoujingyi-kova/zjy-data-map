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
    print(f"frame={frame}, len(df)={len(df)}")  # 调试输出
    for i in range(min(frame, len(df))):
        print(f"i={i}, value={df['value'].iloc[i]}")  # 调试输出
        pcolor = np.random.rand(3,)
        angle += 91
        rad = np.deg2rad(angle)
        new_x = x_[-1] + n * np.cos(rad)
        new_y = y_[-1] + n * np.sin(rad)
        linewidth = 2 + df['value'].iloc[i] / max(df['value'])
        ax.plot([x_[-1], new_x], [y_[-1], new_y], color=pcolor, linewidth=linewidth, alpha=0.8)
        x_.append(new_x)
        y_.append(new_y)
        n += 3
    return []

ani = animation.FuncAnimation(
    fig, update, frames=len(df), init_func=init, blit=True, interval=200, repeat=False
)

plt.show()
# ani.save('spiral_rainfall.gif', writer='imagemagick')
# ani.save('spiral_rainfall.mp4', writer='ffmpeg')
# plt.show()  # <--- 加上这一行
