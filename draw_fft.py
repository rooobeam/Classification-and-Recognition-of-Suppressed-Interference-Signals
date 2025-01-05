import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 1. 设置Matplotlib使用支持中文的字体
plt.rcParams['font.family'] = 'SimHei'  # 使用SimHei或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 2. 定义信号参数
Fs = 10000           # 采样频率为10000 Hz
T = 1.0              # 信号持续时间为1秒
t = np.arange(0, T, 1/Fs)  # 时间向量

# 3. 生成叠加信号
half_point = int(0.5 * Fs)  # 0.5秒对应的采样点数
x = np.zeros_like(t)
z = np.zeros_like(t)
times=0.3
# 前0.5秒，原始频率
x[:half_point] = (0.7 * np.sin(2 * np.pi * 50 * t[:half_point]) +
                 0.9 * np.sin(2 * np.pi * 100 * t[:half_point]) +
                 0.3 * np.sin(2 * np.pi * 200 * t[:half_point]))

# 后0.5秒，频率减小一倍
x[half_point:] = (0.7 * np.sin(2 * np.pi * 50*times * t[:half_point]) +
                 0.9 * np.sin(2 * np.pi * 100*times * t[:half_point]) +
                 0.3 * np.sin(2 * np.pi * 200*times * t[:half_point]))


z[half_point:] = (0.7 * np.sin(2 * np.pi * 50 * t[:half_point]) +
                 0.9 * np.sin(2 * np.pi * 100 * t[:half_point]) +
                 0.3 * np.sin(2 * np.pi * 200 * t[:half_point]))

# 后0.5秒，频率减小一倍
z[:half_point] = (0.7 * np.sin(2 * np.pi * 50*times * t[:half_point]) +
                 0.9 * np.sin(2 * np.pi * 100*times * t[:half_point]) +
                 0.3 * np.sin(2 * np.pi * 200*times * t[:half_point]))
N = len(x)  # 信号长度

# 4. 进行傅里叶变换
Y = fft(x)
YY = fft(z)
f = Fs * np.arange(0, N//2 + 1) / N  # 计算频率轴

# 5. 计算功率谱
power = np.abs(Y)**2 / N
power1 = np.abs(YY)**2 / N
# 6. 绘制时域图和频域图
plt.figure(figsize=(14, 8))

# 时域图（仅实部）
plt.subplot(3, 1, 1)
plt.plot(t, z, color='steelblue', linewidth=0.7)
plt.title('叠加信号的时域图')
plt.xlabel('时间 (秒)')
plt.ylabel('幅度')
plt.grid(True)
plt.xlim(0.3, 0.7)  # 控制时间取值范围

# 时域图（仅实部）
plt.subplot(3, 1, 2)
plt.plot(t, x, color='seagreen', linewidth=0.7)
plt.title('变化后的信号的时域图')
plt.xlabel('时间 (秒)')
plt.ylabel('幅度')
plt.grid(True)
plt.xlim(0.3, 0.7)  # 控制时间取值范围

# 频域图
plt.subplot(3, 1, 3)
plt.plot(f, power[:N//2 + 1], color='#9B6622', linewidth=0.7)
plt.title('二者的傅里叶变换频域图')
plt.xlabel('频率 (Hz)')
plt.ylabel('功率')
plt.grid(True)
plt.xlim(0, 300)  # 根据信号频率范围调整显示范围

# 调整子图之间的垂直间距
plt.subplots_adjust(hspace=0.4)
plt.savefig('fft1.png', dpi=600)

plt.tight_layout()
plt.show()
