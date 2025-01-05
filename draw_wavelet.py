import numpy as np
import matplotlib.pyplot as plt
import pywt

plt.rcParams['font.family'] = 'SimHei'  # 使用SimHei或其他支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

# 小波
sampling_rate = 2000
t = np.arange(0, 1.0, 1.0 / sampling_rate)
times = 0.5

# 定义频率
f1 = 50
f2 = 100
f3 = 200

# # 使用 np.piecewise 定义信号 x 和 z
# z = np.piecewise(t, [t < 0.5, t >= 0.5],
#                 [lambda t: 0.7 * np.sin(2 * np.pi * f1 * t) +
#                            0.9 * np.sin(2 * np.pi * f2 * t) +
#                            0.3 * np.sin(2 * np.pi * f3 * t),
#                  lambda t: 0.7 * np.sin(2 * np.pi * f1 * times * t) +
#                            0.9 * np.sin(2 * np.pi * f2 * times * t) +
#                            0.3 * np.sin(2 * np.pi * f3 * times * t)])
#
# x = np.piecewise(t, [t < 0.5, t >= 0.5],
#                 [lambda t: 0.7 * np.sin(2 * np.pi * f1 * times * t) +
#                            0.9 * np.sin(2 * np.pi * f2 * times * t) +
#                            0.3 * np.sin(2 * np.pi * f3 * times * t),
#                  lambda t: 0.7 * np.sin(2 * np.pi * f1 * t) +
#                            0.9 * np.sin(2 * np.pi * f2 * t) +
#                            0.3 * np.sin(2 * np.pi * f3 * t)])

# 定义频率缩放因子
times = 2

# 定义复指数信号生成函数
def create_complex_signal(t, condition_times):
    return -1j * (
        0.7 * np.exp(1j * 2 * np.pi * f1 * condition_times * t) +
        0.9 * np.exp(1j * 2 * np.pi * f2 * condition_times * t) +
        0.3 * np.exp(1j * 2 * np.pi * f3 * condition_times * t)
    )

# 使用 numpy.where 构建复数信号 z
z = np.where(
    t < 0.5,
    create_complex_signal(t, 1),
    create_complex_signal(t, times)
)

# 使用 numpy.where 构建复数信号 x
x = np.where(
    t < 0.5,
    create_complex_signal(t, times),
    create_complex_signal(t, 1)
)

# 检查信号的数据类型
print(f"z dtype: {z.dtype}")  # 应输出 complex
print(f"x dtype: {x.dtype}")  # 应输出 complex

# 定义小波函数和尺度
wavename = 'morl'
totalscal = 256
fc = pywt.central_frequency(wavename)
cparam = 2 * fc * totalscal
scales = cparam / np.arange(totalscal, 1, -1)

# 进行连续小波变换
cwtmatr_x, frequencies_x = pywt.cwt(x, scales, wavename, 1.0 / sampling_rate)
cwtmatr_z, frequencies_z = pywt.cwt(z, scales, wavename, 1.0 / sampling_rate)

# 绘制时频图的函数
def plot_scalogram(data, coef, freqs, t, title, name):
    plt.figure(figsize=(12, 6))
    # plt.imshow(np.abs(coef), extent=[t[0], t[-1], freqs[-1], freqs[0]],
    #            cmap='jet', aspect='auto')
    # plt.colorbar(label='Magnitude')
    # plt.xlabel("Time (s)")
    # plt.ylabel("Frequency (Hz)")
    # plt.title(title, fontsize=20)
    # plt.yscale('log')
    # plt.ylim(200, 800)
    # plt.axhline(y=600, color='white', linestyle='--', linewidth=1)
    # plt.text(t[-1], 600, '600 Hz', color='white', va='center', ha='right', fontsize=10,
    #          backgroundcolor='black')
    # plt.tick_params(axis='y', which='both', labelleft=False)
    # plt.tight_layout()
    # plt.savefig('{}_wavelet.png'.format(name.replace(" ", "_")), dpi=800)
    # plt.show()

    plt.subplot(211)
    plt.xlim(0.3, 0.7)
    plt.plot(t, data, color='#057748')
    plt.xlabel("t(s)")
    plt.title(title, fontsize=10)
    plt.subplot(212)
    plt.xlim(0.3, 0.7)
    plt.ylim(0, 400)
    plt.contourf(t, freqs, abs(coef))
    plt.ylabel(u"prinv(Hz)")
    plt.xlabel(u"t(s)")
    plt.subplots_adjust(hspace=0.4)
    # plt.savefig('{}_wavelet.png'.format(name.replace(" ", "_")), dpi=800)
    plt.show()


# 绘制信号 x 的时频图
plot_scalogram(x,cwtmatr_x, frequencies_x, t, '原始信号时频谱图', '原始信号_x')

# 绘制信号 z 的时频图
plot_scalogram(z, cwtmatr_z, frequencies_z, t, '变化信号时频谱图', '频率变化信号_z')

