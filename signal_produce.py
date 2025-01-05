import numpy as np
from scipy.signal import firwin, kaiserord, lfilter
import random
import pywt
import matplotlib.pyplot as plt
import os
from skimage.transform import resize

# Define the sampling rate and time vector globally
sampling_rate = 2000
t = np.arange(0, 1.0, 1.0 / sampling_rate)

# Initialize global frequency variable
signal_f = 0


def f_multi_jam(jnr):
    Q = random.randint(4, 7)  # Number of multiple sinusoids
    fs0 = random.randint(50, 150)
    deltafs = 30
    fs = np.arange(fs0, fs0 + (Q - 1) * deltafs + 1, deltafs)
    global signal_f
    signal_f = np.mean(fs)
    theta = 2 * np.pi * np.random.rand(Q)  # Random phase
    z = np.exp(1j * (2 * np.pi * fs[:, None] * t + theta[:, None]))
    z = np.sum(z, axis=0)
    y = awgn(z, jnr)
    return y


def f_broadbandnoise_jam(jnr):
    """
    Generates a complex broadband noise jamming signal.

    Parameters:
    - jnr: Signal-to-noise ratio in dB.

    Returns:
    - Complex broadband noise signal with AWGN.
    """
    # Noise power
    P = 1 - 0.1 * np.random.rand()
    y = np.random.normal(0, np.sqrt(P), int(sampling_rate)) + 1j * np.random.normal(0, np.sqrt(P),
                                                                                    int(sampling_rate))  # White noise

    # Bandpass filter parameters
    WI = random.randint(20, 40)
    global signal_f
    signal_f = WI + 200

    fl_low = WI / (0.5 * sampling_rate)  # Normalized lower frequency
    fl_high = (WI + 400) / (0.5 * sampling_rate)  # Normalized upper frequency

    # Kaiser window design parameters
    width = 0.05  # Transition bandwidth
    ripple_db = 60  # Stopband attenuation (dB)

    # Design filter using Kaiser window
    fl_n_kaiser, beta = kaiserord(ripple_db, width)
    taps = firwin(fl_n_kaiser, [fl_low, fl_high], pass_zero=False, window=('kaiser', beta))

    # Apply filter
    Y_bp = lfilter(taps, 1.0, y)

    # Convert filtered signal to complex exponential signal
    phase = np.cumsum(2 * np.pi * WI / sampling_rate)  # Cumulative phase
    y_complex = Y_bp * np.exp(1j * phase)  # Complex signal

    # Add AWGN
    y_noisy = awgn(y_complex, jnr)
    return y_noisy


def f_narrowbandnoise_jam(jnr, sampling_rate=2000):
    """
    Generates a narrowband noise jamming signal.

    Parameters:
    - jnr: Signal-to-noise ratio in dB.
    - sampling_rate: Sampling rate in Hz.

    Returns:
    - Narrowband noise signal with AWGN.
    """
    # Noise power
    P = 1 - 0.1 * np.random.rand()
    y = np.random.normal(0, np.sqrt(P), int(sampling_rate)) + 1j * np.random.normal(0, np.sqrt(P),
                                                                                    int(sampling_rate))  # White noise

    # Narrowband parameters
    WI = 20  # Narrowband window width
    FJ = random.randint(100, 300)  # Random interference center frequency
    global signal_f
    signal_f = FJ + 20

    # Kaiser window filter design parameters
    ripple_db = 60  # Stopband attenuation (dB)
    width = 0.05  # Transition bandwidth (normalized frequency)
    fl_n_kaiser, beta = kaiserord(ripple_db, width)  # Filter order and Kaiser parameter

    # Define narrowband range (4 subbands)
    fl_kaiser = [
        FJ - WI / 2,  # Narrowband 1 lower
        FJ - WI / 4,  # Narrowband 1 upper
        FJ + WI / 4,  # Narrowband 2 lower
        FJ + WI / 2  # Narrowband 2 upper
    ]
    fl_wn = [freq / (0.5 * sampling_rate) for freq in fl_kaiser]  # Normalized frequency range

    # Create filter
    h = firwin(fl_n_kaiser, fl_wn, window=('kaiser', beta), pass_zero=False)

    # Apply bandpass filter
    Y_bp = lfilter(h, 1, y)

    # Add AWGN
    y = awgn(Y_bp, jnr)
    return y


def f_combnoise_jam(jnr):
    """
    Generates a comb noise jamming signal.

    Parameters:
    - jnr: Signal-to-noise ratio in dB.

    Returns:
    - Comb noise signal with AWGN.
    """
    # Noise power
    P = 1 - 0.1 * np.random.rand()
    y = np.random.normal(0, np.sqrt(P), int(sampling_rate)) + 1j * np.random.normal(0, np.sqrt(P),
                                                                                    int(sampling_rate))  # White noise

    # Bandpass filter parameters
    WI = random.randint(10, 20)
    num = random.randint(3, 4)
    startf = random.randint(90, 150)
    freq_delta = random.randint(60, 90)
    FJ_set = [startf + i * freq_delta for i in range(num)]
    global signal_f
    signal_f = np.mean(FJ_set)
    Y_bp_temp = []

    for FJ in FJ_set:
        # Define narrowband frequency ranges
        fcuts = [FJ - WI / 3, FJ - WI / 4, FJ + WI / 4, FJ + WI / 3]
        ripple_db = 100  # Stopband attenuation (dB)
        width = 0.01  # Transition bandwidth (normalized frequency)

        # Design filter using Kaiser window
        n, beta = kaiserord(ripple_db, width)
        wn = [fc / (0.5 * sampling_rate) for fc in fcuts]  # Normalize frequencies

        h = firwin(n, wn, window=('kaiser', beta), pass_zero=False)
        Y_bp_temp.append(lfilter(h, 1, y))  # Apply bandpass filter

    Y_bp = np.sum(Y_bp_temp, axis=0)
    y = awgn(Y_bp, jnr)
    return y


def f_scanning_jam(jnr):
    """
    Generates a scanning (chirp) jamming signal.

    Parameters:
    - jnr: Signal-to-noise ratio in dB.

    Returns:
    - Scanning signal with AWGN.
    """
    global signal_f
    signal_f = 250
    om = 2 * np.pi * random.randint(20, 50)
    be = 2 * np.pi * random.randint(400, 500)
    phi = 2 * np.pi * random.random()

    x = np.exp(1j * 0.5 * be * t ** 2 + 1j * om * t + 1j * phi)
    y = awgn(x, jnr)
    return y


def awgn(signal, jnr):
    """
    Adds Additive White Gaussian Noise (AWGN) to a complex signal.

    Parameters:
    - signal (numpy.ndarray): Input complex signal.
    - jnr (float): Signal-to-noise ratio in dB.

    Returns:
    - numpy.ndarray: Noisy complex signal.
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    jnr_linear = 10 ** (jnr / 10)
    noise_power = signal_power / jnr_linear
    noise = np.sqrt(noise_power / 2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    return signal + noise


def create_signal(jnr, type, split, data_n):
    """
    Creates and saves scalogram images for a specific signal type, JNR, and data split.

    Parameters:
    - jnr (int): Jamming-to-Noise Ratio (e.g., -2, -4, -6, -8).
    - type (int): Signal type index (0 to 4).
    - split (str): Data split ('train' or 'val').
    - data_n (int): Number of samples to generate.
    """
    # Map signal type to corresponding function
    signal_functions = {
        0: f_multi_jam,
        1: f_broadbandnoise_jam,
        2: f_narrowbandnoise_jam,
        3: f_combnoise_jam,
        4: f_scanning_jam
    }

    if type not in signal_functions:
        raise ValueError('Invalid signal type. Must be between 0 and 4.')

    fc = signal_functions[type]

    # Define the folder path: jnr{JNR}/{split}/{type+1}
    folder_path = os.path.join(os.getcwd(), f'jnr{jnr}', split, f'{type + 1}')

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    for i in range(data_n):
        y = fc(jnr)

        # Define wavelet parameters
        wavename = 'morl'
        totalscal = 224
        fc_wavelet = pywt.central_frequency(wavename)
        cparam = 2 * fc_wavelet * totalscal
        scales = cparam / np.arange(totalscal, 1, -1)

        # Compute Continuous Wavelet Transform (CWT)
        coef, freqs = pywt.cwt(y, scales, wavename, 1.0 / sampling_rate)

        # target_size = (224, 224)
        # coef_resized = resize(abs(coef), target_size, mode='constant', anti_aliasing=True)
        #
        # # 归一化到 [0, 1] 范围（根据需要调整）
        # coef_normalized = (coef_resized - coef_resized.min()) / (coef_resized.max() - coef_resized.min())
        #
        # # 定义文件路径（保存为 .npy 文件）
        # file_path = os.path.join(folder_path, f'{i + 1}.npy')
        #
        # # 保存为 NumPy 数组
        # np.save(file_path, coef_normalized)

        # Create the figure
        fig, ax = plt.subplots(figsize=(6, 6))
        half = 0.5
        times = 20
        plt.xlim(half - times / signal_f, half + times / signal_f)
        plt.ylim(0, 600)
        contourf = ax.contourf(t, freqs, abs(coef))

        # Remove axes for cleaner images
        ax.axis('off')

        # Define the file path
        file_path = os.path.join(folder_path, f'{i + 1}.png')

        # Save the figure with specified DPI
        fig.savefig(file_path, dpi=37.4)

        # Close the figure to free memory
        plt.close(fig)


def create_dataset(jnr, split):
    """
    Creates a dataset for a given JNR and data split.

    Parameters:
    - jnr (int): Jamming-to-Noise Ratio (e.g., -2, -4, -6, -8).
    - split (str): Data split ('train' or 'val').
    """
    if split == 'train':
        data_n = 500
    elif split == 'val':
        data_n = 200
    else:
        raise ValueError("Split must be 'train' or 'val'")

    type_n = 5  # Number of signal types

    for type in range(type_n):
        print(f"Generating {split} data for JNR={jnr}, Signal Type={type + 1}")
        create_signal(jnr, type, split, data_n)


if __name__ == "__main__":
    # Define the list of JNRs
    jnr_list = [-2, -4, -6, -8]
    # jnr_list=[0,-10]
    splits = ['train', 'val']

    for jnr in jnr_list:
        for split in splits:
            print(f"Creating {split} dataset for JNR={jnr}")
            create_dataset(jnr, split)
    print("Dataset creation complete.")
