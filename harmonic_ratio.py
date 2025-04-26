# harmonic_ratio.py
import numpy as np

def calculate_harmonic_ratio(signal, fs, num_harmonics=10, axis='AP'):
    """
    1サイクルの加速度信号に対してHRを計算する。
    - signal: 1次元 numpy 配列 (1歩行周期分)
    - fs: サンプリング周波数 (Hz)
    - num_harmonics: 使用するハーモニック数 (デフォルト 10)
    - axis: 'AP' or 'VT' → HR = (even harmonics)/(odd harmonics)
            'ML'         → HR = (odd harmonics)/(even harmonics)
    戻り値: HR (float)。長さ不足や分母ゼロは np.nan を返す。
    """
    sig = np.asarray(signal, dtype=float)
    N = len(sig)
    # FFTスペクトルを取り、ハーモニックインデックスを算出
    X = np.fft.rfft(sig)
    max_idx = len(X) - 1

    even_inds = np.arange(2, 2 * num_harmonics + 1, 2)
    odd_inds  = np.arange(1, 2 * num_harmonics, 2)
    even_inds = even_inds[even_inds <= max_idx]
    odd_inds  = odd_inds[odd_inds  <= max_idx]

    even_sum = np.sum(np.abs(X[even_inds]))
    odd_sum  = np.sum(np.abs(X[odd_inds]))
    if even_sum == 0 or odd_sum == 0:
        return np.nan

    if axis in ('AP', 'VT'):
        return even_sum / odd_sum
    elif axis == 'ML':
        return odd_sum / even_sum
    else:
        raise ValueError("axis は 'AP' / 'VT' / 'ML' のいずれかで指定してください。")


def calculate_integrated_harmonic_ratio(signal, fs, num_harmonics=10, axis='AP'):
    """
    Calculate the integrated Harmonic Ratio (iHR) for a single gait cycle signal.
    iHR = (even power) / (even + odd power) * 100 for AP/VT,
          (odd power) / (even + odd power) * 100 for ML.
    - signal: 1D numpy array of acceleration data for one cycle.
    - fs: sampling frequency in Hz.
    - num_harmonics: number of harmonics to include.
    - axis: 'AP' or 'VT' for even/(even+odd), 'ML' for odd/(even+odd).
    Returns iHR as a percentage (0–100). Returns np.nan if insufficient data or division by zero.
    """
    sig = np.asarray(signal, dtype=float)
    N = len(sig)
    if N < 2 * num_harmonics + 1:
        return np.nan

    X = np.fft.rfft(sig)
    max_idx = len(X) - 1

    even_inds = np.arange(2, 2 * num_harmonics + 1, 2)
    odd_inds  = np.arange(1, 2 * num_harmonics, 2)
    even_inds = even_inds[even_inds <= max_idx]
    odd_inds  = odd_inds[odd_inds  <= max_idx]

    # Power is amplitude squared
    even_power = np.sum((np.abs(X[even_inds]))**2)
    odd_power  = np.sum((np.abs(X[odd_inds]))**2)
    total_power = even_power + odd_power
    if total_power == 0:
        return np.nan

    if axis in ('AP', 'VT'):
        return (even_power / total_power) * 100.0
    elif axis == 'ML':
        return (odd_power / total_power) * 100.0
    else:
        raise ValueError("axis must be 'AP', 'VT', or 'ML'")