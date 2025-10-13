import numpy as np
import pywt
from scipy import signal as sig
from config import FS, LOWCUT, HIGHCUT, FILTER_ORDER

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return sig.butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return sig.filtfilt(b, a, data)

def notch_filter(data, fs, freq=50, bandwidth=5):
    w0 = freq / (fs/2)
    Q = freq / bandwidth
    b, a = sig.iirnotch(w0, Q)
    return sig.filtfilt(b, a, data)

def wavelet_transform(data, channel_type):
    if len(data) == 0:
        return data

    if channel_type in ["ch2", "ch8"]:
        wavelet = 'rbio3.1'
        level = min(pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len), 4)
        coeffs = pywt.wavedec(data, wavelet, level=level)
        new_coeffs = [coeffs[0]]
        for i in range(1, len(coeffs)):
            threshold = np.std(coeffs[i]) * (0.7 if i==1 else 0.2)
            new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    else:
        wavelet = 'rbio3.1'
        level = 4
        coeffs = pywt.wavedec(data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        new_coeffs = [coeffs[0]]
        for i in range(1, len(coeffs)):
            new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))

    # Reconstruct once
    data = pywt.waverec(new_coeffs, wavelet)
    data = data[:len(data)]  # truncate/pad to original length
    return data

def process_signal(data, fs, channel_type):
    """Apply notch, bandpass filters, and wavelet transform to signal data"""
    # First apply notch filter to remove power line interference
    data = notch_filter(data, fs)

    # Then apply bandpass filter
    data = bandpass_filter(data, LOWCUT, HIGHCUT, fs, FILTER_ORDER)

    # Apply wavelet transform
    data = wavelet_transform(data, channel_type)

    return data

def process_eog_signals(ch1, ch2, ch3, ch8, calibration_params=None):
    """
    Process EOG signals with wavelet denoising, alpha compensation, and final smoothing.

    Args:
        ch1, ch2, ch3, ch8: EOG channels as np.arrays
        calibration_params: Dictionary with baselines, channel_norm_factors, alpha

    Returns:
        H_filt: Processed horizontal signal
        V_denoised: Vertical signal before alpha compensation
        V_filt: Alpha-compensated vertical signal (ready for detection)
    """
    import numpy as np
    import pywt
    from scipy import signal as sig
    from signal_processing_wavelet import process_signal  # keep your notch+bandpass

    # Use default calibration if none provided
    if calibration_params is None:
        calibration_params = {
            "baselines": {"H": 0, "V": 0},
            "thresholds": {"left": 0.3, "right": 0.3, "up": 0.3, "down": 0.3},
            "channel_norm_factors": {"ch1": 1, "ch2": 1, "ch3": 1, "ch8": 1},
            "alpha": 0.0
        }

    # Ensure arrays are same length
    min_len = min(len(ch1), len(ch2), len(ch3), len(ch8))
    if min_len == 0:
        return np.array([]), np.array([]), np.array([])

    ch1 = ch1[:min_len]
    ch2 = ch2[:min_len]
    ch3 = ch3[:min_len]
    ch8 = ch8[:min_len]

    # Apply notch + bandpass
    ch1 = process_signal(ch1, FS, "ch1")
    ch2 = process_signal(ch2, FS, "ch2")
    ch3 = process_signal(ch3, FS, "ch3")
    ch8 = process_signal(ch8, FS, "ch8")

    # Normalize
    ch1 /= calibration_params["channel_norm_factors"]["ch1"]
    ch2 /= calibration_params["channel_norm_factors"]["ch2"]
    ch3 /= calibration_params["channel_norm_factors"]["ch3"]
    ch8 /= calibration_params["channel_norm_factors"]["ch8"]

    # H and V signals
    H_raw = ch1 - ch3
    V_raw = ch8 - ch2

    # Baseline correction
    H_raw -= calibration_params["baselines"]["H"]
    V_raw -= calibration_params["baselines"]["V"]

    # Wavelet denoising
    def wavelet_denoise(x, wavelet='db6', level=3):
        coeffs = pywt.wavedec(x, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(x)))
        coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
        return pywt.waverec(coeffs, wavelet)[:len(x)]

    H_denoised = wavelet_denoise(H_raw)
    V_denoised = wavelet_denoise(V_raw)

    # Alpha compensation (vertical)
    alpha = calibration_params.get("alpha", 0.0)
    V_compensated = V_denoised - alpha * H_denoised

    # Final low-pass smoothing to remove blockiness (cutoff ~10Hz)
    b, a = sig.butter(2, 10 / (FS/2), btype='low')
    H_filt = sig.filtfilt(b, a, H_denoised)
    V_filt = sig.filtfilt(b, a, V_compensated)

    return H_filt, V_denoised, V_filt


