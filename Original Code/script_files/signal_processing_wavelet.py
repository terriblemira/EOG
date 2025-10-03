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
    """Apply wavelet transform to signal data"""
    if len(data) == 0:
        return data

    # Choose wavelet type and decomposition level
    if channel_type in ["ch2", "ch8"]:  # Vertical channels
        wavelet = 'sym5'  # Symlet 5 wavelet
        level = min(pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len), 4)

        # Perform wavelet decomposition
        coeffs = pywt.wavedec(data, wavelet, level=level)

        # Special processing for vertical channels
        new_coeffs = [coeffs[0]]  # Keep approximation coefficients

        # Apply different processing to different scales
        for i in range(1, len(coeffs)):
            if i == 1:  # Highest frequency details - aggressive denoising
                threshold = np.std(coeffs[i]) * 0.7
                new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            else:  # Lower frequency details - preserve more signal
                threshold = np.std(coeffs[i]) * 0.2
                new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))

        # Reconstruct the signal
        data = pywt.waverec(new_coeffs, wavelet)[:len(data)]

    else:  # Horizontal channels
        wavelet = 'db4'  # Daubechies 4 wavelet
        level = 4

        coeffs = pywt.wavedec(data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))

        new_coeffs = [coeffs[0]]
        for i in range(1, len(coeffs)):
            new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))

        data = pywt.waverec(new_coeffs, wavelet)[:len(data)]

    # Ensure the output has the same length as input
    if len(data) > len(coeffs[0]):
        data = data[:len(coeffs[0])]
    elif len(data) < len(coeffs[0]):
        data = np.pad(data, (0, len(coeffs[0]) - len(data)), 'constant')

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