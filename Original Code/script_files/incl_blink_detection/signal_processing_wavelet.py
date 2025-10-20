import numpy as np
import pywt
from scipy import signal as sig
from config import FS, LOWCUT, HIGHCUT, FILTER_ORDER, BLINK_THRESHOLD, BLINK_MIN_DURATION, BLINK_MAX_DURATION

# Minimum length required for filtfilt based on filter order
# For a 4th order filter, we need at least 3*(max(len(b), len(a))-1) samples
# For a notch filter, this is typically around 30 samples
MIN_SIGNAL_LENGTH = 50  # Conservative minimum to ensure all filters work

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return sig.butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order):
    # Check if data is long enough for filtering
    if len(data) < MIN_SIGNAL_LENGTH:
        print(f"Warning: Signal too short for bandpass filtering ({len(data)} < {MIN_SIGNAL_LENGTH} samples)")
        return data  # Return unfiltered data

    b, a = butter_bandpass(lowcut, highcut, fs, order)
    try:
        return sig.filtfilt(b, a, data)
    except Exception as e:
        print(f"Error in bandpass filtering: {e}")
        return data  # Return unfiltered data if filtering fails

def notch_filter(data, fs, freq=50, bandwidth=5):
    # Check if data is long enough for filtering
    if len(data) < MIN_SIGNAL_LENGTH:
        print(f"Warning: Signal too short for notch filtering ({len(data)} < {MIN_SIGNAL_LENGTH} samples)")
        return data  # Return unfiltered data

    w0 = freq / (fs/2)
    Q = freq / bandwidth
    b, a = sig.iirnotch(w0, Q)
    try:
        return sig.filtfilt(b, a, data)
    except Exception as e:
        print(f"Error in notch filtering: {e}")
        return data  # Return unfiltered data if filtering fails

def wavelet_transform(data, channel_type):
    if len(data) == 0:
        return data

    # Check if data is long enough for wavelet transform
    if len(data) < 10:  # Minimum for wavelet transform
        print(f"Warning: Signal too short for wavelet transform ({len(data)} < 10 samples)")
        return data

    if channel_type in ["ch2", "ch8"]:
        wavelet = 'db4'
        level = min(pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len), 1)
        coeffs = pywt.wavedec(data, wavelet, level=level)
        new_coeffs = [coeffs[0]]
        for i in range(1, len(coeffs)):
            threshold = np.std(coeffs[i]) * (0.7 if i==1 else 0.2)
            new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
    else:
        wavelet = 'db4'
        level = min(pywt.dwt_max_level(len(data), pywt.Wavelet(wavelet).dec_len), 1)
        coeffs = pywt.wavedec(data, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        new_coeffs = [coeffs[0]]
        for i in range(1, len(coeffs)):
            new_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))

    try:
        # Reconstruct once
        data = pywt.waverec(new_coeffs, wavelet)
        data = data[:len(data)]  # truncate/pad to original length
        return data
    except Exception as e:
        print(f"Error in wavelet transform: {e}")
        return data  # Return original data if transform fails

def process_signal(data, fs, channel_type):
    """Apply notch, bandpass filters, and wavelet transform to signal data"""
    # Check if data is empty or too short
    if len(data) == 0:
        return data

    if len(data) < MIN_SIGNAL_LENGTH:
        print(f"Warning: Signal too short for full processing ({len(data)} < {MIN_SIGNAL_LENGTH} samples)")
        return data

    # First apply notch filter to remove power line interference
    try:
        data = notch_filter(data, fs)
    except Exception as e:
        print(f"Error in notch filter: {e}")

    # Then apply bandpass filter
    try:
        data = bandpass_filter(data, LOWCUT, HIGHCUT, fs, FILTER_ORDER)
    except Exception as e:
        print(f"Error in bandpass filter: {e}")

    # Apply wavelet transform
    try:
        data = wavelet_transform(data, channel_type)
    except Exception as e:
        print(f"Error in wavelet transform: {e}")

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

    # Check if data is long enough for processing
    if min_len < MIN_SIGNAL_LENGTH:
        # Return empty arrays if data is too short
        return np.array([]), np.array([]), np.array([])

    ch1 = ch1[:min_len]
    ch2 = ch2[:min_len]
    ch3 = ch3[:min_len]
    ch8 = ch8[:min_len]

    # Apply notch + bandpass with error handling
    try:
        ch1 = process_signal(ch1, FS, "ch1")
        ch2 = process_signal(ch2, FS, "ch2")
        ch3 = process_signal(ch3, FS, "ch3")
        ch8 = process_signal(ch8, FS, "ch8")
    except Exception as e:
        print(f"Error processing individual channels: {e}")
        return np.array([]), np.array([]), np.array([])

    # Normalize
    try:
        ch1 /= calibration_params["channel_norm_factors"]["ch1"]
        ch2 /= calibration_params["channel_norm_factors"]["ch2"]
        ch3 /= calibration_params["channel_norm_factors"]["ch3"]
        ch8 /= calibration_params["channel_norm_factors"]["ch8"]
    except Exception as e:
        print(f"Error normalizing channels: {e}")
        return np.array([]), np.array([]), np.array([])

    # H and V signals
    try:
        H_raw = ch1 - ch3
        V_raw = ch8 - ch2
    except Exception as e:
        print(f"Error calculating H and V signals: {e}")
        return np.array([]), np.array([]), np.array([])

    # Baseline correction
    try:
        H_raw -= calibration_params["baselines"]["H"]
        V_raw -= calibration_params["baselines"]["V"]
    except Exception as e:
        print(f"Error applying baseline correction: {e}")
        return np.array([]), np.array([]), np.array([])

    # Wavelet denoising with error handling
    def wavelet_denoise(x, wavelet='db4', level=3):
        if len(x) < 10:
            return x

        try:
            level = min(pywt.dwt_max_level(len(x), pywt.Wavelet(wavelet).dec_len), level)
            coeffs = pywt.wavedec(x, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            uthresh = sigma * np.sqrt(2 * np.log(len(x)))
            coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
            return pywt.waverec(coeffs, wavelet)[:len(x)]
        except Exception as e:
            print(f"Error in wavelet denoising: {e}")
            return x

    try:
        H_denoised = wavelet_denoise(H_raw)
        V_denoised = wavelet_denoise(V_raw)
    except Exception as e:
        print(f"Error in wavelet denoising: {e}")
        return np.array([]), np.array([]), np.array([])

    # Alpha compensation (vertical)
    try:
        alpha = calibration_params.get("alpha", 0.0)
        V_compensated = V_denoised - alpha * H_denoised
    except Exception as e:
        print(f"Error in alpha compensation: {e}")
        return np.array([]), np.array([]), np.array([])

    # Final low-pass smoothing with error handling
    try:
        b, a = sig.butter(2, 10 / (FS/2), btype='low')
        H_filt = sig.filtfilt(b, a, H_denoised)
        V_filt = sig.filtfilt(b, a, V_compensated)
        return H_filt, V_denoised, V_filt
    except Exception as e:
        print(f"Error in final smoothing: {e}")
        return H_denoised, V_denoised, V_compensated  # Return without final smoothing

def detect_blinks(V_signal, fs, blink_threshold=BLINK_THRESHOLD):
    """
    Detect blinks in the vertical EOG signal.
    Returns a list of blink events with start, end, and peak information.
    """
    from scipy.signal import find_peaks

    # Check if signal is long enough
    if len(V_signal) < MIN_SIGNAL_LENGTH:
        print(f"Warning: Signal too short for blink detection ({len(V_signal)} < {MIN_SIGNAL_LENGTH} samples)")
        return []

    try:
        # Find peaks in the absolute value of the signal
        peaks, properties = find_peaks(
            np.abs(V_signal),
            height=blink_threshold,
            distance=int(0.1 * fs),
            width=(int(0.03 * fs), int(0.3 * fs))
        )

        blink_events = []
        for peak_idx in peaks:
            try:
                peak_value = V_signal[peak_idx]

                # Only consider positive peaks (blinks typically show as positive peaks in V)
                if peak_value > blink_threshold:
                    # Find the start and end of the blink
                    half_peak = peak_value / 2

                    # Search backward for the start
                    start_idx = peak_idx
                    while start_idx > 0 and np.abs(V_signal[start_idx]) > half_peak:
                        start_idx -= 1

                    # Search forward for the end
                    end_idx = peak_idx
                    while end_idx < len(V_signal) - 1 and np.abs(V_signal[end_idx]) > half_peak:
                        end_idx += 1

                    # Calculate duration in seconds
                    duration = (end_idx - start_idx) / fs

                    # Only accept blinks with reasonable duration
                    if BLINK_MIN_DURATION <= duration <= BLINK_MAX_DURATION:
                        blink_events.append({
                            'peak_index': peak_idx,
                            'start_index': start_idx,
                            'end_index': end_idx,
                            'peak_value': peak_value,
                            'duration': duration,
                            'peak_time': peak_idx / fs
                        })
            except Exception as e:
                print(f"Error processing peak {peak_idx}: {e}")
                continue

        return blink_events
    except Exception as e:
        print(f"Error in blink detection: {e}")
        return []

def process_eog_signals_with_blinks(ch1, ch2, ch3, ch8, calibration_params=None):
    """
    Process EOG signals with blink detection.
    Returns H, V, V_compensated, and blink_events.
    """
    # First get the standard processed signals
    H, V, V_compensated = process_eog_signals(ch1, ch2, ch3, ch8, calibration_params)

    # Check if we got valid signals
    if len(H) == 0 or len(V) == 0 or len(V_compensated) == 0:
        return np.array([]), np.array([]), np.array([]), []

    # Detect blinks in the V signal
    blink_events = detect_blinks(V_compensated, FS, BLINK_THRESHOLD)

    return H, V, V_compensated, blink_events
