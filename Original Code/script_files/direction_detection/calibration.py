import numpy as np
from signal_processing_wavelet import process_signal, process_eog_signals
from config import FS
import matplotlib
matplotlib.use('Agg')   # <-- non-interactive backend safe for threads & headless envs
import matplotlib.pyplot as plt
from datetime import datetime
import os
from config import DEBUG_PLOTS
# Create a shared, date-stamped results folder
from datetime import datetime
import os

RESULTS_DIR = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(RESULTS_DIR, exist_ok=True)


def is_valid_step(step):
    """Check if a step is a valid list or array of samples."""
    return isinstance(step, (list, np.ndarray)) and len(step) > 0

def process_channel_data(calibration_data, channel_name, channel_norm_factors, direction=None):
    """
    Process channel data for a specific channel and optionally for a specific direction.
    Returns concatenated, filtered, and normalized data.
    """
    try:
        # Get all valid steps for the channel
        valid_steps = []

        if direction:
            # For a specific direction
            if channel_name in calibration_data[direction]:
                # Check if calibration_data[direction][channel_name] exists and is not empty
                channel_data = calibration_data[direction][channel_name]
                if channel_data and len(channel_data) > 0:  # Explicitly check length
                    for step in channel_data:
                        if is_valid_step(step):
                            valid_steps.append(step)
        else:
            # For all directions
            for dir_name, dir_data in calibration_data.items():
                if channel_name in dir_data:
                    # Check if dir_data[channel_name] exists and is not empty
                    channel_data = dir_data[channel_name]
                    if channel_data and len(channel_data) > 0:  # Explicitly check length
                        for step in channel_data:
                            if is_valid_step(step):
                                valid_steps.append(step)

        # Check if we have any valid steps
        if len(valid_steps) == 0:
            return None

        # Concatenate, filter, and normalize
        ch_data = np.concatenate(valid_steps)
        ch_data = process_signal(ch_data, FS, channel_name)

        # Check if channel_norm_factors is a dictionary and has the channel
        if isinstance(channel_norm_factors, dict) and channel_name in channel_norm_factors:
            ch_data = ch_data / channel_norm_factors[channel_name]

        return ch_data
    except Exception as e:
        print(f"Error processing {channel_name} data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def calculate_normalized_baseline(calibration_data, channel_norm_factors, channel):
    """Calculate baseline from normalized H or V signals during center fixation"""
    center_data = calibration_data["center"]

    # Check if we have all required channels
    required_channels = ["ch1", "ch3"] if channel == "H" else ["ch8", "ch2"]
    if not all(center_data[ch] for ch in required_channels):
        print(f"Warning: Insufficient center data for {channel} baseline. Using 0.")
        return 0

    try:
        # Process all required channels at once
        processed_channels = {}
        for ch in required_channels:
            ch_data = process_channel_data(calibration_data, ch, channel_norm_factors, "center")
            if ch_data is None:
                raise ValueError(f"No valid data for {ch}")
            processed_channels[ch] = ch_data

        # Compute H or V based on channel type
        if channel == "H":
            signals = processed_channels["ch1"] - processed_channels["ch3"]
        else:  # V
            signals = processed_channels["ch8"] - processed_channels["ch2"]

        # Use median for robustness to outliers
        baseline = np.median(signals)

        # Validate baseline is reasonable
        if abs(baseline) > 5:
            print(f"Warning: {channel} baseline {baseline:.2f} seems unusually high. Clipping to ±5.")
            baseline = np.clip(baseline, -5, 5)

        print(f"Normalized {channel} baseline: {baseline:.4f}")
        return baseline
    except Exception as e:
        print(f"Error calculating normalized {channel} baseline: {str(e)}")
        return 0

def calculate_channel_norm_factor(calibration_data, channel_name):
    """Calculate normalization factor for a specific channel across all directions"""
    amplitudes = []

    for direction in calibration_data:
        ch_data = process_channel_data(calibration_data, channel_name, {channel_name: 1.0}, direction)
        if ch_data is not None:
            amplitudes.append(np.percentile(np.abs(ch_data), 90))

    if amplitudes:
        factor = np.median(amplitudes)
        return max(factor, 0.1)  # Ensure minimum factor
    return 1.0

def process_direction_data(calibration_data, direction, channel_norm_factors, baselines, alpha=0.0):
    """
    Process data for a specific direction, returning H, V, and V_compensated signals.
    Improved version with better alpha compensation.
    """
    try:
        # Process all channels for this direction
        ch1 = process_channel_data(calibration_data, "ch1", channel_norm_factors, direction)
        ch3 = process_channel_data(calibration_data, "ch3", channel_norm_factors, direction)
        ch8 = process_channel_data(calibration_data, "ch8", channel_norm_factors, direction)
        ch2 = process_channel_data(calibration_data, "ch2", channel_norm_factors, direction)

        # Check if we have valid data for all channels
        if any(data is None or len(data) == 0 for data in [ch1, ch3, ch8, ch2]):
            print(f"No valid data for {direction}")
            return None, None, None

        # Check if arrays have the same length
        min_len = min(len(arr) for arr in [ch1, ch3, ch8, ch2] if isinstance(arr, np.ndarray) and len(arr) > 0)
        if min_len == 0:
            print(f"No valid data for {direction}")
            return None, None, None

        # Trim arrays to the same length
        ch1 = ch1[:min_len]
        ch3 = ch3[:min_len]
        ch8 = ch8[:min_len]
        ch2 = ch2[:min_len]

        # Calculate H and V components
        H = ch1 - ch3
        V = ch8 - ch2

        # Apply baseline correction
        H = H - baselines["H"]
        V = V - baselines["V"]

        # Apply improved alpha compensation to vertical component
        # Use a more sophisticated compensation that accounts for signal dynamics
        if abs(alpha) > 0.01:  # Only apply if alpha is significant
            # Apply frequency-domain compensation for better results
            fft_H = np.fft.fft(H)
            fft_V = np.fft.fft(V)

            # Apply compensation in frequency domain
            fft_V_comp = fft_V - alpha * fft_H

            # Inverse FFT to get time-domain signal
            V_compensated = np.fft.ifft(fft_V_comp).real
        else:
            # Simple time-domain compensation for small alpha values
            V_compensated = V - alpha * H

        return H, V, V_compensated
    except Exception as e:
        print(f"Error processing {direction} data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def calculate_alpha(calibration_data, channel_norm_factors):
    """
    Calculate the alpha coefficient to remove interdependency between H and V EOG components.
    Improved version with better data selection and compensation.
    """
    try:
        # Process horizontal movement data with more robust data selection
        left_H_data = process_channel_data(calibration_data, "ch1", channel_norm_factors, "left")
        left_H_data_3 = process_channel_data(calibration_data, "ch3", channel_norm_factors, "left")
        right_H_data = process_channel_data(calibration_data, "ch1", channel_norm_factors, "right")
        right_H_data_3 = process_channel_data(calibration_data, "ch3", channel_norm_factors, "right")

        # Check if we have valid data
        if any(data is None or len(data) == 0 for data in [left_H_data, left_H_data_3, right_H_data, right_H_data_3]):
            print("Missing data for horizontal channels")
            return 0.0

        # Process vertical data during horizontal movements
        left_V_data = process_channel_data(calibration_data, "ch8", channel_norm_factors, "left")
        left_V_data_2 = process_channel_data(calibration_data, "ch2", channel_norm_factors, "left")
        right_V_data = process_channel_data(calibration_data, "ch8", channel_norm_factors, "right")
        right_V_data_2 = process_channel_data(calibration_data, "ch2", channel_norm_factors, "right")

        # Check if we have valid data
        if any(data is None or len(data) == 0 for data in [left_V_data, left_V_data_2, right_V_data, right_V_data_2]):
            print("Missing data for vertical channels")
            return 0.0

        # Calculate horizontal components
        left_H = left_H_data - left_H_data_3
        right_H = right_H_data - right_H_data_3

        # Calculate vertical components
        left_V = left_V_data - left_V_data_2
        right_V = right_V_data - right_V_data_2

        # Combine horizontal and vertical components
        H = np.concatenate([left_H, right_H])
        V = np.concatenate([left_V, right_V])

        # Find the optimal alpha using a more robust method
        alphas = np.linspace(-2, 2, 400)  # Wider range with more points

        # Calculate cross-correlation between H and V
        cross_corr = np.correlate(H - np.mean(H), V - np.mean(V), mode='full')
        lag = np.argmax(cross_corr) - (len(cross_corr) - 1) // 2

        # Calculate initial alpha estimate based on cross-correlation
        initial_alpha = np.corrcoef(H, V)[0, 1] * np.std(V) / np.std(H)

        # Refine alpha using variance minimization with better constraints
        variances = []
        for alpha in alphas:
            V_compensated = V - alpha * H
            # Use a more robust variance measure (median absolute deviation)
            variances.append(np.median(np.abs(V_compensated - np.median(V_compensated))))

        # Find alpha with minimum variance
        optimal_alpha = alphas[np.argmin(variances)]

        # Apply additional constraints to ensure reasonable alpha value
        if abs(optimal_alpha) > 1.5:
            optimal_alpha = np.sign(optimal_alpha) * 1.5

        print(f"Optimal alpha for interdependency removal: {optimal_alpha:.4f}")
        print(f"Minimum variance achieved: {min(variances):.4f}")
        print(f"Cross-correlation lag: {lag} samples")
        print(f"Initial alpha estimate: {initial_alpha:.4f}")

        # Plot the variance vs alpha curve for visualization
        if DEBUG_PLOTS:
            plot_alpha_optimization(alphas, variances, optimal_alpha, H, V)

        return optimal_alpha
    except Exception as e:
        print(f"Error calculating alpha: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0

def plot_alpha_optimization(alphas, variances, optimal_alpha, H, V):
    """Plot the alpha optimization curve with additional diagnostics"""
    try:
        plt.figure(figsize=(12, 8))

        # Plot variance vs alpha
        plt.subplot(2, 1, 1)
        plt.plot(alphas, variances)
        plt.axvline(x=optimal_alpha, color='r', linestyle='--',
                   label=f'Optimal alpha = {optimal_alpha:.4f}')
        plt.title("Variance of Compensated Vertical Component vs Alpha")
        plt.xlabel("Alpha")
        plt.ylabel("Variance")
        plt.legend()

        # Plot H and V signals with optimal compensation
        plt.subplot(2, 1, 2)
        plt.plot(H, label='H signal')
        plt.plot(V, label='V signal (uncompensated)')
        plt.plot(V - optimal_alpha * H, label=f'V signal (α={optimal_alpha:.2f})')
        plt.title("Signal Comparison with Optimal Alpha Compensation")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        plt.legend()

        plot_dir = os.path.join(RESULTS_DIR, "calibration_plots")
        os.makedirs(plot_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(plot_dir, f"alpha_optimization_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved alpha optimization plot to {plot_path}")
        plt.close()
    except Exception as e:
        print(f"Could not save alpha optimization plot: {e}")

def calculate_direction_thresholds(calibration_data, channel_norm_factors, baselines):
    """Calculate direction-specific thresholds using normalized signals with alpha compensation"""
    thresholds = {}

    # Calculate alpha
    alpha = calculate_alpha(calibration_data, channel_norm_factors)
    print(f"Calculated alpha: {alpha:.4f}")

    # Calculate thresholds for each direction
    for direction, is_horizontal in [("left", True), ("right", True), ("up", False), ("down", False)]:
        try:
            result = process_direction_data(
                calibration_data, direction, channel_norm_factors, baselines, alpha
            )

            # Check if we got valid results
            if any(data is None or (isinstance(data, np.ndarray) and len(data) == 0)
                   for data in result):
                print(f"No valid data for {direction}")
                thresholds[direction] = 0.2 if not is_horizontal else 0.1
                continue

            H, V, V_compensated = result

            # Use the appropriate signal based on direction type
            signals = H if is_horizontal else V_compensated

            # Check if we have a valid signals array
            if not isinstance(signals, np.ndarray) or len(signals) == 0:
                thresholds[direction] = 0.2 if not is_horizontal else 0.1
                continue

            # Calculate threshold
            abs_signals = np.abs(signals)
            if len(abs_signals) > 0:
                threshold = np.percentile(abs_signals, 85)
                min_threshold = 0.02 if not is_horizontal else 0.01
                threshold = max(threshold, min_threshold)
                thresholds[direction] = threshold
                print(f"{direction} threshold: {threshold:.4f}")
            else:
                thresholds[direction] = 0.02 if not is_horizontal else 0.01
        except Exception as e:
            print(f"Error calculating {direction} threshold: {str(e)}")
            import traceback
            traceback.print_exc()
            thresholds[direction] = 0.2 if not is_horizontal else 0.1

    return thresholds, alpha

def plot_direction_signals(calibration_data, direction, channel_norm_factors, baselines, threshold, alpha=0.0, is_horizontal=True):
    """Plot signals for a specific direction with thresholds"""
    try:
        plt.figure(figsize=(10, 6))

        # Process data for this direction
        H, V, V_compensated = process_direction_data(
            calibration_data, direction, channel_norm_factors, baselines, alpha
        )

        if H is None:
            return

        # Determine which signal to plot
        signal = H if is_horizontal else V
        compensated_signal = None if is_horizontal else V_compensated
        signal_label = "H" if is_horizontal else "V"

        # Get all valid steps for plotting individual steps
        ch1_key = "ch1" if is_horizontal else "ch8"
        ch2_key = "ch3" if is_horizontal else "ch2"

        valid_ch1 = [step for step in calibration_data[direction][ch1_key] if is_valid_step(step)]
        valid_ch2 = [step for step in calibration_data[direction][ch2_key] if is_valid_step(step)]
        valid_steps = min(len(valid_ch1), len(valid_ch2))

        for step_index in range(min(valid_steps, 3)):  # Limit to 3 steps for clarity
            try:
                ch1_data = np.array(valid_ch1[step_index])
                ch2_data = np.array(valid_ch2[step_index])

                ch1_data = process_signal(ch1_data, FS, ch1_key) / channel_norm_factors[ch1_key]
                ch2_data = process_signal(ch2_data, FS, ch2_key) / channel_norm_factors[ch2_key]

                step_signal = (ch1_data - ch2_data) - (baselines["H"] if is_horizontal else baselines["V"])

                if not is_horizontal:
                    # For vertical signals, we need H for compensation
                    step_ch3 = np.array([s for s in calibration_data[direction]["ch3"] if is_valid_step(s)][step_index])
                    step_ch4 = np.array([s for s in calibration_data[direction]["ch1"] if is_valid_step(s)][step_index])

                    step_ch3 = process_signal(step_ch3, FS, "ch3") / channel_norm_factors["ch3"]
                    step_ch4 = process_signal(step_ch4, FS, "ch1") / channel_norm_factors["ch1"]

                    step_H = (step_ch4 - step_ch3) - baselines["H"]
                    step_signal = step_signal - alpha * step_H

                t = np.arange(len(step_signal)) / FS
                plt.plot(t, step_signal, label=f"Step {step_index + 1}")
            except Exception as e:
                print(f"Error plotting {direction} step {step_index}: {e}")
                continue

        if valid_steps > 0:
            color = 'r' if direction in ["left", "up"] else 'g'
            plt.axhline(y=threshold, color=color, linestyle='--', label=f'Threshold={threshold:.2f}')
            plt.axhline(y=-threshold, color=color, linestyle='--')

            title = f"{direction.capitalize()} {signal_label} Signals"
            if not is_horizontal:
                title += f" (with α={alpha:.2f} compensation)"

            plt.title(title)
            plt.xlabel("Time (s)")
            plt.ylabel("Amplitude")
            plt.legend()
            plt.tight_layout()

            plot_dir = os.path.join(RESULTS_DIR, "calibration_plots")
            os.makedirs(plot_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = f"{direction}_{signal_label}_signals"
            if not is_horizontal:
                filename += "_compensated"
            filename += f"_{timestamp}.png"

            plot_path = os.path.join(plot_dir, filename)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Saved {direction} {signal_label} signals plot to {plot_path}")

        plt.close()
    except Exception as e:
        print(f"Error plotting {direction} signals: {e}")

def plot_calibration_signals(calibration_data, channel_norm_factors, baselines, thresholds, alpha=0.0):
    """Plot filtered signals with thresholds for debugging"""
    if not DEBUG_PLOTS:
        return

    try:
        # Plot H signals for left and right directions
        for direction in ["left", "right"]:
            plot_direction_signals(
                calibration_data, direction, channel_norm_factors, baselines,
                thresholds[direction], alpha, is_horizontal=True
            )

        # Plot V signals for up and down directions
        for direction in ["up", "down"]:
            plot_direction_signals(
                calibration_data, direction, channel_norm_factors, baselines,
                thresholds[direction], alpha, is_horizontal=False
            )
    except Exception as e:
        print(f"Could not display/save signal plots: {e}")