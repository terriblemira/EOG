import numpy as np
from signal_processing_wavelet import process_signal
from config import FS
import matplotlib.pyplot as plt
from datetime import datetime
import os
from config import DEBUG_PLOTS

def calculate_normalized_baseline(calibration_data, channel_norm_factors, channel):
    """Calculate baseline from normalized H or V signals during center fixation"""
    center_data = calibration_data["center"]
    if (not center_data["ch1"] or not center_data["ch3"] or
        not center_data["ch8"] or not center_data["ch2"]):
        print(f"Warning: Insufficient center data for {channel} baseline. Using 0.")
        return 0
    try:
        ch1 = np.array(center_data["ch1"])
        ch3 = np.array(center_data["ch3"])
        ch8 = np.array(center_data["ch8"])
        ch2 = np.array(center_data["ch2"])

        # Apply filters
        ch1 = process_signal(ch1, FS, "ch1")
        ch3 = process_signal(ch3, FS, "ch3")
        ch8 = process_signal(ch8, FS, "ch8")
        ch2 = process_signal(ch2, FS, "ch2")

        # Normalize channels
        ch1 = ch1 / channel_norm_factors["ch1"]
        ch3 = ch3 / channel_norm_factors["ch3"]
        ch8 = ch8 / channel_norm_factors["ch8"]
        ch2 = ch2 / channel_norm_factors["ch2"]

        # Compute H or V
        if channel == "H":
            signals = ch1 - ch3
        else:  # V
            signals = ch8 - ch2  # Reversed for vertical

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
        if not calibration_data[direction][channel_name]:
            continue
        try:
            ch_data = np.array(calibration_data[direction][channel_name])
            ch_data = process_signal(ch_data, FS, channel_name)
            amplitudes.append(np.percentile(np.abs(ch_data), 90))
        except Exception as e:
            print(f"Error processing {direction} {channel_name}: {str(e)}")
            continue

    if amplitudes:
        factor = np.median(amplitudes)
        return max(factor, 0.1)  # Ensure minimum factor
    return 1.0

def calculate_direction_thresholds(calibration_data, channel_norm_factors, baselines):
    """Calculate direction-specific thresholds using normalized signals"""
    thresholds = {}

    # Calculate horizontal thresholds
    for direction, channel in [("left", "H"), ("right", "H")]:
        try:
            ch1 = np.array(calibration_data[direction]["ch1"])
            ch3 = np.array(calibration_data[direction]["ch3"])

            # Apply filters
            ch1 = process_signal(ch1, FS, "ch1")
            ch3 = process_signal(ch3, FS, "ch3")

            # Normalize channels
            ch1 = ch1 / channel_norm_factors["ch1"]
            ch3 = ch3 / channel_norm_factors["ch3"]

            # Compute H and apply baseline correction
            signals = (ch1 - ch3) - baselines["H"]

            # Use 95th percentile as threshold
            abs_signals = np.abs(signals)
            if len(abs_signals) > 0:
                threshold = np.percentile(abs_signals, 80)
                threshold = max(threshold, 0.1)
                thresholds[direction] = threshold
                print(f"{direction} threshold: {threshold:.4f}")
            else:
                thresholds[direction] = 0.1

        except Exception as e:
            print(f"Error calculating {direction} threshold: {str(e)}")
            thresholds[direction] = 0.1

    # Calculate vertical thresholds with special handling
    for direction in ["up", "down"]:
        try:
            ch8 = np.array(calibration_data[direction]["ch8"])
            ch2 = np.array(calibration_data[direction]["ch2"])

            # Apply filters
            ch8 = process_signal(ch8, FS, "ch8")
            ch2 = process_signal(ch2, FS, "ch2")

            # Normalize channels
            ch8 = ch8 / channel_norm_factors["ch8"]
            ch2 = ch2 / channel_norm_factors["ch2"]

            # Compute V with reversed calculation and apply baseline correction
            signals = (ch8 - ch2) - baselines["V"]

            # Use 90th percentile for vertical thresholds
            abs_signals = np.abs(signals)
            if len(abs_signals) > 0:
                threshold = np.percentile(abs_signals, 80)
                threshold = max(threshold, 0.2)  # Higher minimum threshold for vertical
                thresholds[direction] = threshold
                print(f"{direction} threshold: {threshold:.4f}")
            else:
                thresholds[direction] = 0.2

        except Exception as e:
            print(f"Error calculating {direction} threshold: {str(e)}")
            thresholds[direction] = 0.2

    return thresholds

def plot_calibration_signals(calibration_data, channel_norm_factors, baselines, thresholds):
    """Plot filtered signals with thresholds for debugging"""
    if not DEBUG_PLOTS:
        return

    try:
        plot_dir = "calibration_plots"
        os.makedirs(plot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for direction in ["left", "right", "up", "down"]:
            if direction in ["left", "right"]:
                ch1_data = np.array(calibration_data[direction]["ch1"])
                ch3_data = np.array(calibration_data[direction]["ch3"])

                ch1_data = process_signal(ch1_data, FS, "ch1")
                ch3_data = process_signal(ch3_data, FS, "ch3")

                # Normalize and apply baseline
                ch1_data = ch1_data / channel_norm_factors["ch1"]
                ch3_data = ch3_data / channel_norm_factors["ch3"]
                H_signal = (ch1_data - ch3_data) - baselines["H"]

                plt.figure(figsize=(10, 4))
                plt.plot(H_signal[:1000], label=f"{direction} H")
                threshold = -thresholds["left"] if direction == "left" else thresholds["right"]
                color = 'r' if direction == "left" else 'g'
                plt.axhline(y=threshold, color=color, linestyle='--',
                           label=f'{direction.capitalize()} Threshold={threshold:.2f}')
                plt.title(f"{direction.capitalize()} H Signal (First 1000 Samples)")
                plt.legend()
                plt.tight_layout()

                plot_path = os.path.join(plot_dir, f"{direction}_H_signal_{timestamp}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Saved {direction} H signal plot to {plot_path}")
                plt.close()

            else:  # up or down
                ch8_data = np.array(calibration_data[direction]["ch8"])
                ch2_data = np.array(calibration_data[direction]["ch2"])

                ch8_data = process_signal(ch8_data, FS, "ch8")
                ch2_data = process_signal(ch2_data, FS, "ch2")

                # Normalize and apply baseline
                ch8_data = ch8_data / channel_norm_factors["ch8"]
                ch2_data = ch2_data / channel_norm_factors["ch2"]
                V_signal = (ch8_data - ch2_data) - baselines["V"]

                plt.figure(figsize=(10, 4))
                plt.plot(V_signal[:1000], label=f"{direction} V")
                threshold = -thresholds["up"] if direction == "up" else thresholds["down"]
                color = 'r' if direction == "up" else 'g'
                plt.axhline(y=threshold, color=color, linestyle='--',
                           label=f'{direction.capitalize()} Threshold={threshold:.2f}')
                plt.title(f"{direction.capitalize()} V Signal (First 1000 Samples)")
                plt.legend()
                plt.tight_layout()

                plot_path = os.path.join(plot_dir, f"{direction}_V_signal_{timestamp}.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                print(f"Saved {direction} V signal plot to {plot_path}")
                plt.close()

    except Exception as e:
        print(f"Could not display/save signal plots: {e}")