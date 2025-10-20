# -*- coding: utf-8 -*-
"""
Combined static jump target task + EOG LSL detection + per-step accuracy scoring.
- Pygame task: blue dot jumps through a fixed sequence every STEP_DURATION seconds.
- EOG thread: continuously reads LSL Explore_8441_ExG, filters H/V channels, detects saccades.
- After each jump, we open a RESPONSE_WINDOW and take the first valid detection to score.
- "Center" steps are scored as correct if NO valid saccade is detected in the window.
Outputs:
- On-screen overlays with step info, latest detection, running accuracy.
- CSV: eog_trial_results.csv with per-step results and summary.
Author: you + ChatGPT
Date: 2025-09-11
"""
import pygame
import time
import threading
import collections
import csv
import os
import numpy as np
import pywt
from pylsl import StreamInlet, resolve_byprop
from scipy import signal as sig
import matplotlib.pyplot as plt
from datetime import datetime
from dataclasses import dataclass

# ==============================
# -------- Settings ------------
# ==============================
# --- Pygame / task ---
WIDTH, HEIGHT = 1900, 1000
BG_COLOR = (220, 220, 220)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
STEP_DURATION = 5.0        # seconds between target jumps
RESPONSE_WINDOW = 3         # seconds after each jump during which we accept the first valid detection
DOT_RADIUS_ACTIVE = 20
DOT_RADIUS_STATIC = 10
CENTER_CROSS = 10
center_pos = [WIDTH // 2, HEIGHT // 2]

# --- LSL / signal processing ---
FS = 250
PLOT_BUFFER_DURATION = 5
PLOT_MAX_SAMPLES = FS * PLOT_BUFFER_DURATION
DETECT_WINDOW_DURATION = 1
DETECT_MAX_SAMPLES = FS * DETECT_WINDOW_DURATION
TOTAL_CHANNELS = 8
CHANNEL_INDICES = [0, 1, 2, 7]  # (we'll form H = ch1 - ch3, V = ch8 - ch2)
LOWCUT = 0.4
HIGHCUT = 60
FILTER_ORDER = 4
MERGE_WINDOW = int(0.12 * FS)  # samples
GLOBAL_COOLDOWN = 0.8  # seconds between ANY two accepted detections
LSL_STREAM_NAME = 'Explore_8441_ExG'


# Debug flags
DEBUG_SIGNALS = True
DEBUG_DETECTION = True
DEBUG_PLOTS = True

# ==============================
# ------- Helper Functions -----
# ==============================

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return sig.butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return sig.lfilter(b, a, data)

def notch_filter(data, fs, freq=50, bandwidth=5):
    w0 = freq / (fs/2)
    Q = freq / bandwidth
    b, a = sig.iirnotch(w0, Q)
    return sig.lfilter(b, a, data)

def process_signal(data, fs, channel_type):
    """Apply notch and bandpass filters to signal data"""
    data = notch_filter(data, fs)
    data = bandpass_filter(data, LOWCUT, HIGHCUT, fs, FILTER_ORDER)
    return data

def wait_for_spacebar(window, font, message="Press SPACEBAR to continue"):
    """Display message and wait for SPACEBAR press"""
    window.fill(BG_COLOR)
    instruction_surf = font.render(message, True, BLACK)
    window.blit(instruction_surf, (WIDTH // 2 - instruction_surf.get_width() // 2, HEIGHT // 2))
    pygame.display.flip()

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
        pygame.time.delay(100)
    return True

@dataclass
class Detection:
    ts: float
    direction: str
    is_horizontal: bool
    h_value: float = 0.0
    v_value: float = 0.0
    h_velocity: float = 0.0
    v_velocity: float = 0.0

# ==============================
# ------- Signal Processing -----
# ==============================

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
            signals = ch8 - ch2

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
    thresholds = {}
    for direction, channel in [("left", "H"), ("right", "H"), ("up", "V"), ("down", "V")]:
        try:
            ch1 = np.array(calibration_data[direction]["ch1"])
            ch3 = np.array(calibration_data[direction]["ch3"])
            ch8 = np.array(calibration_data[direction]["ch8"])
            ch2 = np.array(calibration_data[direction]["ch2"])

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

            # Compute H or V and apply baseline correction
            if channel == "H":
                signals = (ch1 - ch3) - baselines["H"]
            else:  # V
                signals = (ch8 - ch2) - baselines["V"]

            # Use 95th percentile as threshold (more robust for EOG)
            abs_signals = np.abs(signals)
            if len(abs_signals) > 0:
                threshold = np.percentile(abs_signals, 95)  # 95th percentile
                threshold = max(threshold, 0.05)  # Minimum threshold of 0.05
                thresholds[direction] = threshold
                print(f"{direction} {channel} threshold: {threshold:.2f}")
            else:
                thresholds[direction] = 0.05  # Default threshold

        except Exception as e:
            print(f"Error calculating {direction} {channel} threshold: {str(e)}")
            thresholds[direction] = 0.05  # Default threshold

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

def plot_detection_window(eog_reader, step_index, target_name, expected_direction, detection=None, calibration_params=None):
    """
    Plot and save the detection window for a specific step
    This plots the entire detection window for every step, regardless of whether a movement was detected
    """
    if not DEBUG_PLOTS:
        return

    try:
        # Create directory for plots if it doesn't exist
        plot_dir = "detection_plots"
        os.makedirs(plot_dir, exist_ok=True)

        # Get current timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get the latest signals from the EOG reader
        times = np.array(eog_reader.latest_times)
        H = np.array(eog_reader.latest_H)
        V = np.array(eog_reader.latest_V)

        # Ensure all arrays have the same length
        min_length = min(len(times), len(H), len(V))
        if min_length == 0:
            print(f"No signal data available for step {step_index+1} ({target_name})")
            return

        times = times[-min_length:]
        H = H[-min_length:]
        V = V[-min_length:]

        # Create a figure
        plt.figure(figsize=(12, 8))

        # Plot H signal
        plt.subplot(2, 1, 1)
        plt.plot(times, H, label="H Signal", color='blue')
        plt.axhline(y=calibration_params['thresholds']['right'], color='green', linestyle='--',
                    label=f'Right Threshold={calibration_params["thresholds"]["right"]:.2f}')
        plt.axhline(y=-calibration_params['thresholds']['left'], color='red', linestyle='--',
                    label=f'Left Threshold={calibration_params["thresholds"]["left"]:.2f}')
        plt.title(f"Horizontal Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()

        # Plot V signal
        plt.subplot(2, 1, 2)
        plt.plot(times, V, label="V Signal", color='purple')
        plt.axhline(y=calibration_params['thresholds']['down'], color='green', linestyle='--',
                    label=f'Down Threshold={calibration_params["thresholds"]["down"]:.2f}')
        plt.axhline(y=-calibration_params['thresholds']['up'], color='red', linestyle='--',
                    label=f'Up Threshold={calibration_params["thresholds"]["up"]:.2f}')
        plt.title(f"Vertical Signal")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()

        # Add detection markers if available
        if detection:
            detection_time = detection.ts
            if detection.is_horizontal:
                plt.subplot(2, 1, 1)
                plt.axvline(x=detection_time, color='black', linestyle=':',
                           label=f'Detection: {detection.direction} at {detection_time:.2f}s')
                plt.legend()
            else:
                plt.subplot(2, 1, 2)
                plt.axvline(x=detection_time, color='black', linestyle=':',
                           label=f'Detection: {detection.direction} at {detection_time:.2f}s')
                plt.legend()

        # Create metadata string
        expected_dir = ""
        if expected_direction["expected_h"]:
            expected_dir = expected_direction["expected_h"]
        elif expected_direction["expected_v"]:
            expected_dir = expected_direction["expected_v"]
        else:
            expected_dir = "center (no movement)"

        detection_status = "No detection" if not detection else f"Detected: {detection.direction}"

        # Add comprehensive title
        plt.suptitle(f"Step {step_index+1}: {target_name}\nExpected: {expected_dir} | {detection_status}")

        # Adjust layout
        plt.tight_layout()

        # Create filename based on target direction
        direction_category = "center"
        if expected_direction["expected_h"] == "left" or expected_direction["expected_h"] == "right":
            direction_category = "horizontal"
        elif expected_direction["expected_v"] == "up" or expected_direction["expected_v"] == "down":
            direction_category = "vertical"

        # Create subdirectory for direction category
        direction_dir = os.path.join(plot_dir, direction_category)
        os.makedirs(direction_dir, exist_ok=True)

        # Save the plot with metadata in filename
        plot_filename = f"step{step_index+1:03d}_{target_name}_{detection_status.replace(' ', '_')}_{timestamp}.png"
        plot_path = os.path.join(direction_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        print(f"Saved detection plot: {plot_path}")

        # Also save a CSV with metadata
        metadata = {
            "step_index": step_index,
            "target_name": target_name,
            "expected_direction": expected_dir,
            "detection_status": detection_status,
            "timestamp": timestamp,
            "plot_path": plot_path,
            "h_threshold_left": calibration_params['thresholds']['left'],
            "h_threshold_right": calibration_params['thresholds']['right'],
            "v_threshold_up": calibration_params['thresholds']['up'],
            "v_threshold_down": calibration_params['thresholds']['down'],
            "h_max": np.max(np.abs(H)) if len(H) > 0 else 0,
            "v_max": np.max(np.abs(V)) if len(V) > 0 else 0,
            "h_mean": np.mean(H) if len(H) > 0 else 0,
            "v_mean": np.mean(V) if len(V) > 0 else 0
        }

        # Save metadata to CSV
        metadata_file = os.path.join(plot_dir, "detection_metadata.csv")
        file_exists = os.path.isfile(metadata_file)

        with open(metadata_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metadata.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metadata)

        # Close the figure
        plt.close()

    except Exception as e:
        print(f"Could not save detection plot for step {step_index+1}: {str(e)}")
        import traceback
        traceback.print_exc()

def run_calibration(eog_reader, window, font, calibration_sequence, clock):
    """Run calibration to determine baselines and thresholds"""
    # Data structure to store raw signals for each direction
    calibration_data = {
        "left": {"ch1": [], "ch2": [], "ch3": [], "ch8": []},
        "right": {"ch1": [], "ch2": [], "ch3": [], "ch8": []},
        "up": {"ch1": [], "ch2": [], "ch3": [], "ch8": []},
        "down": {"ch1": [], "ch2": [], "ch3": [], "ch8": []},
        "center": {"ch1": [], "ch2": [], "ch3": [], "ch8": []}
    }

    # Clear the detection queue
    eog_reader.out_queue.clear()

    # Instructions for calibration
    if not wait_for_spacebar(window, font, "Press SPACEBAR to begin calibration..."):
        return {
            "baselines": {"H": 0, "V": 0},
            "thresholds": {"left": 0.1, "right": 0.1, "up": 0.1, "down": 0.1},
            "channel_norm_factors": {"ch1": 1, "ch2": 1, "ch3": 1, "ch8": 1}
        }

    # Record data for each target
    for i, (target_name, pos) in enumerate(calibration_sequence):
        target_key = target_name.lower()

        # Add a rest step every 8 targets
        if i > 0 and i % 8 == 0:
            window.fill(BG_COLOR)
            rest_surf = font.render("Rest your eyes. Press SPACEBAR to continue.", True, BLACK)
            window.blit(rest_surf, (WIDTH // 2 - rest_surf.get_width() // 2, HEIGHT // 2))
            pygame.display.flip()

            if not wait_for_spacebar(window, font, "Rest your eyes. Press SPACEBAR to continue..."):
                return {
                    "baselines": {"H": 0, "V": 0},
                    "thresholds": {"left": 0.1, "right": 0.1, "up": 0.1, "down": 0.1},
                    "channel_norm_factors": {"ch1": 1, "ch2": 1, "ch3": 1, "ch8": 1}
                }

        # Show the target
        window.fill(BG_COLOR)
        for name, p in calibration_sequence:
            pygame.draw.circle(window, RED, p, DOT_RADIUS_STATIC)
        pygame.draw.circle(window, BLUE, pos, DOT_RADIUS_ACTIVE)
        instruction = f"Looking at {target_name}..."
        surf = font.render(instruction, True, BLACK)
        window.blit(surf, (10, 50))
        pygame.display.flip()

        # Record EOG data for 3 seconds
        start_time = time.time()
        end_time = start_time + 3.0
        target_ch1, target_ch2, target_ch3, target_ch8 = [], [], [], []

        print(f"Recording {target_name} for 3 seconds...")
        sample_count = 0
        while time.time() < end_time:
            sample, _ = eog_reader.inlet.pull_sample(timeout=0.01)
            if sample is not None:
                sample_count += 1
                target_ch1.append(sample[0])
                target_ch2.append(sample[1])
                target_ch3.append(sample[2])
                target_ch8.append(sample[7])
            pygame.event.pump()
            clock.tick(60)

        print(f"Recorded {len(target_ch1)} samples for {target_name}")

        # Store the raw channels for this target
        if target_ch1 and target_ch2 and target_ch3 and target_ch8:
            calibration_data[target_key]["ch1"].extend(target_ch1)
            calibration_data[target_key]["ch2"].extend(target_ch2)
            calibration_data[target_key]["ch3"].extend(target_ch3)
            calibration_data[target_key]["ch8"].extend(target_ch8)
        else:
            print(f"WARNING: No samples collected for {target_name}!")

    # Calculate channel normalization factors
    channel_norm_factors = {
        "ch1": calculate_channel_norm_factor(calibration_data, "ch1"),
        "ch2": calculate_channel_norm_factor(calibration_data, "ch2"),
        "ch3": calculate_channel_norm_factor(calibration_data, "ch3"),
        "ch8": calculate_channel_norm_factor(calibration_data, "ch8")
    }

    print(f"\nChannel normalization factors: "
          f"ch1={channel_norm_factors['ch1']:.2f}, ch2={channel_norm_factors['ch2']:.2f}, "
          f"ch3={channel_norm_factors['ch3']:.2f}, ch8={channel_norm_factors['ch8']:.2f}")

    # Calculate baselines from normalized signals
    H_baseline = calculate_normalized_baseline(calibration_data, channel_norm_factors, "H")
    V_baseline = calculate_normalized_baseline(calibration_data, channel_norm_factors, "V")
    baselines = {"H": H_baseline, "V": V_baseline}

    # Calculate direction-specific thresholds
    thresholds = calculate_direction_thresholds(calibration_data, channel_norm_factors, baselines)

    # Format thresholds for return
    formatted_thresholds = {
        "left": thresholds.get("left", 0.05),
        "right": thresholds.get("right", 0.05),
        "up": thresholds.get("up", 0.05),
        "down": thresholds.get("down", 0.05)
    }

    # Plot calibration signals
    plot_calibration_signals(calibration_data, channel_norm_factors, baselines, formatted_thresholds)

    return {
        "baselines": baselines,
        "thresholds": formatted_thresholds,
        "channel_norm_factors": channel_norm_factors
    }
# ==============================
# ------- EOG Reader Class -----
# ==============================

class EOGReader(threading.Thread):
    def __init__(self, out_queue, max_queue=50, calibration_params=None):
        super().__init__()
        self.out_queue = out_queue
        self.max_queue = max_queue
        self.calibration_params = calibration_params or {
            "baselines": {"H": 0, "V": 0},
            "thresholds": {"left": 0.3, "right": 0.3, "up": 0.3, "down": 0.3},
            "channel_norm_factors": {"ch1": 1, "ch2": 1, "ch3": 1, "ch8": 1}
        }
        self.running = True

        # Buffers for plotting and detection
        self.channel_buffers = [collections.deque(maxlen=PLOT_MAX_SAMPLES) for _ in range(TOTAL_CHANNELS)]
        self.time_buffer = collections.deque(maxlen=PLOT_MAX_SAMPLES)
        self.detect_channel_buffers = [collections.deque(maxlen=DETECT_MAX_SAMPLES) for _ in range(TOTAL_CHANNELS)]
        self.detect_time_buffer = collections.deque(maxlen=DETECT_MAX_SAMPLES)

        self.last_detection_time = 0.0
        self.last_any_movement_time = -1e9

        print("Looking for LSL stream...")
        streams = resolve_byprop('name', LSL_STREAM_NAME, timeout=5.0)
        if not streams:
            raise RuntimeError(f"{LSL_STREAM_NAME} stream not found")
        self.inlet = StreamInlet(streams[0])
        print("Connected to stream.")
        self.start_time = time.time()

        # For plotting and debugging
        self.latest_H = np.array([])
        self.latest_V = np.array([])
        self.latest_times = np.array([])
        self.debug_counter = 0

    def _push(self, det: Detection):
        """Push detection to queue with cooldown check"""
        current_time = time.time() - self.start_time
        if (current_time - self.last_any_movement_time) >= GLOBAL_COOLDOWN:
            self.out_queue.append(det)
            while len(self.out_queue) > self.max_queue:
                self.out_queue.popleft()
            self.last_any_movement_time = current_time
            return True
        return False

    def process_detection_window(self):
        """Process the detection window and check for saccades"""
        try:
            # Use the detection window buffers
            times = np.array(self.detect_time_buffer)
            ch1 = np.array(self.detect_channel_buffers[0])
            ch2 = np.array(self.detect_channel_buffers[1])
            ch3 = np.array(self.detect_channel_buffers[2])
            ch8 = np.array(self.detect_channel_buffers[7])

            # Ensure all arrays have the same length
            min_length = min(len(times), len(ch1), len(ch2), len(ch3), len(ch8))
            if min_length == 0:
                return

            times = times[-min_length:]
            ch1 = ch1[-min_length:]
            ch2 = ch2[-min_length:]
            ch3 = ch3[-min_length:]
            ch8 = ch8[-min_length:]

            # Apply filters
            ch1 = process_signal(ch1, FS, "ch1")
            ch2 = process_signal(ch2, FS, "ch2")
            ch3 = process_signal(ch3, FS, "ch3")
            ch8 = process_signal(ch8, FS, "ch8")

            # Normalize channels, calculate H and V, etc.


            # Normalize channels
            ch1 = ch1 / self.calibration_params["channel_norm_factors"]["ch1"]
            ch2 = ch2 / self.calibration_params["channel_norm_factors"]["ch2"]
            ch3 = ch3 / self.calibration_params["channel_norm_factors"]["ch3"]
            ch8 = ch8 / self.calibration_params["channel_norm_factors"]["ch8"]

            # Calculate H and V
            H = ch1 - ch3
            V = ch8 - ch2

            # Apply baseline correction
            H_corrected = H - self.calibration_params["baselines"]["H"]
            V_corrected = V - self.calibration_params["baselines"]["V"]

            # Update latest signals for plotting
            self.latest_H = H_corrected
            self.latest_V = V_corrected
            self.latest_times = np.array(self.time_buffer)

            # Debug output
            self.debug_counter += 1
            if DEBUG_SIGNALS and self.debug_counter % 10 == 0:
                print(f"\nSignal Stats:")
                print(f"H: mean={np.mean(H_corrected):.4f}, std={np.std(H_corrected):.4f}, max={np.max(np.abs(H_corrected)):.4f}")
                print(f"V: mean={np.mean(V_corrected):.4f}, std={np.std(V_corrected):.4f}, max={np.max(np.abs(V_corrected)):.4f}")

            # Define velocity thresholds
            H_VELOCITY_THRESHOLD = 0.1
            V_VELOCITY_THRESHOLD = 0.1

            # Calculate velocity (derivative)
            H_velocity = np.gradient(H_corrected)
            V_velocity = np.gradient(V_corrected)

            # Check each sample for threshold crossing and velocity
            now = time.time() - self.start_time
            for idx in range(1, len(H_corrected)):
                h_val = H_corrected[idx]
                v_val = V_corrected[idx]
                h_vel = abs(H_velocity[idx])
                v_vel = abs(V_velocity[idx])

                # Check for horizontal movements with velocity
                if h_val > self.calibration_params["thresholds"]["right"] and h_vel > H_VELOCITY_THRESHOLD:
                    if DEBUG_DETECTION and self.debug_counter % 10 == 0:
                        print(f"Right detection at sample {idx}: H={h_val:.4f}, vel={h_vel:.4f}")
                    det = Detection(
                        ts=times[idx],
                        direction='right',
                        is_horizontal=True,
                        h_value=h_val,
                        v_value=v_val,
                        h_velocity=h_vel,
                        v_velocity=v_vel
                    )
                    if self._push(det) and DEBUG_DETECTION:
                        print(f"Pushed right detection to queue at {times[idx]:.2f}s")

                elif h_val < -self.calibration_params["thresholds"]["left"] and h_vel > H_VELOCITY_THRESHOLD:
                    if DEBUG_DETECTION and self.debug_counter % 10 == 0:
                        print(f"Left detection at sample {idx}: H={h_val:.4f}, vel={h_vel:.4f}")
                    det = Detection(
                        ts=times[idx],
                        direction='left',
                        is_horizontal=True,
                        h_value=h_val,
                        v_value=v_val,
                        h_velocity=h_vel,
                        v_velocity=v_vel
                    )
                    if self._push(det) and DEBUG_DETECTION:
                        print(f"Pushed left detection to queue at {times[idx]:.2f}s")

                # Check for vertical movements with velocity
                if v_val > self.calibration_params["thresholds"]["down"] and v_vel > V_VELOCITY_THRESHOLD:
                    if DEBUG_DETECTION and self.debug_counter % 10 == 0:
                        print(f"Down detection at sample {idx}: V={v_val:.4f}, vel={v_vel:.4f}")
                    det = Detection(
                        ts=times[idx],
                        direction='down',
                        is_horizontal=False,
                        h_value=h_val,
                        v_value=v_val,
                        h_velocity=h_vel,
                        v_velocity=v_vel
                    )
                    if self._push(det) and DEBUG_DETECTION:
                        print(f"Pushed down detection to queue at {times[idx]:.2f}s")

                elif v_val < -self.calibration_params["thresholds"]["up"] and v_vel > V_VELOCITY_THRESHOLD:
                    if DEBUG_DETECTION and self.debug_counter % 10 == 0:
                        print(f"Up detection at sample {idx}: V={v_val:.4f}, vel={v_vel:.4f}")
                    det = Detection(
                        ts=times[idx],
                        direction='up',
                        is_horizontal=False,
                        h_value=h_val,
                        v_value=v_val,
                        h_velocity=h_vel,
                        v_velocity=v_vel
                    )
                    if self._push(det) and DEBUG_DETECTION:
                        print(f"Pushed up detection to queue at {times[idx]:.2f}s")

        except Exception as e:
            print(f"Error in detection processing: {str(e)}")
            import traceback
            traceback.print_exc()

    def run(self):
        """Main thread loop for reading and processing EOG data"""
        DETECT_PERIOD = 0.1  # seconds between running detection on the buffer
        last_detection_check = time.time()

        while self.running:
            sample, timestamp = self.inlet.pull_sample(timeout=0.05)
            if sample is None:
                continue

            now = time.time() - self.start_time

            # Add to both buffers
            self.time_buffer.append(now)
            self.detect_time_buffer.append(now)

            for i in range(TOTAL_CHANNELS):
                self.channel_buffers[i].append(sample[i])
                self.detect_channel_buffers[i].append(sample[i])

            # Run detection every 0.1 seconds
            current_time = time.time()
            if (current_time - last_detection_check) >= DETECT_PERIOD and len(self.detect_time_buffer) >= DETECT_MAX_SAMPLES:
                last_detection_check = current_time
                self.process_detection_window()

    def stop(self):
        """Stop the thread"""
        self.running = False
        self.join()

# ==============================
# ------- Main Application ------
# ==============================

def expected_from_name(name: str):
    """Return expected H and V directions for a given target name"""
    name = name.lower()
    if name == "left":
        return {"expected_h": "left", "expected_v": None}
    elif name == "right":
        return {"expected_h": "right", "expected_v": None}
    elif name == "up":
        return {"expected_h": None, "expected_v": "up"}
    elif name == "down":
        return {"expected_h": None, "expected_v": "down"}
    else:  # center
        return {"expected_h": None, "expected_v": None}

def save_results(trials, calibration_params, out_path="eog_trial_results.csv"):
    """Save trial results to CSV file"""
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "step_index", "target_name", "expected_h", "expected_v",
            "detected_h", "detected_v", "ts_detected_h", "ts_detected_v",
            "correct", "h_value_h", "v_value_h", "h_value_v", "v_value_v",
            "h_velocity_h", "v_velocity_h", "h_velocity_v", "v_velocity_v",
            "h_value_max", "h_value_min", "v_value_max", "v_value_min",
            "h_threshold_left", "h_threshold_right", "v_threshold_up", "v_threshold_down"
        ])
        writer.writeheader()

        for row in trials:
            clean_row = {
                "step_index": row.get("step_index", ""),
                "target_name": row.get("target_name", ""),
                "expected_h": row.get("expected_h", ""),
                "expected_v": row.get("expected_v", ""),
                "detected_h": row.get("detected_h", "") if row.get("detected_h") is not None else "",
                "detected_v": row.get("detected_v", "") if row.get("detected_v") is not None else "",
                "ts_detected_h": row.get("ts_detected_h", "") if row.get("ts_detected_h") is not None else "",
                "ts_detected_v": row.get("ts_detected_v", "") if row.get("ts_detected_v") is not None else "",
                "correct": row.get("correct", ""),
                "h_value_h": row.get("h_value_h", 0),
                "v_value_h": row.get("v_value_h", 0),
                "h_value_v": row.get("h_value_v", 0),
                "v_value_v": row.get("v_value_v", 0),
                "h_velocity_h": row.get("h_velocity_h", 0),
                "v_velocity_h": row.get("v_velocity_h", 0),
                "h_velocity_v": row.get("h_velocity_v", 0),
                "v_velocity_v": row.get("v_velocity_v", 0),
                "h_value_max": row.get("h_value_max", 0),
                "h_value_min": row.get("h_value_min", 0),
                "v_value_max": row.get("v_value_max", 0),
                "v_value_min": row.get("v_value_min", 0),
                "h_threshold_left": row.get("h_threshold_left", 0),
                "h_threshold_right": row.get("h_threshold_right", 0),
                "v_threshold_up": row.get("v_threshold_up", 0),
                "v_threshold_down": row.get("v_threshold_down", 0)
            }
            writer.writerow(clean_row)

        # Summary row
        total = len(trials)
        correct = sum(1 for r in trials if r["correct"])
        writer.writerow({})
        writer.writerow({
            "target_name": "SUMMARY",
            "expected_h": f"{correct}/{total} ({(correct/total*100.0 if total else 0.0):.1f}%)",
        })

        # Thresholds row
        writer.writerow({
            "target_name": "THRESHOLDS",
            "expected_h": f"Left: {calibration_params['thresholds']['left']:.4f}, Right: {calibration_params['thresholds']['right']:.4f}",
            "expected_v": f"Up: {calibration_params['thresholds']['up']:.4f}, Down: {calibration_params['thresholds']['down']:.4f}",
        })

    print(f"Saved results to {out_path}")

def main():
    """Main application function"""
    # Initialize Pygame
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Static Jumps + EOG Accuracy Test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)

    # Define target sequences
    center_pos = [WIDTH // 2, HEIGHT // 2]
    sequence = [
        ("center", center_pos),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("center", center_pos),
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),
        ("center", center_pos),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
    ]

    calibration_sequence = [
        ("center", center_pos),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("center", center_pos),
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),
        ("center", center_pos),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("center", center_pos),
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),
        ("center", center_pos),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("center", center_pos),
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)])
    ]

    # Initialize EOG reader
    det_queue = collections.deque(maxlen=50)
    eog = EOGReader(det_queue)
    eog.start()

    # Run calibration
    calibration_params = run_calibration(eog, window, font, calibration_sequence, clock)
    eog.calibration_params = calibration_params

    print(f"\nCalibration complete:")
    print(f"Baselines: {calibration_params['baselines']}")
    print(f"Thresholds: {calibration_params['thresholds']}")
    print(f"Channel norm factors: {calibration_params['channel_norm_factors']}")

    # Wait for user to start test
    if not wait_for_spacebar(window, font, "Calibration complete! Press SPACEBAR to start the test..."):
        return

    # Initialize task state
    step_index = 0
    dot_pos = sequence[0][1]
    step_start = time.time()
    step_max_h = -float('inf')
    step_min_h = float('inf')
    step_max_v = -float('inf')
    step_min_v = float('inf')

    # Initialize scoring
    trials = []
    running_correct = 0
    running_total = 0
    step_captured = False
    current_expected = expected_from_name(sequence[step_index][0])
    running = True

    step_detection = []

    try:
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            now = time.time()

            # Track max and min H/V values
            if len(eog.latest_H) > 0 and len(eog.latest_V) > 0:
                current_max_h = np.max(np.abs(eog.latest_H))
                current_min_h = np.min(eog.latest_H)
                current_max_v = np.max(np.abs(eog.latest_V))
                current_min_v = np.min(eog.latest_V)

                step_max_h = max(step_max_h, current_max_h)
                step_min_h = min(step_min_h, current_min_h)
                step_max_v = max(step_max_v, current_max_v)
                step_min_v = min(step_min_v, current_min_v)

            # Handle step advancement
            if (now - step_start) >= STEP_DURATION:
                # Finalize scoring for previous step if no detection captured
                # Plot the detection window for this step before advancing
                # This ensures we plot the window for every step, regardless of detection
                plot_detection_window(
                    eog_reader=eog,
                    step_index=step_index,
                    target_name=sequence[step_index][0],
                    expected_direction=current_expected,
                    detection=None,  # No detection for this step
                    calibration_params=calibration_params
                )

                if not step_captured:
                    is_correct = (current_expected["expected_h"] is None and current_expected["expected_v"] is None)
                    running_total += 1
                    running_correct += int(is_correct)

                    trials.append({
                        "step_index": step_index,
                        "target_name": sequence[step_index][0],
                        "expected_h": current_expected["expected_h"],
                        "expected_v": current_expected["expected_v"],
                        "detected_h": None,
                        "detected_v": None,
                        "ts_detected_h": None,
                        "ts_detected_v": None,
                        "correct": is_correct,
                        "h_value_h": 0,
                        "v_value_h": 0,
                        "h_value_v": 0,
                        "v_value_v": 0,
                        "h_velocity_h": 0,
                        "v_velocity_h": 0,
                        "h_velocity_v": 0,
                        "v_velocity_v": 0,
                        "h_value_max": step_max_h,
                        "h_value_min": step_min_h,
                        "v_value_max": step_max_v,
                        "v_value_min": step_min_v,
                        "h_threshold_left": calibration_params['thresholds']['left'],
                        "h_threshold_right": calibration_params['thresholds']['right'],
                        "v_threshold_up": calibration_params['thresholds']['up'],
                        "v_threshold_down": calibration_params['thresholds']['down']
                    })

                # Advance to next step
                step_index += 1
                if step_index >= len(sequence):
                    running = False
                    break

                dot_pos = sequence[step_index][1]
                step_start = now
                step_captured = False
                current_expected = expected_from_name(sequence[step_index][0])
                step_max_h = -float('inf')
                step_min_h = float('inf')
                step_max_v = -float('inf')
                step_min_v = float('inf')

            # Handle detection window
            if not step_captured and (now - step_start) <= RESPONSE_WINDOW:
                first_h_det = None
                first_v_det = None

                # Process all detections in the queue
                while det_queue:
                    det = det_queue.popleft()
                    if det.is_horizontal and first_h_det is None:
                        first_h_det = det
                    elif not det.is_horizontal and first_v_det is None:
                        first_v_det = det

                # If we have at least one detection, mark step as captured
                if first_h_det is not None or first_v_det is not None:
                    step_captured = True
                    running_total += 1

                    # Plot the detection window with the detection marked
                    if first_h_det is not None:
                        plot_detection_window(
                            eog_reader=eog,
                            step_index=step_index,
                            target_name=sequence[step_index][0],
                            expected_direction=current_expected,
                            detection=first_h_det,
                            calibration_params=calibration_params
                        )
                    elif first_v_det is not None:
                        plot_detection_window(
                            eog_reader=eog,
                            step_index=step_index,
                            target_name=sequence[step_index][0],
                            expected_direction=current_expected,
                            detection=first_v_det,
                            calibration_params=calibration_params
                        )

                    # Determine if the step is correct
                    is_correct = False

                    # Check horizontal movement if expected
                    if current_expected["expected_h"] is not None and first_h_det is not None:
                        if current_expected["expected_h"] == first_h_det.direction:
                            is_correct = True

                    # Check vertical movement if expected
                    if current_expected["expected_v"] is not None and first_v_det is not None:
                        if current_expected["expected_v"] == first_v_det.direction:
                            is_correct = True

                    # Special case: if we got a detection but none was expected
                    if current_expected["expected_h"] is None and first_h_det is not None:
                        is_correct = False
                    if current_expected["expected_v"] is None and first_v_det is not None:
                        is_correct = False

                    running_correct += int(is_correct)

                    # Log both detections to trials
                    trials.append({
                        "step_index": step_index,
                        "target_name": sequence[step_index][0],
                        "expected_h": current_expected["expected_h"],
                        "expected_v": current_expected["expected_v"],
                        "detected_h": first_h_det.direction if first_h_det is not None else None,
                        "detected_v": first_v_det.direction if first_v_det is not None else None,
                        "ts_detected_h": float(first_h_det.ts) if first_h_det is not None else None,
                        "ts_detected_v": float(first_v_det.ts) if first_v_det is not None else None,
                        "correct": is_correct,
                        "h_value_h": first_h_det.h_value if first_h_det is not None else 0,
                        "v_value_h": first_h_det.v_value if first_h_det is not None else 0,
                        "h_value_v": first_v_det.h_value if first_v_det is not None else 0,
                        "v_value_v": first_v_det.v_value if first_v_det is not None else 0,
                        "h_velocity_h": first_h_det.h_velocity if first_h_det is not None else 0,
                        "v_velocity_h": first_h_det.v_velocity if first_h_det is not None else 0,
                        "h_velocity_v": first_v_det.h_velocity if first_v_det is not None else 0,
                        "v_velocity_v": first_v_det.v_velocity if first_v_det is not None else 0,
                        "h_value_max": step_max_h,
                        "h_value_min": step_min_h,
                        "v_value_max": step_max_v,
                        "v_value_min": step_min_v,
                    })

            # Draw the interface
            window.fill(BG_COLOR)

            # Draw all static red targets
            for name, pos in sequence:
                pygame.draw.circle(window, RED, pos, DOT_RADIUS_STATIC)

            # Draw active blue dot
            pygame.draw.circle(window, BLUE, dot_pos, DOT_RADIUS_ACTIVE)

            # Draw center cross
            cx, cy = WIDTH // 2, HEIGHT // 2
            pygame.draw.line(window, BLACK, (cx - CENTER_CROSS, cy), (cx + CENTER_CROSS, cy), 3)
            pygame.draw.line(window, BLACK, (cx, cy - CENTER_CROSS), (cx, cy + CENTER_CROSS), 3)

            # Draw overlays
            acc = (running_correct / running_total * 100.0) if running_total > 0 else 0.0
            overlay_lines = [
                f"Step {step_index+1}/{len(sequence)} | Target: {sequence[step_index][0]} | "
                f"H det: {first_h_det.direction if 'first_h_det' in locals() and first_h_det is not None else 'None'}, "
                f"V det: {first_v_det.direction if 'first_v_det' in locals() and first_v_det is not None else 'None'}",
                f"Max H: {step_max_h:.2f}, Max V: {step_max_v:.2f}",
                f"Score: {running_correct}/{running_total} ({acc:.1f}%)"
            ]

            y = 10
            for line in overlay_lines:
                surf = font.render(line, True, BLACK)
                window.blit(surf, (10, y))
                y += 32

            pygame.display.flip()
            clock.tick(60)

    finally:
        eog.stop()
        save_results(trials, calibration_params)

        # Display completion message
        window.fill(BG_COLOR)
        completion_surf = font.render("Task complete! Press SPACEBAR to exit.", True, BLACK)
        window.blit(completion_surf, (WIDTH // 2 - completion_surf.get_width() // 2, HEIGHT // 2))
        pygame.display.flip()

        # Wait for SPACEBAR to exit
        wait_for_spacebar(window, font, "Task complete! Press SPACEBAR to exit.")
        pygame.quit()

if __name__ == "__main__":
    main()