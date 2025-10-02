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
from dataclasses import dataclass

import numpy as np
from pylsl import StreamInlet, resolve_byprop
from scipy import signal as sig

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
RESPONSE_WINDOW = 3      # seconds after each jump during which we accept the first valid detection
DOT_RADIUS_ACTIVE = 20
DOT_RADIUS_STATIC = 10
CENTER_CROSS = 10
center_pos = [WIDTH // 2, HEIGHT // 2]

# --- LSL / signal processing ---
FS = 250
BUFFER_DURATION = 5
MAX_SAMPLES = FS * BUFFER_DURATION
TOTAL_CHANNELS = 8
CHANNEL_INDICES = [0, 1, 2, 7]  # (we'll form H = ch1 - ch3, V = ch8 - ch2)

LOWCUT = 0.4
HIGHCUT = 30
FILTER_ORDER = 4

PEAK_DISTANCE = 75  # samples
#H_THRESH = 95
#V_THRESH = 50
MERGE_WINDOW = 500  # samples

MIN_CONFIDENCE = 100
GLOBAL_COOLDOWN = 0.8  # seconds between ANY two accepted detections

LSL_STREAM_NAME = 'Explore_8441_ExG'

clock = pygame.time.Clock()

# ==============================
# ------- Helper code ----------
# ==============================

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return sig.butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return sig.filtfilt(b, a, data)

def notch_filter(data, fs, freq=50, bandwidth=5):
    w0 = freq / (fs/2)  # normalized frequency
    Q = freq / bandwidth
    b, a = sig.iirnotch(w0, Q)
    return sig.filtfilt(b, a, data)

def detect_eye_movements(signal, timestamps, h_thresh, v_thresh, use_highest_peak=False):
    """
    Detects eye movements in EOG signals.
    Only returns movements where H or V exceeds thresholds.
    Direction assignment based on sign:
        Horizontal: H<0 → left, H>0 → right
        Vertical:   V<0 → up,   V>0 → down
    """
    horizontal = signal[:, 0]
    vertical = signal[:, 1]

    raw_movements = []

    # Horizontal peaks above thresholds
    for idx, val in enumerate(horizontal):
        if val > h_thresh:
            raw_movements.append((idx, 'right', val))
        elif val < -h_thresh:
            raw_movements.append((idx, 'left', val))

    # Vertical peaks above thresholds
    for idx, val in enumerate(vertical):
        if val > v_thresh:
            raw_movements.append((idx, 'down', val))
        elif val < -v_thresh:
            raw_movements.append((idx, 'up', val))

    if not raw_movements:
        return []

    # Sort and merge nearby movements
    raw_movements.sort(key=lambda x: x[0])
    filtered_movements = []
    i = 0
    while i < len(raw_movements):
        group = [raw_movements[i]]
        j = i + 1
        while j < len(raw_movements) and raw_movements[j][0] - raw_movements[i][0] <= MERGE_WINDOW:
            group.append(raw_movements[j])
            j += 1
        peak = max(group, key=lambda x: abs(x[2])) if use_highest_peak else group[0]
        filtered_movements.append(
            Detection(
                ts=timestamps[peak[0]],
                direction=peak[1],
                confidence=abs(peak[2]),
                h_value=signal[peak[0], 0],
                v_value=signal[peak[0], 1]
            )
        )
        i = j

    return filtered_movements

def run_calibration(eog_reader, window, font, calibration_sequence):
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
    window.fill(BG_COLOR)
    instruction_surf = font.render("Calibration: Press SPACEBAR to begin.", True, BLACK)
    window.blit(instruction_surf, (WIDTH // 2 - instruction_surf.get_width() // 2, HEIGHT // 2))
    pygame.display.flip()

    # Wait for SPACEBAR to start calibration
    if not wait_for_spacebar(window, font, "Press SPACEBAR to begin calibration..."):
        return 95, 50

    # Step 2: Record raw signals for ALL targets (including center)
    for i, (target_name, pos) in enumerate(calibration_sequence):
        target_key = target_name.lower()

        # Add a rest step every 8 targets
        if i > 0 and i % 8 == 0:
            window.fill(BG_COLOR)
            rest_surf = font.render("Rest your eyes. Press SPACEBAR to continue.", True, BLACK)
            window.blit(rest_surf, (WIDTH // 2 - rest_surf.get_width() // 2, HEIGHT // 2))
            pygame.display.flip()
            if not wait_for_spacebar(window, font):
                return 95, 50

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

    # Verify we collected data
    for direction in calibration_data:
        print(f"{direction}: {len(calibration_data[direction]['ch1'])} samples")

    # Step 3: Calculate thresholds using filtered signals
    def calculate_threshold(direction, channel, default):
        if not calibration_data[direction]["ch1"] or not calibration_data[direction]["ch2"] or not calibration_data[direction]["ch3"] or not calibration_data[direction]["ch8"]:
            print(f"Warning: No signals for {direction}. Using default.")
            return default
        # Get raw channels
        ch1 = np.array(calibration_data[direction]["ch1"])
        ch2 = np.array(calibration_data[direction]["ch2"])
        ch3 = np.array(calibration_data[direction]["ch3"])
        ch8 = np.array(calibration_data[direction]["ch8"])
        # Apply filters
        ch1 = notch_filter(ch1, FS)
        ch2 = notch_filter(ch2, FS)
        ch3 = notch_filter(ch3, FS)
        ch8 = notch_filter(ch8, FS)
        ch1 = bandpass_filter(ch1, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        ch2 = bandpass_filter(ch2, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        ch3 = bandpass_filter(ch3, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        ch8 = bandpass_filter(ch8, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        # Compute H or V
        if channel == "H":
            signals = ch1 - ch3
        else:  # V
            signals = ch8 - ch2
        # Use absolute values and find top 10% peaks
        abs_signals = np.abs(signals)
        if len(abs_signals) == 0:
            print(f"No signals after filtering for {direction} {channel}")
            return default
        sorted_signals = np.sort(abs_signals)[::-1]
        top_count = max(1, int(len(sorted_signals) * 0.1))
        top_signals = sorted_signals[:top_count]
        # Calculate threshold as 0.75 * mean of top signals
        threshold = np.mean(top_signals) * 0.75
        print(f"{direction} {channel} threshold: {threshold:.2f} (top mean: {np.mean(top_signals):.2f})")
        return max(default, threshold)

    # Calculate H_THRESH using ONLY left and right directions (H channel)
    try:
        print("\nCalculating H threshold from left/right H channels:")
        left_h_thresh = calculate_threshold("left", "H", 95)
        right_h_thresh = calculate_threshold("right", "H", 95)
        H_THRESH = min(left_h_thresh, right_h_thresh)
        print(f"H_THRESH = {H_THRESH:.2f}")
    except Exception as e:
        print(f"Error calculating H threshold: {e}")
        H_THRESH = 95

    # Calculate V_THRESH using ONLY up and down directions (V channel)
    try:
        print("\nCalculating V threshold from up/down V channels:")
        up_v_thresh = calculate_threshold("up", "V", 50)
        down_v_thresh = calculate_threshold("down", "V", 50)
        V_THRESH = min(up_v_thresh, down_v_thresh)
        print(f"V_THRESH = {V_THRESH:.2f}")
    except Exception as e:
        print(f"Error calculating V threshold: {e}")
        V_THRESH = 50

    # Plot filtered signals with thresholds
    try:
        import matplotlib.pyplot as plt
        import os
        from datetime import datetime
        # Create directory for plots if it doesn't exist
        plot_dir = "calibration_plots"
        os.makedirs(plot_dir, exist_ok=True)
        # Create a timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Filter raw channels for left/right and compute H
        ch1_left = np.array(calibration_data["left"]["ch1"])
        ch3_left = np.array(calibration_data["left"]["ch3"])
        ch1_left = notch_filter(ch1_left, FS)
        ch3_left = notch_filter(ch3_left, FS)
        ch1_left = bandpass_filter(ch1_left, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        ch3_left = bandpass_filter(ch3_left, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        H_left = ch1_left - ch3_left

        ch1_right = np.array(calibration_data["right"]["ch1"])
        ch3_right = np.array(calibration_data["right"]["ch3"])
        ch1_right = notch_filter(ch1_right, FS)
        ch3_right = notch_filter(ch3_right, FS)
        ch1_right = bandpass_filter(ch1_right, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        ch3_right = bandpass_filter(ch3_right, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        H_right = ch1_right - ch3_right

        # Filter raw channels for up/down and compute V
        ch8_up = np.array(calibration_data["up"]["ch8"])
        ch2_up = np.array(calibration_data["up"]["ch2"])
        ch8_up = notch_filter(ch8_up, FS)
        ch2_up = notch_filter(ch2_up, FS)
        ch8_up = bandpass_filter(ch8_up, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        ch2_up = bandpass_filter(ch2_up, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        V_up = ch8_up - ch2_up

        ch8_down = np.array(calibration_data["down"]["ch8"])
        ch2_down = np.array(calibration_data["down"]["ch2"])
        ch8_down = notch_filter(ch8_down, FS)
        ch2_down = notch_filter(ch2_down, FS)
        ch8_down = bandpass_filter(ch8_down, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        ch2_down = bandpass_filter(ch2_down, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        V_down = ch8_down - ch2_down

        # Plot
        plt.figure(figsize=(12, 8))
        # Horizontal signals
        plt.subplot(2, 1, 1)
        plt.plot(H_left[:500], label="Left H (filtered)")
        plt.plot(H_right[:500], label="Right H (filtered)")
        plt.axhline(y=H_THRESH, color='r', linestyle='--', label=f'H Threshold={H_THRESH:.2f}')
        plt.axhline(y=-H_THRESH, color='r', linestyle='--')
        plt.title("Filtered Horizontal Signals (First 500 Samples)")
        plt.legend()
        # Vertical signals
        plt.subplot(2, 1, 2)
        plt.plot(V_up[:500], label="Up V (filtered)")
        plt.plot(V_down[:500], label="Down V (filtered)")
        plt.axhline(y=V_THRESH, color='g', linestyle='--', label=f'V Threshold={V_THRESH:.2f}')
        plt.axhline(y=-V_THRESH, color='g', linestyle='--')
        plt.title("Filtered Vertical Signals (First 500 Samples)")
        plt.legend()
        plt.tight_layout()
        # Save the plot
        plot_path = os.path.join(plot_dir, f"calibration_signals_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved calibration plot to {plot_path}")
        # Show the plot briefly
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    except Exception as e:
        print(f"Could not display/save signal plots: {e}")

    print(f"\nFinal thresholds: H={H_THRESH:.2f}, V={V_THRESH:.2f}")
    return H_THRESH, V_THRESH


def wait_for_spacebar(window, font, message="rest your eyes and press SPACEBAR to continue when ready"):
    """
    Displays a message and waits for the user to press the SPACEBAR.
    """
    # Make sure window is filled before blitting
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
        pygame.time.delay(100)  # Reduce CPU usage
    return True

@dataclass
class Detection:
    ts: float
    direction: str
    confidence: float
    h_value: float = 0.0  # H value at detection
    v_value: float = 0.0  # V value at detection

class EOGReader(threading.Thread):
    def __init__(self, out_queue, max_queue=50, h_thresh=95, v_thresh=50):
        # First call the parent class's __init__ with no arguments
        super().__init__()
        # Then set our own attributes
        self.out_queue = out_queue
        self.max_queue = max_queue
        self.H_THRESH = h_thresh
        self.V_THRESH = v_thresh
        self.running = True
        self.channel_buffers = [collections.deque(maxlen=MAX_SAMPLES) for _ in range(TOTAL_CHANNELS)]
        self.time_buffer = collections.deque(maxlen=MAX_SAMPLES)
        self.last_detection_time = 0.0
        self.last_any_movement_time = -1e9
        print("Looking for LSL stream...")
        streams = resolve_byprop('name', LSL_STREAM_NAME, timeout=5.0)
        if not streams:
            raise RuntimeError(f"{LSL_STREAM_NAME} stream not found")
        self.inlet = StreamInlet(streams[0])
        print("Connected to stream.")
        self.start_time = time.time()
        self.latest_H = np.array([])
        self.latest_V = np.array([])
        self.latest_times = np.array([])


    def run(self):
        DETECT_PERIOD = 0.1  # seconds between running detection on the buffer
        while self.running:
            sample, _ = self.inlet.pull_sample(timeout=0.05)
            now = time.time() - self.start_time
            if sample is None:
                continue

            self.time_buffer.append(now)
            for i in range(TOTAL_CHANNELS):
                self.channel_buffers[i].append(sample[i])

            if (now - self.last_detection_time) >= DETECT_PERIOD and len(self.time_buffer) >= MAX_SAMPLES:
                self.last_detection_time = now
                times = np.array(self.time_buffer)
                ch1 = np.array(self.channel_buffers[0])
                ch2 = np.array(self.channel_buffers[1])
                ch3 = np.array(self.channel_buffers[2])
                ch8 = np.array(self.channel_buffers[7])
                
                # Apply notch filter to raw channels
                ch1 = notch_filter(ch1, FS)
                ch2 = notch_filter(ch2, FS)
                ch3 = notch_filter(ch3, FS)
                ch8 = notch_filter(ch8, FS)

                # Apply bandpass filter to raw channels
                ch1 = bandpass_filter(ch1, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
                ch2 = bandpass_filter(ch2, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
                ch3 = bandpass_filter(ch3, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
                ch8 = bandpass_filter(ch8, LOWCUT, HIGHCUT, FS, FILTER_ORDER)

                # Calculate H and V from filtered channels
                H = ch1 - ch3
                V = ch8 - ch2

                
                # Apply filters
                try:
                    self.latest_H = H
                    self.latest_V = V
                    self.latest_times = times
                except Exception as e:
                    print(f"Filter error: {e}")
                    continue

                sig_array = np.stack((H, V), axis=-1)

                if len(H) > 0 and len(V) > 0:
                    print(f"Current signals - H: {np.mean(H):.2f}±{np.std(H):.2f} (max: {np.max(np.abs(H)):.2f}), "
                      f"V: {np.mean(V):.2f}±{np.std(V):.2f} (max: {np.max(np.abs(V)):.2f})")
                    print(f"Thresholds - H: {self.H_THRESH:.2f}, V: {self.V_THRESH:.2f}")

                    # Check if signals are crossing thresholds
                    h_crossed = np.max(np.abs(H)) > self.H_THRESH
                    v_crossed = np.max(np.abs(V)) > self.V_THRESH
                    print(f"Thresholds crossed - H: {h_crossed}, V: {v_crossed}")

                # Check if we have significant signals in either direction
                h_max = np.max(np.abs(H))
                v_max = np.max(np.abs(V))

                if h_max > self.H_THRESH or v_max > self.V_THRESH:
                    sig_array = np.stack((H, V), axis=-1)
                    movements = detect_eye_movements(sig_array, times, self.H_THRESH, self.V_THRESH)

                    if movements:
                        det = movements[-1]  # take last movement

                        idx = np.searchsorted(times, det.ts)
                        idx = np.clip(idx, 0, len(times)-1)

                        # Update h/v values with filtered samples
                        det.h_value = H[idx] if idx < len(H) else 0
                        det.v_value = V[idx] if idx < len(V) else 0

                        # Push only if confidence above MIN_CONFIDENCE
                        if det.confidence >= MIN_CONFIDENCE and (now - self.last_any_movement_time) >= GLOBAL_COOLDOWN:
                            self._push(det)
                            self.last_any_movement_time = now

    def _push(self, det: Detection):
        self.out_queue.append(det)
        # cap queue size
        while len(self.out_queue) > self.max_queue:
            self.out_queue.popleft()

    def stop(self):
        self.running = False

# ==============================
# --------- Main app -----------
# ==============================

def main():
    # --- Pygame setup ---
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Static Jumps + EOG Accuracy Test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)

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
        # Second half without returning to center
        ("center", center_pos), # Return to center
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos), # Return to center
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
    ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),  # Note: lowercase 'up'
    ("center", center_pos),  
    ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),  # Note: lowercase 'down'
    ("center", center_pos), 
    ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
    ("center", center_pos),  
    ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
    ("center", center_pos),  
    ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),  # Note: lowercase 'up'
    ("center", center_pos),  
    ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),  # Note: lowercase 'down'
    ("center", center_pos),  
    ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
    ("center", center_pos),  
    ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
    ("center", center_pos), 
    ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),  # Note: lowercase 'up'
    ("center", center_pos),  
    ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),  # Note: lowercase 'down'
    ("center", center_pos),  
    ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
    ("center", center_pos),  
    ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
    ("center", center_pos),  
    ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),  # Note: lowercase 'up'
    ("center", center_pos),  
    ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),  # Note: lowercase 'down'
    
]


    def expected_from_name(name: str) -> str:
        name = name.lower()  # Convert to lowercase
        if name in ("left", "right", "up", "down"):  # All lowercase
            return {"left":"left","right":"right","up":"up","down":"down"}[name]  # All lowercase
        return "center"

    # queue for detections from the EOG thread
    det_queue = collections.deque(maxlen=50)
    eog = EOGReader(det_queue,h_thresh=95, v_thresh=50)
    eog.start()

    # --- Run calibration ---
    H_THRESH, V_THRESH = run_calibration(eog, window, font, calibration_sequence)
    eog.H_THRESH = H_THRESH
    eog.V_THRESH = V_THRESH
    print(f"Calibration complete. H_THRESH: {H_THRESH}, V_THRESH: {V_THRESH}")
    calibrated_H_THRESH = H_THRESH
    calibrated_V_THRESH = V_THRESH

    # Add pause here
    if not wait_for_spacebar(window, font, "Calibration complete! Press SPACEBAR to start the test..."):
        return  # Exit if the user closes the window

    # task state
    step_index = 0
    dot_pos = sequence[0][1]
    step_start = time.time()
    step_max_h = 0
    step_max_v = 0

    # scoring state
    trials = []
    running_correct = 0
    running_total = 0
    latest_det_display = "None"
    latest_det_conf = 0.0
    step_captured = False
    current_expected = expected_from_name(sequence[0][0])
    running = True

    try:
        while running:
            # ---------------- events ----------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            now = time.time()

            # Track max H/V values during the step using filtered signals
            if len(eog.latest_H) > 0 and len(eog.latest_V) > 0:
                step_max_h = max(step_max_h, np.max(np.abs(eog.latest_H)))
                step_max_v = max(step_max_v, np.max(np.abs(eog.latest_V)))


            # ------------- step / movement -----------
            if (now - step_start) >= STEP_DURATION:
                # finalize scoring for previous step if no detection captured
                if not step_captured:
                    is_correct = (current_expected == "center")
                    running_total += 1
                    running_correct += int(is_correct)

                    # Check if max values crossed thresholds
                    crossed_h_thresh = step_max_h > eog.H_THRESH
                    crossed_v_thresh = step_max_v > eog.V_THRESH

                    trials.append({
                        "step_index": step_index,
                        "target_name": sequence[step_index][0],
                        "expected": current_expected,
                        "detected": None,
                        "confidence": None,
                        "ts_detected": None,
                        "correct": is_correct,
                        "h_value": 0,
                        "v_value": 0
                    })

                # advance step
                step_index += 1
                if step_index >= len(sequence):
                    running = False
                    break

                dot_pos = sequence[step_index][1]
                step_start = now
                step_captured = False
                current_expected = expected_from_name(sequence[step_index][0])
                latest_det_display = "None"
                latest_det_conf = 0.0
                step_max_h = 0
                step_max_v = 0

            # ------------- detection window ----------
            if not step_captured and (now - step_start) <= RESPONSE_WINDOW:
                if det_queue:
                    det = det_queue.pop()
                    latest_det_display = det.direction
                    latest_det_conf = det.confidence
                    step_captured = True
                    running_total += 1

                    is_correct = (current_expected == det.direction) and (det.confidence >= MIN_CONFIDENCE)
                    running_correct += int(is_correct)

                    trials.append({
                        "step_index": step_index,
                        "target_name": sequence[step_index][0],
                        "expected": current_expected,
                        "detected": det.direction,
                        "confidence": float(det.confidence),
                        "ts_detected": float(det.ts),
                        "correct": is_correct,
                        "h_value": det.h_value,
                        "v_value": det.v_value
                    })
                    det_queue.clear()

            # ---------------- draw -------------------
            window.fill(BG_COLOR)
            # draw all static red targets
            for name, pos in sequence:
                pygame.draw.circle(window, RED, pos, DOT_RADIUS_STATIC)
            # active blue dot
            pygame.draw.circle(window, BLUE, dot_pos, DOT_RADIUS_ACTIVE)
            # center cross
            cx, cy = WIDTH // 2, HEIGHT // 2
            pygame.draw.line(window, BLACK, (cx - CENTER_CROSS, cy), (cx + CENTER_CROSS, cy), 3)
            pygame.draw.line(window, BLACK, (cx, cy - CENTER_CROSS), (cx, cy + CENTER_CROSS), 3)

            # overlays
            acc = (running_correct / running_total * 100.0) if running_total > 0 else 0.0
            overlay_lines = [
                f"Step {step_index+1}/{len(sequence)} | Target: {sequence[step_index][0]} | Expect: {current_expected}",
                f"Window: {max(0.0, RESPONSE_WINDOW - (now - step_start)):.2f}s | Latest det: {latest_det_display} ({latest_det_conf:.0f})",
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
        pygame.quit()

        # ------------- save results ---------------
        out_path = os.path.abspath("eog_trial_results.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "step_index", "target_name", "expected", "detected", "confidence",
                "ts_detected", "correct", "h_value", "v_value"
            ])
            writer.writeheader()

            for row in trials:
                clean_row = {
                    "step_index": row["step_index"],
                    "target_name": row["target_name"],
                    "expected": row["expected"],
                    "detected": row["detected"],
                    "confidence": row.get("confidence"),
                    "ts_detected": row.get("ts_detected"),
                    "correct": row["correct"],
                    "h_value": row.get("h_value", 0),
                    "v_value": row.get("v_value", 0)
                }
                writer.writerow(clean_row)

            # summary row
            total = len(trials)
            correct = sum(1 for r in trials if r["correct"])
            writer.writerow({})
            writer.writerow({"target_name": "SUMMARY", "expected": "—", "detected": "—",
                            "confidence": "—", "ts_detected": "—",
                            "correct": f"{correct}/{total} ({(correct/total*100.0 if total else 0.0):.1f}%)"})
            # thresholds row
            writer.writerow({"target_name": "THRESHOLDS", "expected": f"H_THRESH: {calibrated_H_THRESH}",
                            "detected": f"V_THRESH: {calibrated_V_THRESH}", "confidence": "—",
                            "ts_detected": "—", "correct": "—"})

        print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
