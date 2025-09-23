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
HIGHCUT = 20
FILTER_ORDER = 4

PEAK_DISTANCE = 125  # samples
#H_THRESH = 95
#V_THRESH = 50
MERGE_WINDOW = 500  # samples

MIN_CONFIDENCE = 150
GLOBAL_COOLDOWN = 1.2  # seconds between ANY two accepted detections

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
    b, a = sig.iirnotch(freq, bandwidth, fs)
    return sig.filtfilt(b, a, data)

def detect_eye_movements(signal, timestamps, h_thresh, v_thresh, use_highest_peak=False):
    """
    Detects eye movements in EOG signals.

    Args:
        signal: np.array shape (N, 2) with columns [H, V]
        timestamps: np.array shape (N,)
        h_thresh: Horizontal threshold
        v_thresh: Vertical threshold
        use_highest_peak: If True, uses highest peak (for calibration).
                          If False, uses first peak (for real-time detection)

    Returns:
        list of (timestamp, direction:str) where direction in {'left','right','up','down'}
    """
    horizontal = signal[:, 0]
    vertical = signal[:, 1]

    # Find peaks with appropriate parameters
    h_pos, _ = sig.find_peaks(horizontal, distance=PEAK_DISTANCE, height=h_thresh, prominence=5)
    h_neg, _ = sig.find_peaks(-horizontal, distance=PEAK_DISTANCE, height=h_thresh, prominence=5)
    v_pos, _ = sig.find_peaks(vertical, distance=PEAK_DISTANCE, height=v_thresh, prominence=5)
    v_neg, _ = sig.find_peaks(-vertical, distance=PEAK_DISTANCE, height=v_thresh, prominence=5)

    # Create peak lists with indices, types, and amplitudes
    h_peaks = [(i, 'pos', abs(horizontal[i])) for i in h_pos] + [(i, 'neg', abs(horizontal[i])) for i in h_neg]
    v_peaks = [(i, 'pos', abs(vertical[i])) for i in v_pos] + [(i, 'neg', abs(vertical[i])) for i in v_neg]

    raw_movements = []

    # Detect vertical movements
    for i in range(len(v_peaks) - 1):
        idx1, type1, amp1 = v_peaks[i]
        idx2, type2, amp2 = v_peaks[i + 1]
        if 0 < idx2 - idx1 < PEAK_DISTANCE:
            if type1 == 'pos' and type2 == 'neg':
                raw_movements.append((idx1, 'down', max(amp1, amp2)))
            elif type1 == 'neg' and type2 == 'pos':
                raw_movements.append((idx1, 'up', max(amp1, amp2)))

    # Detect horizontal movements
    for i in range(len(h_peaks) - 1):
        idx1, type1, amp1 = h_peaks[i]
        idx2, type2, amp2 = h_peaks[i + 1]
        if 0 < idx2 - idx1 < PEAK_DISTANCE:
            if type1 == 'pos' and type2 == 'neg':
                raw_movements.append((idx1, 'left', max(amp1, amp2)))
            elif type1 == 'neg' and type2 == 'pos':
                raw_movements.append((idx1, 'right', max(amp1, amp2)))

    if not raw_movements:
        return []

    # Sort and select peaks based on use case
    if use_highest_peak:
        # For calibration: sort by amplitude (highest first)
        raw_movements.sort(key=lambda x: x[2], reverse=True)
    else:
        # For real-time detection: sort by time (earliest first)
        raw_movements.sort(key=lambda x: x[0])

    # Merge nearby movements
    filtered_movements = []
    i = 0
    while i < len(raw_movements):
        group = [raw_movements[i]]
        j = i + 1
        while j < len(raw_movements) and raw_movements[j][0] - raw_movements[i][0] <= MERGE_WINDOW:
            group.append(raw_movements[j])
            j += 1

        # Select peak based on use case
        if use_highest_peak:
            # For calibration: use highest amplitude peak
            peak = max(group, key=lambda x: x[2])
        else:
            # For real-time detection: use first peak
            peak = group[0]

        filtered_movements.append((timestamps[peak[0]], peak[1], peak[2]))
        i = j

    return filtered_movements

def run_calibration(eog_reader, window, font, calibration_sequence):
    # Data structure to store raw signals for each direction
    calibration_data = {
        "left": {"H": [], "V": []},
        "right": {"H": [], "V": []},
        "up": {"H": [], "V": []},      # Note: lowercase 'up'
        "down": {"H": [], "V": []},    # Note: lowercase 'down'
        "center": {"H": [], "V": []}
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
        return 95, 50, 0

    # Step 1: Record baseline at center FIRST (before any movements)
    window.fill(BG_COLOR)
    pygame.draw.circle(window, BLUE, center_pos, DOT_RADIUS_ACTIVE)
    instruction = "Look at the CENTER (baseline). Hold your gaze steady."
    surf = font.render(instruction, True, BLACK)
    window.blit(surf, (10, 50))
    pygame.display.flip()

    # Record baseline for 5 seconds (no movement)
    start_time = time.time()
    baseline_H = []
    baseline_V = []
    print("Recording baseline for 5 seconds...")
    while time.time() - start_time < 5.0:
        sample, _ = eog_reader.inlet.pull_sample(timeout=0.01)
        if sample is not None:
            ch1 = sample[0]
            ch2 = sample[1]
            ch3 = sample[2]
            ch8 = sample[7]
            H = ch1 - ch3
            V = ch8 - ch2
            baseline_H.append(H)
            baseline_V.append(V)
        pygame.event.pump()
        clock.tick(60)

    # Calculate center baseline from this clean baseline recording
    if baseline_H and baseline_V:
        center_baseline_H = np.mean(baseline_H)
        center_baseline_V = np.mean(baseline_V)
    else:
        print("Warning: No baseline samples received. Using default values.")
        center_baseline_H = 0
        center_baseline_V = 0

    print(f"Center baseline H: {center_baseline_H:.1f}, V: {center_baseline_V:.1f}")
    center_baseline = (center_baseline_H + center_baseline_V) / 2
    print(f"center baseline: {center_baseline}")

    # Step 2: Record raw signals for each directional target
    for i, (target_name, pos) in enumerate(calibration_sequence):
        target_key = target_name.lower()  # Convert to lowercase for dictionary access

        # Make sure the target_key exists in our calibration_data
        if target_key not in calibration_data:
            print(f"Warning: Unknown target direction '{target_name}'. Skipping.")
            continue

        # Add a rest step every 5 targets
        if i > 0 and i % 8 == 0:
            window.fill(BG_COLOR)
            rest_surf = font.render("Rest your eyes and blink if needed. Press SPACEBAR to continue.", True, BLACK)
            window.blit(rest_surf, (WIDTH // 2 - rest_surf.get_width() // 2, HEIGHT // 2))
            pygame.display.flip()
            if not wait_for_spacebar(window, font):
                return 95, 50, max(center_baseline_H, center_baseline_V)

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
        target_H = []
        target_V = []
        print(f"Recording signals for {target_name}...")

        while time.time() < end_time:
            sample, _ = eog_reader.inlet.pull_sample(timeout=0.01)
            if sample is not None:
                ch1 = sample[0]
                ch2 = sample[1]
                ch3 = sample[2]
                ch8 = sample[7]
                H = ch1 - ch3
                V = ch8 - ch2
                target_H.append(H)
                target_V.append(V)
            pygame.event.pump()
            clock.tick(60)

        # Store the raw signals for this target
        if target_H and target_V:
            calibration_data[target_key]["H"].extend(target_H)
            calibration_data[target_key]["V"].extend(target_V)
            print(f"Recorded {len(target_H)} samples for {target_name}")
        else:
            print(f"Warning: No samples recorded for {target_name}")

    # Step 3: Calculate thresholds based on the top 10% of signals for each direction
    def calculate_threshold(direction, channel, default):
        if not calibration_data[direction][channel]:
            print(f"Warning: No {channel} signals recorded for {direction}. Using default threshold.")
            return default

        # Get all signals for this direction and channel
        signals = np.array(calibration_data[direction][channel])

        # Apply bandpass filtering to remove noise
        try:
            filtered = bandpass_filter(signals, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        except Exception as e:
            print(f"Error filtering signals for {direction} {channel}: {e}")
            return default

        # Find absolute values to detect peaks regardless of direction
        abs_signals = np.abs(filtered)

        # Sort signals by amplitude (highest first)
        sorted_signals = np.sort(abs_signals)[::-1]

        # Take the top 10% of signals
        top_count = max(1, int(len(sorted_signals) * 0.1))
        top_signals = sorted_signals[:top_count]

        # Calculate threshold as 0.75 times the mean of top signals
        threshold = np.mean(top_signals) * 0.75

        return max(default, threshold)

    # Calculate horizontal threshold (based on H channel for left/right)
    try:
        H_THRESH = max(
            calculate_threshold("left", "H", 95),
            calculate_threshold("right", "H", 95)
        )
    except Exception as e:
        print(f"Error calculating horizontal threshold: {e}")
        H_THRESH = 95

    # Calculate vertical threshold (based on V channel for up/down)
    try:
        V_THRESH = max(
            calculate_threshold("up", "V", 50),
            calculate_threshold("down", "V", 50)
        )
    except Exception as e:
        print(f"Error calculating vertical threshold: {e}")
        V_THRESH = 50

    # Ensure minimum threshold values
    H_THRESH = max(H_THRESH, 40)
    V_THRESH = max(V_THRESH, 30)

    print(f"Calculated thresholds based on top 10% of signals: H_THRESH={H_THRESH:.1f}, V_THRESH={V_THRESH:.1f}")
    return H_THRESH, V_THRESH, center_baseline

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

class EOGReader(threading.Thread):
    """
    Continuously reads LSL, maintains a rolling buffer, filters, runs detection periodically,
    and publishes accepted detections (with cooldown + confidence threshold) into `out_queue`.
    """
    def __init__(self, out_queue: collections.deque, max_queue=50, h_thresh=95, v_thresh=50):
        super().__init__(daemon=True)
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

                H = notch_filter(ch1 - ch3, FS)
                V = notch_filter(ch8 - ch2, FS)


                try:
                    Hf = bandpass_filter(H, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
                    Vf = bandpass_filter(V, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
                except Exception:
                    # In case of filter warmup issues
                    continue

                sig_array = np.stack((Hf, Vf), axis=-1)
                movements = detect_eye_movements(sig_array, times, self.H_THRESH, self.V_THRESH, use_highest_peak=False)

                if movements:
                    latest_time, direction = movements[-1]
                    # confidence ~ |H| or |V| at that time
                    idx = np.searchsorted(times, latest_time)
                    idx = np.clip(idx, 0, len(times)-1)
                    confidence = (abs(sig_array[idx, 0]) if direction in ('left','right')
                                  else abs(sig_array[idx, 1]))

                    if confidence >= MIN_CONFIDENCE and (now - self.last_any_movement_time) >= GLOBAL_COOLDOWN:
                        det = Detection(ts=latest_time, direction=direction, confidence=float(confidence))
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
        ("center", center_pos),  # Baseline - lowercase
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),  # Return to baseline
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),  # Return to baseline
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),  # Note: lowercase 'up'
        ("center", center_pos),  # Return to baseline
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),  # Note: lowercase 'down'
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
    ("center", center_pos),  # Baseline - lowercase
    ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
    ("center", center_pos),  # Return to baseline
    ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
    ("center", center_pos),  # Return to baseline
    ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),  # Note: lowercase 'up'
    ("center", center_pos),  # Return to baseline
    ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),  # Note: lowercase 'down'
    ("center", center_pos),  # Baseline - lowercase
    ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
    ("center", center_pos),  # Return to baseline
    ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
    ("center", center_pos),  # Return to baseline
    ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),  # Note: lowercase 'up'
    ("center", center_pos),  # Return to baseline
    ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),  # Note: lowercase 'down'
    ("center", center_pos),  # Baseline - lowercase
    ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
    ("center", center_pos),  # Return to baseline
    ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
    ("center", center_pos),  # Return to baseline
    ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),  # Note: lowercase 'up'
    ("center", center_pos),  # Return to baseline
    ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),  # Note: lowercase 'down'
    ("center", center_pos),  # Baseline - lowercase
    ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
    ("center", center_pos),  # Return to baseline
    ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
    ("center", center_pos),  # Return to baseline
    ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),  # Note: lowercase 'up'
    ("center", center_pos),  # Return to baseline
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
    H_THRESH, V_THRESH, center_baseline = run_calibration(eog, window, font, calibration_sequence)
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

    # scoring state
    trials = []  # list of dicts
    running_correct = 0
    running_total = 0
    latest_det_display = "None"
    latest_det_conf = 0.0

    # For each step, we accept the *first* detection within [step_start, step_start+RESPONSE_WINDOW]
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

            # ------------- step / movement -----------
            if (now - step_start) >= STEP_DURATION:
                # finalize scoring for previous step if no detection captured
                if not step_captured:
                    # For center, "no detection" counts as correct; else it's incorrect (miss)
                    is_correct = (current_expected == "center")
                    running_total += 1
                    running_correct += int(is_correct)
                    trials.append({
                        "step_index": step_index,
                        "target_name": sequence[step_index][0],
                        "expected": current_expected,
                        "detected": None,
                        "confidence": None,
                        "ts_detected": None,
                        "correct": is_correct
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

            # ------------- detection window ----------
            if not step_captured and (now - step_start) <= RESPONSE_WINDOW:
                # consume from queue (take newest; if multiple, prefer earliest in window)
                # We'll scan queue and pick the first detection whose ts >= (relative to reader start)
                # Since we don't have absolute sync, use wall time match: we just take the *latest* available
                # and mark as captured (practical for real-time UX). Optionally, you can preserve det.ts.
                # In the main loop, inside the detection window block:
                if det_queue:
                    det = det_queue.pop()
                    latest_det_display = det.direction
                    latest_det_conf = det.confidence
                    step_captured = True
                    running_total += 1
                    # Always log the detection, regardless of confidence
                    is_correct = (current_expected == det.direction) and (det.confidence >= MIN_CONFIDENCE)
                    running_correct += int(is_correct)
                    trials.append({
                        "step_index": step_index,
                        "target_name": sequence[step_index][0],
                        "expected": current_expected,
                        "detected": det.direction,
                        "confidence": float(det.confidence),
                        "ts_detected": float(det.ts),
                        "correct": is_correct  # Only True if confidence >= MIN_CONFIDENCE and direction matches
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
    with open(out_path, "w", newline="") as f:  # <-- Fixed indentation
        writer = csv.DictWriter(f, fieldnames=[
            "step_index", "target_name", "expected", "detected", "confidence", "ts_detected", "correct"
        ])
        writer.writeheader()
        for row in trials:
            writer.writerow(row)
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
        # baseline row
        writer.writerow({"target_name": "BASELINE", "expected": f"Center Baseline: {center_baseline:.1f}",
                        "detected": "-", "confidence": "—", "ts_detected": "—", "correct": "—"})
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
