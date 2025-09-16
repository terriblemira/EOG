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
WIDTH, HEIGHT = 2000, 1000
BG_COLOR = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

STEP_DURATION = 5.0        # seconds between target jumps
RESPONSE_WINDOW = 2.5      # seconds after each jump during which we accept the first valid detection
DOT_RADIUS_ACTIVE = 20
DOT_RADIUS_STATIC = 10
CENTER_CROSS = 10

# --- LSL / signal processing ---
FS = 250
BUFFER_DURATION = 5
MAX_SAMPLES = FS * BUFFER_DURATION
TOTAL_CHANNELS = 8
CHANNEL_INDICES = [0, 1, 2, 7]  # (we'll form H = ch1 - ch3, V = ch8 - ch2)

LOWCUT = 0.4
HIGHCUT = 40
FILTER_ORDER = 4

PEAK_DISTANCE = 125  # samples
H_THRESH = 95
V_THRESH = 50
MERGE_WINDOW = 500  # samples

MIN_CONFIDENCE = 200
GLOBAL_COOLDOWN = 1.2  # seconds between ANY two accepted detections

LSL_STREAM_NAME = 'Explore_8441_ExG'

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

def detect_eye_movements(signal, timestamps):
    """
    signal: np.array shape (N, 2) with columns [H, V]
    timestamps: np.array shape (N,)
    Returns: list of (timestamp, direction:str) where direction in {'left','right','up','down'}
    """
    horizontal = signal[:, 0]
    vertical = signal[:, 1]

    h_pos, _ = sig.find_peaks(horizontal, distance=PEAK_DISTANCE, height=H_THRESH)
    h_neg, _ = sig.find_peaks(-horizontal, distance=PEAK_DISTANCE, height=H_THRESH)
    v_pos, _ = sig.find_peaks(vertical, distance=PEAK_DISTANCE, height=V_THRESH)
    v_neg, _ = sig.find_peaks(-vertical, distance=PEAK_DISTANCE, height=V_THRESH)

    h_peaks = sorted([(i, 'pos', abs(horizontal[i])) for i in h_pos] +
                     [(i, 'neg', abs(horizontal[i])) for i in h_neg])
    v_peaks = sorted([(i, 'pos', abs(vertical[i])) for i in v_pos] +
                     [(i, 'neg', abs(vertical[i])) for i in v_neg])

    raw_movements = []
    for i in range(len(v_peaks) - 1):
        idx1, type1, amp1 = v_peaks[i]
        idx2, type2, amp2 = v_peaks[i + 1]
        if 0 < idx2 - idx1 < PEAK_DISTANCE:
            if type1 == 'pos' and type2 == 'neg':
                raw_movements.append((idx1, 'down', max(amp1, amp2)))
            elif type1 == 'neg' and type2 == 'pos':
                raw_movements.append((idx1, 'up', max(amp1, amp2)))

    for i in range(len(h_peaks) - 1):
        idx1, type1, amp1 = h_peaks[i]
        idx2, type2, amp2 = h_peaks[i + 1]
        if 0 < idx2 - idx1 < PEAK_DISTANCE:
            if type1 == 'pos' and type2 == 'neg':
                raw_movements.append((idx1, 'left', max(amp1, amp2)))
            elif type1 == 'neg' and type2 == 'pos':
                raw_movements.append((idx1, 'right', max(amp1, amp2)))

    raw_movements.sort(key=lambda x: x[0])
    filtered_movements = []
    i = 0
    while i < len(raw_movements):
        group = [raw_movements[i]]
        j = i + 1
        while j < len(raw_movements) and raw_movements[j][0] - raw_movements[i][0] <= MERGE_WINDOW:
            group.append(raw_movements[j])
            j += 1
        peak = max(group, key=lambda x: x[2])
        filtered_movements.append((timestamps[peak[0]], peak[1]))
        i = j

    return filtered_movements

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
    def __init__(self, out_queue: collections.deque, max_queue=50):
        super().__init__(daemon=True)
        self.out_queue = out_queue
        self.max_queue = max_queue
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
        DETECT_PERIOD = 0.25  # seconds between running detection on the buffer
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

                H = ch1 - ch3
                V = ch8 - ch2

                try:
                    Hf = bandpass_filter(H, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
                    Vf = bandpass_filter(V, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
                except Exception:
                    # In case of filter warmup issues
                    continue

                sig_array = np.stack((Hf, Vf), axis=-1)
                movements = detect_eye_movements(sig_array, times)

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
        ("Center", center_pos),
        ("Left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("Center", center_pos),
        ("Right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("Center", center_pos),
        ("Top", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("Center", center_pos),
        ("Bottom", [WIDTH // 2, int(0.95 * HEIGHT)]),
        ("Center", center_pos),
        ("Left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("Right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("Left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("Top", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("Bottom", [WIDTH // 2, int(0.95 * HEIGHT)]),
        ("Top", [WIDTH // 2, int(0.05 * HEIGHT)]),
    ]

    def expected_from_name(name: str) -> str:
        name = name.lower()
        if name in ("left", "right", "top", "bottom"):
            return {"left":"left","right":"right","top":"up","bottom":"down"}[name]
        return "center"

    # queue for detections from the EOG thread
    det_queue = collections.deque(maxlen=50)
    eog = EOGReader(det_queue)
    eog.start()

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
                if det_queue:
                    det = det_queue.pop()
                    # push back others (we only want the first valid one for this window)
                    # Check confidence already enforced by thread; just take it
                    latest_det_display = det.direction
                    latest_det_conf = det.confidence
                    step_captured = True
                    running_total += 1
                    is_correct = (current_expected == det.direction)
                    running_correct += int(is_correct)
                    trials.append({
                        "step_index": step_index,
                        "target_name": sequence[step_index][0],
                        "expected": current_expected,
                        "detected": det.direction,
                        "confidence": float(det.confidence),
                        "ts_detected": float(det.ts),
                        "correct": is_correct
                    })
                    # clear queue older items (optional)
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
    print(f"Saved results to {out_path}")

if __name__ == "__main__":
    main()
