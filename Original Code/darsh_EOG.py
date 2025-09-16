import numpy as np
import collections
import time
from pylsl import StreamInlet, resolve_byprop
from scipy import signal as sig

# -------- Settings --------
FS = 250
BUFFER_DURATION = 5
MAX_SAMPLES = FS * BUFFER_DURATION
CHANNEL_INDICES = [0, 1, 2, 7]  # ch1, ch2, ch3, ch8
TOTAL_CHANNELS = 8

# Filtering
LOWCUT = 0.4
HIGHCUT = 40
FILTER_ORDER = 4

# Eye movement detection
PEAK_DISTANCE = 125
H_THRESH = 95
V_THRESH = 50
MERGE_WINDOW = 500  # in samples

#refining variables
MIN_CONFIDENCE = 200       # Adjust based on testing
GLOBAL_COOLDOWN = 1.2      # Seconds between ANY two detections
last_any_movement_time = -np.inf


# -------- LSL Setup --------
print("Looking for LSL stream...")
streams = resolve_byprop('name', 'Explore_8441_ExG', timeout=5.0)
if not streams:
    raise RuntimeError("Explore_8441_ExG stream not found")
inlet = StreamInlet(streams[0])
print("Connected to stream.")

# -------- Filter Functions --------
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return sig.butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    return sig.filtfilt(b, a, data)

# -------- Eye Movement Detection --------
def detect_eye_movements(signal, timestamps):
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

# # -------- Live Buffer --------
channel_buffers = [collections.deque(maxlen=MAX_SAMPLES) for _ in range(TOTAL_CHANNELS)]
time_buffer = collections.deque(maxlen=MAX_SAMPLES)

start_time = time.time()
last_detection_time = 0
detection_interval = 1.5  # seconds between detections

print("Streaming and processing...")

while True:
    sample, timestamp = inlet.pull_sample()
    now = time.time() - start_time

    time_buffer.append(now)
    for i in range(TOTAL_CHANNELS):
        channel_buffers[i].append(sample[i])

    # Only run detection periodically
    if now - last_detection_time > detection_interval and len(time_buffer) >= MAX_SAMPLES:
        last_detection_time = now

        # Convert to NumPy for processing
        times = np.array(time_buffer)
        ch1 = np.array(channel_buffers[0])
        ch2 = np.array(channel_buffers[1])
        ch3 = np.array(channel_buffers[2])
        ch8 = np.array(channel_buffers[7])

        # Compute horizontal and vertical signals
        H = ch1 - ch3
        V = ch8 - ch2

        # Filter
        H_filt = bandpass_filter(H, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
        V_filt = bandpass_filter(V, LOWCUT, HIGHCUT, FS, FILTER_ORDER)

        # Blink removal and detection can be added here

        # Detect gaze direction
        sig_array = np.stack((H_filt, V_filt), axis=-1)
        movements = detect_eye_movements(sig_array, times)

        if movements:
            latest_time, direction = movements[-1]

            # Find index for confidence estimation
            latest_idx = np.where(times == latest_time)[0][0]

            # Use signal amplitude as confidence
            confidence = (
                np.abs(sig_array[latest_idx, 0]) if direction in ['left', 'right']
                else np.abs(sig_array[latest_idx, 1])
            )

            # Apply confidence and cooldown filters
            if confidence >= MIN_CONFIDENCE:
                if now - last_any_movement_time > GLOBAL_COOLDOWN:
                    print(f"[{latest_time:.2f}s] Direction: {direction} | Confidence: {confidence:.1f}")
                    last_any_movement_time = now

# function for reachy robot to get first gaze direction.

def get_first_gaze_direction():
    global last_any_movement_time  # still needed for cooldown
    last_any_movement_time = -np.inf

    # LSL Setup
    print("Looking for LSL stream...")
    streams = resolve_byprop('name', 'Explore_8441_ExG', timeout=5.0)
    if not streams:
        raise RuntimeError("Explore_8441_ExG stream not found")
    inlet = StreamInlet(streams[0])
    print("Connected to stream.")

    # Buffers
    channel_buffers = [collections.deque(maxlen=MAX_SAMPLES) for _ in range(TOTAL_CHANNELS)]
    time_buffer = collections.deque(maxlen=MAX_SAMPLES)

    start_time = time.time()
    last_detection_time = 0

    print("Waiting for first valid gaze direction...")

    while True:
        sample, timestamp = inlet.pull_sample()
        now = time.time() - start_time

        time_buffer.append(now)
        for i in range(TOTAL_CHANNELS):
            channel_buffers[i].append(sample[i])

        if now - last_detection_time > 1.5 and len(time_buffer) >= MAX_SAMPLES:
            last_detection_time = now

            times = np.array(time_buffer)
            ch1 = np.array(channel_buffers[0])
            ch2 = np.array(channel_buffers[1])
            ch3 = np.array(channel_buffers[2])
            ch8 = np.array(channel_buffers[7])

            H = ch1 - ch3
            V = ch8 - ch2

            H_filt = bandpass_filter(H, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
            V_filt = bandpass_filter(V, LOWCUT, HIGHCUT, FS, FILTER_ORDER)

            sig_array = np.stack((H_filt, V_filt), axis=-1)
            movements = detect_eye_movements(sig_array, times)

            if movements:
                latest_time, direction = movements[-1]
                return direction