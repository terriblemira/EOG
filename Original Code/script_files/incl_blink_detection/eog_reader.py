import threading
import collections
import numpy as np
import time
from pylsl import StreamInlet, resolve_byprop
from signal_processing_wavelet import process_eog_signals, process_eog_signals_with_blinks
import config
from dataclasses import dataclass

@dataclass
class Detection:
    ts: float
    direction: str
    is_horizontal: bool
    is_blink: bool = False  # Add blink flag
    blink_duration: float = 0.0
    h_value: float = 0.0
    v_value: float = 0.0
    h_velocity: float = 0.0
    v_velocity: float = 0.0

class EOGReader(threading.Thread):
    def __init__(self, out_queue, max_queue=50, calibration_params=None):
        super().__init__()
        self.out_queue = out_queue
        self.max_queue = max_queue
        self.calibration_params = calibration_params or {
            "baselines": {"H": 0, "V": 0},
            "thresholds": {"left": 0.3, "right": 0.3, "up": 0.3, "down": 0.3},
            "blink_threshold": config.BLINK_THRESHOLD,
            "channel_norm_factors": {"ch1": 1, "ch2": 1, "ch3": 1, "ch8": 1},
            "alpha": 0.0
        }
        self.last_blink_time = -1e9
        self.running = True

        # Buffers for plotting (5 seconds) and detection (1 second)
        self.channel_buffers = [collections.deque(maxlen=config.PLOT_MAX_SAMPLES) for _ in range(config.TOTAL_CHANNELS)]
        self.time_buffer = collections.deque(maxlen=config.PLOT_MAX_SAMPLES)
        self.detect_channel_buffers = [collections.deque(maxlen=config.DETECT_MAX_SAMPLES) for _ in range(config.TOTAL_CHANNELS)]
        self.detect_time_buffer = collections.deque(maxlen=config.DETECT_MAX_SAMPLES)

        self.last_detection_time = 0.0
        self.last_any_movement_time = -1e9

        print("Looking for LSL stream...")
        streams = resolve_byprop('name', config.LSL_STREAM_NAME, timeout=5.0)
        if not streams:
            raise RuntimeError(f"{config.LSL_STREAM_NAME} stream not found")
        self.inlet = StreamInlet(streams[0])
        print("Connected to stream.")
        self.start_time = time.time()

        # For plotting and debugging - store full buffer data
        self.full_H = np.array([])
        self.full_V = np.array([])
        self.full_times = np.array([])

        # For detection - store detection window data
        self.latest_H = np.array([])
        self.latest_V = np.array([])
        self.latest_times = np.array([])

        self.debug_counter = 0
        self.last_plot_update = time.time()

    def _push(self, det: Detection):
        """Push detection to queue with cooldown check"""
        current_time = time.time() - self.start_time
        if (current_time - self.last_any_movement_time) >= config.GLOBAL_COOLDOWN:
            self.out_queue.append(det)
            while len(self.out_queue) > self.max_queue:
                self.out_queue.popleft()
            self.last_any_movement_time = current_time
            return True
        return False

    def process_detection_window(self):
        """Process the detection window and check for saccades and blinks"""
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

            # process signals with alpha compensation
            H_corrected, V_corrected, V_compensated, blink_events = process_eog_signals_with_blinks(ch1, ch2, ch3, ch8, self.calibration_params)

            # Update latest signals for detection
            self.latest_H = H_corrected
            self.latest_V = V_compensated
            self.latest_times = times

            # Define velocity thresholds
            H_VELOCITY_THRESHOLD = 0.05  
            V_VELOCITY_THRESHOLD = 0.05  

            # Calculate velocity (derivative)
            H_velocity = np.gradient(H_corrected)
            V_velocity = np.gradient(V_compensated)

            # Handle blink events
            current_time = time.time() - self.start_time
            for blink in blink_events:
                # check cooldown
                if (current_time - self.last_blink_time) < config.BLINK_COOLDOWN:
                    continue

                if abs(V_compensated[blink['peak_index']]) < self.calibration_params['blink_threshold']:
                    continue  # Ignore small blinks

            # Create a blink detection
                det = Detection(
                    ts=times[blink['peak_index']],
                    direction='blink',
                    is_horizontal=False,
                    is_blink=True,
                    blink_duration=blink['duration'],
                    h_value=H_corrected[blink['peak_index']],
                    v_value=V_compensated[blink['peak_index']],
                    h_velocity=H_velocity[blink['peak_index']],
                    v_velocity=V_velocity[blink['peak_index']]
                )

                if self._push(det):
                    print(f"Pushed blink detection to queue at {times[blink['peak_index']]:.2f}s")
                    self.last_blink_time = current_time

            # Check each sample for threshold crossing and velocity
            now = time.time() - self.start_time
            for idx in range(1, len(H_corrected)):
                h_val = H_corrected[idx]
                v_val = V_compensated[idx]
                h_vel = abs(H_velocity[idx])
                v_vel = abs(V_velocity[idx])

                # Check for horizontal movements with velocity
                if h_val > self.calibration_params["thresholds"]["right"] and h_vel > H_VELOCITY_THRESHOLD:
                    det = Detection(
                        ts=times[idx],
                        direction='right',
                        is_horizontal=True,
                        h_value=h_val,
                        v_value=v_val,
                        h_velocity=h_vel,
                        v_velocity=v_vel
                    )
                    if self._push(det):
                        print(f"Pushed right detection to queue at {times[idx]:.2f}s")

                elif h_val < -self.calibration_params["thresholds"]["left"] and h_vel > H_VELOCITY_THRESHOLD:
                    det = Detection(
                        ts=times[idx],
                        direction='left',
                        is_horizontal=True,
                        h_value=h_val,
                        v_value=v_val,
                        h_velocity=h_vel,
                        v_velocity=v_vel
                    )
                    if self._push(det):
                        print(f"Pushed left detection to queue at {times[idx]:.2f}s")

                # Check for vertical movements with velocity and minimum amplitude
                if self.calibration_params['blink_threshold'] > v_val > self.calibration_params["thresholds"]["up"] and v_vel > V_VELOCITY_THRESHOLD:
                    det = Detection(
                        ts=times[idx],
                        direction='up',
                        is_horizontal=False,
                        h_value=h_val,
                        v_value=v_val,
                        h_velocity=h_vel,
                        v_velocity=v_vel
                    )
                    if self._push(det):
                        print(f"Pushed up detection to queue at {times[idx]:.2f}s")
                
                elif v_val > self.calibration_params["thresholds"]["down"] and v_vel > V_VELOCITY_THRESHOLD:
                    det = Detection(
                        ts=times[idx],
                        direction='down',
                        is_horizontal=False,
                        h_value=h_val,
                        v_value=v_val,
                        h_velocity=h_vel,
                        v_velocity=v_vel
                    )
                    if self._push(det):
                        print(f"Pushed down detection to queue at {times[idx]:.2f}s")

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

            # Add to plot buffers (5 seconds)
            self.time_buffer.append(now)
            for i in range(config.TOTAL_CHANNELS):
                self.channel_buffers[i].append(sample[i])

            # Add to detection buffers (1 second)
            self.detect_time_buffer.append(now)
            for i in range(config.TOTAL_CHANNELS):
                self.detect_channel_buffers[i].append(sample[i])

            # Process full buffers for plotting periodically (every 0.2 seconds)
            current_time = time.time()
            if (current_time - self.last_plot_update) >= 0.005 and len(self.time_buffer) > 0:
                self.last_plot_update = current_time

                try:
                    # Process all channels for plotting
                    ch1 = np.array(self.channel_buffers[0])
                    ch2 = np.array(self.channel_buffers[1])
                    ch3 = np.array(self.channel_buffers[2])
                    ch8 = np.array(self.channel_buffers[7])

                    # Ensure all arrays have the same length
                    min_length = min(len(self.time_buffer), len(ch1), len(ch2), len(ch3), len(ch8))
                    if min_length > 0:
                        times = np.array(self.time_buffer)[-min_length:]
                        ch1 = ch1[-min_length:]
                        ch2 = ch2[-min_length:]
                        ch3 = ch3[-min_length:]
                        ch8 = ch8[-min_length:]

                        # Process signals with alpha compensation
                        H_corrected, V_corrected, V_compensated = process_eog_signals(
                            ch1, ch2, ch3, ch8, self.calibration_params
                        )

                        # Check if we got valid results
                        if all(isinstance(arr, np.ndarray) and len(arr) > 0 for arr in [H_corrected, V_corrected, V_compensated]):
                            # Update full buffers for plotting
                            self.full_H = H_corrected
                            self.full_V = V_compensated
                            self.full_times = times
                except Exception as e:
                    print(f"Error in plotting processing: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    
            # Run detection every 0.1 seconds
            if (current_time - last_detection_check) >= DETECT_PERIOD and len(self.detect_time_buffer) >= config.DETECT_MAX_SAMPLES:
                last_detection_check = current_time
                self.process_detection_window()

    def stop(self):
        """Stop the thread"""
        self.running = False
        self.join()
        print("EOGReader stopped.")
