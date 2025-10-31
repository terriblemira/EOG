import threading
import collections
import numpy as np
import time
from pylsl import StreamInlet, resolve_byprop
from signal_processing_wavelet import process_eog_signals
import config
from dataclasses import dataclass
import asyncio # M:import bc of bool valid...Movement for webapp signal
import websockets # M:import bc of bool valid...Movement for webapp signal

@dataclass
class Detection:
    ts: float
    direction: str
    is_horizontal: bool
    h_value: float = 0.0 #M: ch1 - ch3 --> + = look left/right??(--> Jose has to check bc see push det), rounded to 1 decimal
    v_value: float = 0.0 #M: ch8 - ch2 --> + = look up, "
    h_velocity: float = 0.0
    v_velocity: float = 0.0

signal = None #module-level variable to store current movement signal for webapp

class EOGReader(threading.Thread):

    #M: function to "create connection once, put it in self.ws and keep it open"; not running yet, will be called in run()
    # async def connect_to_webapp(self):
    #     try: 
    #         uri = "ws://localhost:8000/wsRight"
    #         self.ws = await websockets.connect(uri)
    #         print("EOG Reader connected to webapp!")
    #     except Exception as e1: #M: in case of error which otherwise would bring down the program (prints error & continues)
    #         print(f"Error connecting to webapp: {str(e1)}")

    def __init__(self, out_queue, max_queue=50, calibration_params=None): #M: Joses init function
        super().__init__()
  #      self.ws = None #M: ws = variable for websocket connection that is right now empty (None) and will get filled with await... in connect_to_webapp()
    #    self.eventLoop = None #M: event loop = "Main Thread" (1st started function when running, in this case "async def main()" in main.py); "Motor" for all asyncio functions; variable eventLoop gets filled with actual event loop in main.py (with eog.loop = asyncio.get_event_loop())
        self.raw_log = []  # To store raw data if recording is enabled
        self.record_raw = False  # Flag to control raw data recording
        self.out_queue = out_queue
        self.max_queue = max_queue
        self.calibration_params = calibration_params or {
            "baselines": {"H": 0, "V": 0},
            "thresholds": {"left": 0.3, "right": 0.3, "up": 0.3, "down": 0.3},
            "channel_norm_factors": {"ch1": 1, "ch2": 1, "ch3": 1, "ch8": 1},
            "alpha": 0.0
        }
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
            raise RuntimeError(f"{config.LSL_STREAM_NAME} stream not found") # signal sth went wrong (not continuing like "except Exception as e")
        self.inlet = StreamInlet(streams[0])
        print("Connected to stream.")
        self.start_time = time.time()

        # For plotting and debugging - store full buffer data
        self.full_H = np.array([]) #M: all the h_value(s)
        self.full_V = np.array([]) 
        self.full_times = np.array([]) #M: timestamps for each value of h_value+v_value

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
                self.out_queue.popleft()  #M: limit in case you have too many detections (f.ex. blinks) (more than (50=default) times) in 0.1s
            self.last_any_movement_time = current_time
            return True
        return False

    def process_detection_window(self):
        """Process the detection window and check for saccades"""
        global signal #M: access module-level variable, so not just accessible in this function
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
            H_corrected, V_corrected, V_compensated = process_eog_signals(ch1, ch2, ch3, ch8, self.calibration_params)

            # Update latest signals for detection
            self.latest_H = H_corrected
            self.latest_V = V_compensated
            self.latest_times = times

            # Define velocity thresholds
            H_VELOCITY_THRESHOLD = 0.05  # Lower threshold for horizontal movements
            V_VELOCITY_THRESHOLD = 0.05  # Lower threshold for vertical movements

            # Calculate velocity (derivative)
            H_velocity = np.gradient(H_corrected)
            V_velocity = np.gradient(V_compensated)

            # Minimum movement threshold
            MIN_V_MOVEMENT = 0.5

            # Check each sample for threshold crossing and velocity
            now = time.time() - self.start_time
            for idx in range(1, len(H_corrected)):
                h_val = H_corrected[idx]
                v_val = V_compensated[idx]
                h_vel = abs(H_velocity[idx])
                v_vel = abs(V_velocity[idx])

                # Check for horizontal movements with velocity
                if h_val < -self.calibration_params["thresholds"]["right"] and h_vel > H_VELOCITY_THRESHOLD: # 2nd pretty much always true bc Jose made vel.treshh.value low
                    det = Detection(
                        ts=times[idx],
                        direction='right',
                        is_horizontal=True,
                        h_value=h_val,
                        v_value=v_val,
                        h_velocity=h_vel,
                        v_velocity=v_vel
                    )
                    if self._push(det): # ...and if cooldown function allows (if outcome is "yes"): do following:
                        print(f"Pushed right detection to queue at {times[idx]:.2f}s") #M: "times[exact sample]:.2f"(rounded to 2 decimals)
                        ##M: MIRA/DARSH: ADDED: give signal to webapp to move (right):
                        signal = "right"
                        # self.send_valid_movement() #replace (signal) in "self.send_valid_movement(signal)" with ("right")
                        continue

                elif h_val > self.calibration_params["thresholds"]["left"] and h_vel > H_VELOCITY_THRESHOLD:
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
                        ##M: MIRA/DARSH: HERE TO ADD: give signal to webapp to move (left)!

                # Check for vertical movements with velocity and minimum amplitude
                if v_val < -self.calibration_params["thresholds"]["up"] and v_vel > V_VELOCITY_THRESHOLD:
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
                        ##M: MIRA/DARSH: HERE TO ADD: give signal to webapp to move (up)!
                
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
                        ##M: MIRA/DARSH: HERE TO ADD: give signal to webapp to move (down)!

        except Exception as e:
            print(f"Error in detection processing: {str(e)}")
            import traceback
            traceback.print_exc()

    #M: LOCATE AFTER push_functions for overview. function to send movement signal to webapp (not sending yet: just called when movement detected and valid (see below))
    # async def send_valid_movement(self, signal):
    #     if self.ws and self.eventLoop:
    #         asyncio.run_coroutine_threadsafe(self.ws.send(signal), self.eventLoop) #M: equivalent to create_task but compatible with threads

    def run(self):
        """Main thread loop for reading and processing EOG data"""
        DETECT_PERIOD = 0.1  # seconds between running detection on the buffer
        last_detection_check = time.time()

        while self.running:
            sample, timestamp = self.inlet.pull_sample(timeout=0.05)
            if sample is None:
                continue

            now = time.time() - self.start_time

#Add raw data logging if enabled
            if self.record_raw:
                self.raw_log.append([now] + list(sample))

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

    def save_raw_data(self, filename):
        """Save recorded raw data to CSV"""
        if len(self.raw_log) == 0:
            print(f"No raw data to save for {filename}")
            return
        import csv
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["time"] + [f"ch{i}" for i in range(1, len(self.raw_log[0]))])
            writer.writerows(self.raw_log)
        print(f"Raw data saved to {filename}")

    def stop(self):
        """Stop the thread"""
        self.running = False
        self.join()
        print("EOGReader stopped.")
