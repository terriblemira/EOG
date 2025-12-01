import os
from datetime import datetime
# ==============================
# -------- Settings ------------
# ==============================
# --- Pygame / task ---
BG_COLOR = (50, 50, 50)
RED = (213, 94, 0)
BLUE = (0, 114, 178)
WHITE = (255, 255, 255)
STEP_DURATION = 5.0        # seconds between target jumps
RESPONSE_WINDOW = 3         # seconds after each jump during which we accept the first valid detection
CENTER_CROSS = 30
RESULTS_DIR = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- LSL / signal processing ---
FS = 250 #sampling frequency
PLOT_BUFFER_DURATION = 5
PLOT_MAX_SAMPLES = FS * PLOT_BUFFER_DURATION
DETECT_WINDOW_DURATION = 0.5 # size of window in which signal is processed at once (!! attention: should be smaller than cooldown, so not same signal being shown twice/more)
DETECT_MAX_SAMPLES = int(FS * DETECT_WINDOW_DURATION) #no commas
TOTAL_CHANNELS = 8
CHANNEL_INDICES = [0, 1, 2, 4]  # (we'll form H = ch1 - ch3, V = ch5 - ch2)
LOWCUT = 0.5  # frequencies for the bandpass filter (filters out everything below 0.5 and above 15 Hz) (0.1 to 0.4 to exclude possible slow drift)
HIGHCUT = 15
FILTER_ORDER = 2 # how "strictly" it cuts the excluded frequencies out (bandpass filter not perfect, always lets some "forbidden frequencies" slide through) (the higher the stricter)!!if too high: phases get messed up (like cutting parts out of sin/cos)
MERGE_WINDOW = int(0.12 * FS)  # samples
GLOBAL_COOLDOWN = 1  # seconds between ANY two accepted detections
LSL_STREAM_NAME = 'Explore_8441_ExG'
DETECT_PERIOD = 0.1

# Debug flags
DEBUG_SIGNALS = True
DEBUG_DETECTION = True
DEBUG_PLOTS = True

# blink detection settings
BLINK_CALIBRATION_DURATION = 25  # Total duration for blink calibration (seconds)
BLINK_PROMPT_INTERVAL = 2.5       # Time between blink prompts (seconds)
BLINK_MIN_SAMPLES = 3             # Minimum number of good blink samples needed
BLINK_MAX_SAMPLES = 50            # Maximum number of blink samples to collect
BLINK_THRESHOLD_MULTIPLIER = 1.5  # Multiplier for standard deviation to set threshold
BLINK_MIN_DURATION = 0.075          # Minimum blink duration (seconds)
BLINK_MAX_DURATION = 2         # Maximum blink duration (seconds)
BLINK_THRESHOLD = 2.0             # NOT IN USE!! Default blink detection threshold (can be updated after calibration)
BLINK_COOLDOWN = 0.5             # Minimum time between detected blinks (seconds)

# other constants
H_VELOCITY_THRESHOLD = 0.05
V_VELOCITY_THRESHOLD = 0.05