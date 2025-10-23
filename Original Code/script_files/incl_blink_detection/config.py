# ==============================
# -------- Settings ------------
# ==============================
import os
from datetime import datetime
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
RESULTS_DIR = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- LSL / signal processing ---
FS = 500
PLOT_BUFFER_DURATION = 5
PLOT_MAX_SAMPLES = FS * PLOT_BUFFER_DURATION
DETECT_WINDOW_DURATION = 3
DETECT_MAX_SAMPLES = FS * DETECT_WINDOW_DURATION
TOTAL_CHANNELS = 8
CHANNEL_INDICES = [0, 1, 2, 7]  # (we'll form H = ch1 - ch3, V = ch8 - ch2)
LOWCUT = 0.1
HIGHCUT = 40
FILTER_ORDER = 4
MERGE_WINDOW = int(0.12 * FS)  # samples
GLOBAL_COOLDOWN = 0.8  # seconds between ANY two accepted detections
LSL_STREAM_NAME = 'Explore_8441_ExG'

# Debug flags
DEBUG_SIGNALS = True
DEBUG_DETECTION = True
DEBUG_PLOTS = True

# blink detection settings
BLINK_CALIBRATION_DURATION = 25  # Total duration for blink calibration (seconds)
BLINK_PROMPT_INTERVAL = 2.5       # Time between blink prompts (seconds)
BLINK_MIN_SAMPLES = 3             # Minimum number of good blink samples needed
BLINK_MAX_SAMPLES = 50            # Maximum number of blink samples to collect
BLINK_THRESHOLD_MULTIPLIER = 1  # Multiplier for standard deviation to set threshold
BLINK_MIN_DURATION = 0.05          # Minimum blink duration (seconds)
BLINK_MAX_DURATION = 0.5         # Maximum blink duration (seconds)
BLINK_THRESHOLD = 2.0             # Default blink detection threshold (can be updated after calibration)
BLINK_COOLDOWN = 0.1             # Minimum time between detected blinks (seconds)