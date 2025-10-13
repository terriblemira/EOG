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
DETECT_WINDOW_DURATION = 3
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