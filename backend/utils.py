# CONTAINS: important functions that may be needed in multiple modules, f.ex. checking for double_blink (for calibr and test), plotting of detection_plots (for test), saving test results as csv (for test), (spacebar_pressed not used atm, was in between sequences the pause time)
import pygame
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')   # <-- non-interactive backend safe for threads & headless envs
import matplotlib.pyplot as plt
from datetime import datetime
from config import DEBUG_PLOTS, BG_COLOR, WHITE, PLOT_BUFFER_DURATION, BLINK_THRESHOLD
# Create a shared, date-stamped results folder
from datetime import datetime
from config import RESULTS_DIR, DOUBLE_BLINK_COOLDOWN
import time
import test
import eog_reader

csv_path = os.path.join(RESULTS_DIR, "eog_trial_results.csv") #M: may not be needed?

start_time = time.time() #M: store start time of program to calculate timepoints later
startOfBreakTime = 0  #M: global variable to store timepoint of break starting
setBreakMarker = False  #M: global variable to mark breaks in data when spacebar pressed
endOfBreakTime = 0  #M: global variable to store timepoint of break ending

# start pygame for calibration and testing
def init_pygame():
    """Main application function"""
    # Initialize Pygame
    pygame.init()
    calib_and_test_completed = False
    # Get screen dimensions
    screen_info = pygame.display.Info()
    SCREEN_WIDTH, SCREEN_HEIGHT = screen_info.current_w, screen_info.current_h

    window = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN)
    pygame.display.set_caption("Static Jumps + EOG Accuracy Test. Don't double blink unless wanting to exit")

    actual_width, actual_height = pygame.display.get_window_size()
    print(f"Actual window dimensions: {actual_width}x{actual_height}")

    # Use the actual dimensions for everything
    WIDTH = actual_width
    HEIGHT = actual_height
    
    clock = pygame.time.Clock()
    font_size = int(HEIGHT *0.05) # 3% of screen height
    font = pygame.font.SysFont(None, font_size)

    return window, WIDTH, HEIGHT, font, clock, actual_width, actual_height


def check_double_blink(eog_thread, last_blink_time=None, cooldown_end_time=None): #M in brackets r default values (just for 1st call, until get changed)
    blink_detected_time = time.time()

    if cooldown_end_time is not None and blink_detected_time < cooldown_end_time: #M DOUBLE_BLINK_COOLDOWN(=0.5 s) until new one can get detected
        while not eog_thread.signal.empty():
            eog_thread.signal.get()
        return False, last_blink_time, cooldown_end_time

    while not eog_thread.signal.empty():
        direction, timestamp = eog_thread.signal.get()
        if direction == 'blink':
            if last_blink_time is None:
                last_blink_time = blink_detected_time
                return False, last_blink_time, None

            else:
                if blink_detected_time - last_blink_time < 1.5:
                    last_blink_time = None
                    print(f'Utils: Double-blink detected')
                    cooldown_end_time = blink_detected_time + DOUBLE_BLINK_COOLDOWN
                    return True, last_blink_time, cooldown_end_time
                else: # if over 1.5 s
                    time_difference = blink_detected_time - last_blink_time
                    print(f'Utils: time_diff {time_difference: .3f} too long')
                    last_blink_time = blink_detected_time
                    return False, last_blink_time, None
    return False, last_blink_time, cooldown_end_time

def spacebar_pressed(eog_thread, window, font, message="Press SPACEBAR to continue"):
    """Display message and wait for SPACEBAR press"""
    last_blink_time = None
    global startOfBreakTime #M: globals need to be declared AT BEGINNING of functions
    global setBreakMarker
    global endOfBreakTime

    # clear queue before searching for double blinks etc.
    print(f'Utils: Clearing old signals from queue')
    clear_count = 0
    while not eog_thread.signal.empty():
        try:
            eog_thread.signal.get_nowait()
            clear_count += 1
        except:
            break
    print(f'Utils: Cleared {clear_count} old signals from queue')

    window.fill(BG_COLOR)
    instruction_surf = font.render(message, True, WHITE)
    window.blit(instruction_surf, (window.get_width() // 2 - instruction_surf.get_width() // 2,
                                      window.get_height() // 2))
    pygame.display.flip()

    startOfBreakTime = time.time() - start_time  #M: global variable to store timepoint of break start

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                setBreakMarker = True  #M: set global variable to True when spacebar pressed (to mark breaks in data)
                endOfBreakTime = time.time() - start_time #M: store timepoint of break ending
                waiting = False
        
        # while not eog_reader.signal.empty():
        #     direction = eog_reader.signal.get()
        #     if direction == "blink":
        #         current_time = time.time()
        #         print(f'Utils: first blink added to check for double')
        #         if last_blink is None:
        #             last_blink_time = current_time
        #             last_blink = True
        #         else:
        #             time_difference = current_time - last_blink_time
        #             if time_difference < 1.5:  #M: double blink within 0.5 seconds
        #                 print(f"Utils: Double blink detected, skipping test.")
        #                 test.calib_and_test_completed = True
        #                 pygame.quit()
        #                 return False  #M: return False in test.py if double blink detected
        #             else: #M: not a double blink, just a single blink
        #                 last_blink_time = current_time #in case of more than 0.5 s passing in between: old second-blink turns new last-blink
        #                 print(f'Utils: time_diff {time_difference: .3f} too long')
            # else:
            #     eog_reader.signal.clear()
        pygame.time.delay(10) #just alternative to time.sleep() (doesnt make much difference)
    return True # if spacebar pressed --> in test.py: "if not spacebar_pressed:" = "if not True" = "if False" --> skips if --> don't return out of main test function but stay

# used in test.py
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

# used in test.py:
def plot_detection_window( 
    eog_reader,
    step_index=None,
    target_name=None,
    expected_direction=None,
    detection=None,
    calibration_params=None
):
    global startOfBreakTime  #M: to mark breaks in data when spacebar pressed
    global endOfBreakTime
    """
    Plot smooth detection signals from the EOGReader in a single figure with H and V subplots.
    Saves plots to the same RESULTS_DIR as calibration.py.
    """
    from datetime import datetime
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    from signal_processing_wavelet import process_signal
    from config import DEBUG_PLOTS
    from calibration import RESULTS_DIR  # ✅ use shared folder

    if not DEBUG_PLOTS:
        return

    try:
        # Create a "detection_plots" subfolder inside RESULTS_DIR
        save_dir = os.path.join(RESULTS_DIR, "detection_plots")
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_suffix = f"step{step_index}_{target_name or 'unknown'}_{timestamp}"

        # Get rolling buffers
        times = np.array(eog_reader.time_buffer)
        if len(times) == 0:
            print("No data in buffers to plot.")
            return

        ch7 = np.array(eog_reader.channel_buffers[0])
        ch2 = np.array(eog_reader.channel_buffers[1])
        ch3 = np.array(eog_reader.channel_buffers[2])
        ch5 = np.array(eog_reader.channel_buffers[4])

        # Use calibration or defaults
        cal = calibration_params or eog_reader.calibration_params
        norm = cal["channel_norm_factors"]
        baselines = cal["baselines"]
        thresholds = cal["thresholds"]
        alpha = cal.get("alpha", 0.0)

        # Process & normalize
        ch7 = process_signal(ch7, 250, "ch7") / norm["ch7"]
        ch2 = process_signal(ch2, 250, "ch2") / norm["ch2"]
        ch3 = process_signal(ch3, 250, "ch3") / norm["ch3"]
        ch5 = process_signal(ch5, 250, "ch5") / norm["ch5"]

        # Compute H/V and compensation
        H = (ch7 - ch3) - baselines["H"]
        V = (ch5 - ch2) - baselines["V"]
        V_comp = V - alpha * H

        # Smooth with interpolation
        fH = interp1d(times, H, kind="linear")
        fV = interp1d(times, V_comp, kind="linear")
        t_smooth = np.linspace(times[0], times[-1], len(times) * 5)
        H_smooth = fH(t_smooth)
        V_smooth = fV(t_smooth)

        # --- Plot ---
        plt.figure(figsize=(12, 8))

        # H plot
        plt.subplot(2, 1, 1)
        plt.plot(t_smooth, H_smooth, label="H signal")
        plt.axhline(y=thresholds["left"], color="r", linestyle="--", label="Left thr.")
        plt.axhline(y=-thresholds["right"], color="g", linestyle="--", label="Right thr.")
        plt.title(f"H and V Signals (step {step_index} - {target_name})")
        plt.ylabel("H Amplitude")
        if expected_direction:
            plt.text(0.02, 0.9, f"Expected H: {expected_direction.get('expected_h')}", transform=plt.gca().transAxes)
        # if setBreakMarker: #M: mark ending of break (when pressing spacebar) with v line
        #     plt.axvspan(startOfBreakTime, endOfBreakTime, color='b', linestyle='--', label='Break Marker', alpha =0.2)
        #     plt.text(startOfBreakTime + (endOfBreakTime-startOfBreakTime)/2, f"Break", ha='center', color= 'b')
        plt.legend()

        # V plot
        plt.subplot(2, 1, 2)
        plt.plot(t_smooth, V_smooth, label="V signal (alpha-comp.)")
        plt.axhline(y=thresholds["up"], color="r", linestyle="--", label="Up thr.")
        plt.axhline(y=-thresholds["down"], color="g", linestyle="--", label="Down thr.")
        plt.xlabel("Time (s)")
        plt.ylabel("V Amplitude")
        if expected_direction:
            plt.text(0.02, 0.9, f"Expected V: {expected_direction.get('expected_v')}", transform=plt.gca().transAxes)
        plt.legend()

        # Mark detection event if available
        if detection:
            det_time = detection.ts
            if t_smooth[0] <= det_time <= t_smooth[-1]:
                plt.subplot(2, 1, 1)
                plt.axvline(x=det_time, color='k', linestyle='--', label=f'Detection at {det_time:.2f}s')
                plt.legend()
                plt.subplot(2, 1, 2)
                plt.axvline(x=det_time, color='k', linestyle='--', label=f'Detection at {det_time:.2f}s')
                plt.legend()

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"signals_{name_suffix}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"✅ Detection plot saved: {out_path}")

    except Exception as e:
        print(f"❌ Error plotting detection signals: {e}")
        import traceback
        traceback.print_exc()

