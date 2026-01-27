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
from config import RESULTS_DIR
import time
import test
import eog_reader
csv_path = os.path.join(RESULTS_DIR, "eog_trial_results.csv")

start_time = time.time() #M: store start time of program to calculate timepoints later
startOfBreakTime = 0  #M: global variable to store timepoint of break starting
setBreakMarker = False  #M: global variable to mark breaks in data when spacebar pressed
endOfBreakTime = 0  #M: global variable to store timepoint of break ending

def spacebar_pressed(window, font, message="Press SPACEBAR to continue"):
    """Display message and wait for SPACEBAR press"""
    #debug
    print(f"UTILS: Queue type: {type(eog_reader.signal)}")
    print(f"Queue ID: {id(eog_reader.signal)}")
    print(f"Has clear: {hasattr(eog_reader.signal, 'clear')}")
    print(f"EOGReader ID: {id(eog_reader)}")
    last_blink = None
    global startOfBreakTime #M: globals need to be declared AT BEGINNING of functions
    global setBreakMarker
    global endOfBreakTime

    # clear queue before searching for double blinks etc.
    while not eog_reader.signal.empty():
        try:
            eog_reader.signal.get_nowait()
        except:
            break

    window.fill(BG_COLOR)
    instruction_surf = font.render(message, True, WHITE)
    window.blit(instruction_surf, (window.get_width() // 2 - instruction_surf.get_width() // 2,
                                      window.get_height() // 2))
    pygame.display.flip()

    waiting = True
    startOfBreakTime = time.time() - start_time  #M: global variable to store timepoint of break start
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                setBreakMarker = True  #M: set global variable to True when spacebar pressed (to mark breaks in data)
                endOfBreakTime = time.time() - start_time #M: store timepoint of break ending
                waiting = False
        
        while not eog_reader.signal.empty():
            direction = eog_reader.signal.get()
            if direction == "blink":
                current_time = time.time()
                print(f'Utils: first blink added to check for double')
                if last_blink is None:
                    last_blink_time = current_time
                    last_blink = True
                else:
                    time_difference = current_time - last_blink_time
                    if time_difference < 1.5:  #M: double blink within 0.5 seconds
                        print(f"Utils: Double blink detected, skipping test.")
                        test.calib_and_test_completed = True
                        pygame.quit()
                        return False  #M: return False in test.py if double blink detected
                    else: #M: not a double blink, just a single blink
                        last_blink_time = current_time #in case of more than 0.5 s passing in between: old second-blink turns new last-blink
                        print(f'Utils: time_diff {time_difference: .3f} too long')
            # else:
            #     eog_reader.signal.clear()
        pygame.time.delay(100)
    return True # if spacebar pressed --> in test.py: "if not spacebar_pressed:" = "if not True" = "if False" --> skips if --> don't return out of main test function but stay

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

        ch1 = np.array(eog_reader.channel_buffers[0])
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
        ch1 = process_signal(ch1, 250, "ch1") / norm["ch1"]
        ch2 = process_signal(ch2, 250, "ch2") / norm["ch2"]
        ch3 = process_signal(ch3, 250, "ch3") / norm["ch3"]
        ch5 = process_signal(ch5, 250, "ch5") / norm["ch5"]

        # Compute H/V and compensation
        H = (ch1 - ch3) - baselines["H"]
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

def save_results(trials, calibration_params, out_path=None):
    """Save trial results to CSV file"""
    try:
        # Default output path if not provided
        if not out_path or out_path.strip() == "":
            out_dir = os.path.join(os.getcwd(), "results")
            os.makedirs(out_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(out_dir, f"eog_results_{timestamp}.csv")

        else:
            # Ensure directory exists
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "step_index", "target_name", "expected_h", "expected_v",
                "detected_h", "detected_v", "is_blink", "blink_duration",
                "correct", "blink_threshold"
            ])
            writer.writeheader()

            # Write each trial
            for trial in trials:
                clean_row = {
                    "step_index": trial.get("step_index", ""),
                    "target_name": trial.get("target_name", ""),
                    "expected_h": trial.get("expected_h", ""),
                    "expected_v": trial.get("expected_v", ""),
                    "detected_h": trial.get("detected_h", ""),
                    "detected_v": trial.get("detected_v", ""),
                    "is_blink": trial.get("is_blink", False),
                    "correct": trial.get("correct", ""),
                }
                writer.writerow(clean_row)

            # Add summary rows
            total = len(trials)
            correct = sum(1 for r in trials if r.get("correct", False))
            writer.writerow({})
            writer.writerow({
                "target_name": "SUMMARY",
                "expected_h": f"{correct}/{total} ({(correct/total*100.0 if total else 0.0):.1f}%)",
            })

            # Add thresholds row
            writer.writerow({
                "target_name": "THRESHOLDS",
                "expected_h": f"Left: {calibration_params['thresholds']['left']:.4f}, Right: {calibration_params['thresholds']['right']:.4f}",
                "expected_v": f"Up: {calibration_params['thresholds']['up']:.4f}, Down: {calibration_params['thresholds']['down']:.4f}",
                "blink_threshold": f"Blink: {calibration_params.get('blink_threshold', BLINK_THRESHOLD):.4f}"
            })

        print(f"Successfully saved results to {out_path}")
        return True
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
