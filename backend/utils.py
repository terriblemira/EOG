import pygame
import os
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')   # <-- non-interactive backend safe for threads & headless envs
import matplotlib.pyplot as plt
from datetime import datetime
from config import DEBUG_PLOTS, BG_COLOR, BLACK, PLOT_BUFFER_DURATION
# Create a shared, date-stamped results folder
from datetime import datetime
import os
from calibration import RESULTS_DIR
csv_path = os.path.join(RESULTS_DIR, "eog_trial_results.csv")

def wait_for_spacebar(window, font, message="Press SPACEBAR to continue"):
    """Display message and wait for SPACEBAR press"""
    window.fill(BG_COLOR)
    instruction_surf = font.render(message, True, BLACK)
    window.blit(instruction_surf, (window.get_width() // 2 - instruction_surf.get_width() // 2,
                                      window.get_height() // 2))
    pygame.display.flip()
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                waiting = False
        pygame.time.delay(100)
    return True

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
        ch8 = np.array(eog_reader.channel_buffers[7])

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
        ch8 = process_signal(ch8, 250, "ch8") / norm["ch8"]

        # Compute H/V and compensation
        H = (ch1 - ch3) - baselines["H"]
        V = (ch8 - ch2) - baselines["V"]
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
                "detected_h", "detected_v", "ts_detected_h", "ts_detected_v",
                "correct", "h_value_h", "v_value_h", "h_value_v", "v_value_v",
                "h_velocity_h", "v_velocity_h", "h_velocity_v", "v_velocity_v",
                "h_value_max", "h_value_min", "v_value_max", "v_value_min",
                "h_threshold_left", "h_threshold_right", "v_threshold_up", "v_threshold_down"
            ])
            writer.writeheader()

            # Write each trial
            for trial in trials:
                # Create a clean row with proper handling of None values
                clean_row = {
                    "step_index": trial.get("step_index", ""),
                    "target_name": trial.get("target_name", ""),
                    "expected_h": trial.get("expected_h", ""),
                    "expected_v": trial.get("expected_v", ""),
                    "detected_h": trial.get("detected_h", ""),
                    "detected_v": trial.get("detected_v", ""),
                    "ts_detected_h": trial.get("ts_detected_h", ""),
                    "ts_detected_v": trial.get("ts_detected_v", ""),
                    "correct": trial.get("correct", ""),
                    "h_value_h": trial.get("h_value_h", 0),
                    "v_value_h": trial.get("v_value_h", 0),
                    "h_value_v": trial.get("h_value_v", 0),
                    "v_value_v": trial.get("v_value_v", 0),
                    "h_velocity_h": trial.get("h_velocity_h", 0),
                    "v_velocity_h": trial.get("v_velocity_h", 0),
                    "h_velocity_v": trial.get("h_velocity_v", 0),
                    "v_velocity_v": trial.get("v_velocity_v", 0),
                    "h_value_max": trial.get("h_value_max", 0),
                    "h_value_min": trial.get("h_value_min", 0),
                    "v_value_max": trial.get("v_value_max", 0),
                    "v_value_min": trial.get("v_value_min", 0),
                    "h_threshold_left": trial.get("h_threshold_left", 0),
                    "h_threshold_right": trial.get("h_threshold_right", 0),
                    "v_threshold_up": trial.get("v_threshold_up", 0),
                    "v_threshold_down": trial.get("v_threshold_down", 0)
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
            })

        print(f"Successfully saved results to {out_path}")
        return True
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
