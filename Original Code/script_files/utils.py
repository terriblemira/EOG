import pygame
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from config import DEBUG_PLOTS, BG_COLOR, BLACK, PLOT_BUFFER_DURATION

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

def plot_detection_window(eog_reader, step_index, target_name, expected_direction, detection=None, calibration_params=None):
    """
    Plot and save the detection window for a specific step
    This plots the full buffer window (5 seconds) for every step
    """
    if not DEBUG_PLOTS:
        return

    try:
        # Create directory for plots if it doesn't exist
        plot_dir = "detection_plots"
        os.makedirs(plot_dir, exist_ok=True)

        # Get current timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use the full buffers for plotting (5 seconds of data)
        times = np.array(eog_reader.full_times)
        H = np.array(eog_reader.full_H)
        V = np.array(eog_reader.full_V)

        # Check if we have data
        if len(times) == 0 or len(H) == 0 or len(V) == 0:
            print(f"No signal data available for step {step_index+1} ({target_name})")
            print(f"Times length: {len(times)}, H length: {len(H)}, V length: {len(V)}")
            return

        # Ensure all arrays have the same length
        min_length = min(len(times), len(H), len(V))
        if min_length == 0:
            print(f"No valid data after length check for step {step_index+1} ({target_name})")
            return

        times = times[-min_length:]
        H = H[-min_length:]
        V = V[-min_length:]

        # Create a figure
        plt.figure(figsize=(12, 8))

        # Plot H signal
        plt.subplot(2, 1, 1)
        plt.plot(times, H, label="H Signal", color='blue')
        plt.axhline(y=calibration_params['thresholds']['right'], color='green', linestyle='--',
                    label=f'Right Threshold={calibration_params["thresholds"]["right"]:.2f}')
        plt.axhline(y=-calibration_params['thresholds']['left'], color='red', linestyle='--',
                    label=f'Left Threshold={calibration_params["thresholds"]["left"]:.2f}')
        plt.title(f"Horizontal Signal (Full {PLOT_BUFFER_DURATION}s Window)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()

        # Plot V signal
        plt.subplot(2, 1, 2)
        plt.plot(times, V, label="V Signal", color='purple')
        plt.axhline(y=calibration_params['thresholds']['down'], color='green', linestyle='--',
                    label=f'Down Threshold={calibration_params["thresholds"]["down"]:.2f}')
        plt.axhline(y=-calibration_params['thresholds']['up'], color='red', linestyle='--',
                    label=f'Up Threshold={calibration_params["thresholds"]["up"]:.2f}')
        plt.title(f"Vertical Signal (Full {PLOT_BUFFER_DURATION}s Window)")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()

        # Add detection markers if available
        if detection:
            detection_time = detection.ts
            if detection.is_horizontal:
                plt.subplot(2, 1, 1)
                plt.axvline(x=detection_time, color='black', linestyle=':',
                           label=f'Detection: {detection.direction} at {detection_time:.2f}s')
                plt.legend()
            else:
                plt.subplot(2, 1, 2)
                plt.axvline(x=detection_time, color='black', linestyle=':',
                           label=f'Detection: {detection.direction} at {detection_time:.2f}s')
                plt.legend()

        # Create metadata string
        expected_dir = ""
        if expected_direction["expected_h"]:
            expected_dir = expected_direction["expected_h"]
        elif expected_direction["expected_v"]:
            expected_dir = expected_direction["expected_v"]
        else:
            expected_dir = "center (no movement)"

        detection_status = "No detection" if not detection else f"Detected: {detection.direction}"

        # Add comprehensive title
        plt.suptitle(f"Step {step_index+1}: {target_name}\nExpected: {expected_dir} | {detection_status}")

        # Adjust layout
        plt.tight_layout()

        # Create filename based on target direction
        direction_category = "center"
        if expected_direction["expected_h"] == "left" or expected_direction["expected_h"] == "right":
            direction_category = "horizontal"
        elif expected_direction["expected_v"] == "up" or expected_direction["expected_v"] == "down":
            direction_category = "vertical"

        # Create subdirectory for direction category
        direction_dir = os.path.join(plot_dir, direction_category)
        os.makedirs(direction_dir, exist_ok=True)

        # Save the plot with metadata in filename
        plot_filename = f"step{step_index+1:03d}_{target_name}_{detection_status.replace(' ', '_')}_{timestamp}.png"
        plot_path = os.path.join(direction_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved detection plot: {plot_path}")

        # Close the figure
        plt.close()

    except Exception as e:
        print(f"Could not save detection plot for step {step_index+1}: {str(e)}")
        import traceback
        traceback.print_exc()

def save_results(trials, calibration_params, out_path="eog_trial_results.csv"):
    """Save trial results to CSV file"""
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
        for row in trials:
            clean_row = {
                "step_index": row.get("step_index", ""),
                "target_name": row.get("target_name", ""),
                "expected_h": row.get("expected_h", ""),
                "expected_v": row.get("expected_v", ""),
                "detected_h": row.get("detected_h", "") if row.get("detected_h") is not None else "",
                "detected_v": row.get("detected_v", "") if row.get("detected_v") is not None else "",
                "ts_detected_h": row.get("ts_detected_h", "") if row.get("ts_detected_h") is not None else "",
                "ts_detected_v": row.get("ts_detected_v", "") if row.get("ts_detected_v") is not None else "",
                "correct": row.get("correct", ""),
                "h_value_h": row.get("h_value_h", 0),
                "v_value_h": row.get("v_value_h", 0),
                "h_value_v": row.get("h_value_v", 0),
                "v_value_v": row.get("v_value_v", 0),
                "h_velocity_h": row.get("h_velocity_h", 0),
                "v_velocity_h": row.get("v_velocity_h", 0),
                "h_velocity_v": row.get("h_velocity_v", 0),
                "v_velocity_v": row.get("v_velocity_v", 0),
                "h_value_max": row.get("h_value_max", 0),
                "h_value_min": row.get("h_value_min", 0),
                "v_value_max": row.get("v_value_max", 0),
                "v_value_min": row.get("v_value_min", 0),
                "h_threshold_left": row.get("h_threshold_left", 0),
                "h_threshold_right": row.get("h_threshold_right", 0),
                "v_threshold_up": row.get("v_threshold_up", 0),
                "v_threshold_down": row.get("v_threshold_down", 0)
            }
            writer.writerow(clean_row)

        # Summary row
        total = len(trials)
        correct = sum(1 for r in trials if r["correct"])
        writer.writerow({})
        writer.writerow({
            "target_name": "SUMMARY",
            "expected_h": f"{correct}/{total} ({(correct/total*100.0 if total else 0.0):.1f}%)",
        })

        # Thresholds row
        writer.writerow({
            "target_name": "THRESHOLDS",
            "expected_h": f"Left: {calibration_params['thresholds']['left']:.4f}, Right: {calibration_params['thresholds']['right']:.4f}",
            "expected_v": f"Up: {calibration_params['thresholds']['up']:.4f}, Down: {calibration_params['thresholds']['down']:.4f}",
        })

    print(f"Saved results to {out_path}")
