# perform test when "if not wait(ing)_for_spacebar (function from utils)", then save data from the test without stopping EOG Thread, incl. test GUI & sequence
#(called main_calib_and_pyg before, moved stuff & changed name)
import pygame
import asyncio
import time
import threading
import collections
import numpy as np
from config import *
from eog_reader import EOGReader
from calibration import run_calibration, run_blink_calibration 
from utils import expected_from_name, plot_detection_window
import utils
# Create a shared, date-stamped results folder
from datetime import datetime
import os
import json
import csv
RESULTS_DIR = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(RESULTS_DIR, exist_ok=True)

#START MAIN-Function
#M: async def main():
def run_test(eog_thread, calibration_params, window, font, clock, WIDTH, HEIGHT,  saved_calibration_data, actual_width, actual_height):
    last_blink_time = None
    cooldown_end_time = None
    test_skipped = False

 # Create a function to display the rest screen with options
    def show_rest_screen(skip_option=False):
        window.fill(BG_COLOR)
        rest_surf = font.render("Calibration complete! Test starts in 10 seconds", True, WHITE)
        window.blit(rest_surf, (WIDTH // 2 - rest_surf.get_width() // 2, HEIGHT // 2 - 50))

        if skip_option:
            skip_surf = font.render("Double blink to skip test", True, WHITE)
            window.blit(skip_surf, (WIDTH // 2 - skip_surf.get_width() // 2, HEIGHT // 2 + 50))
        
        pygame.display.flip()

    show_rest_screen(skip_option=True)

    # Clear old signals before starting rest period
    print("Clearing old signals before rest...")
    cleared = 0
    while not eog_thread.signal.empty():
        try:
            eog_thread.signal.get_nowait()
            cleared += 1
        except:
            break
    print(f"Test: Cleared {cleared} old signals")

        # Wait for 5 seconds or for user input
    rest_start_time = time.time()

    while time.time() - rest_start_time < 10.0:
        is_double, last_blink_time, cooldown_end_time = utils.check_double_blink(eog_thread, last_blink_time, cooldown_end_time)
        if is_double:
            eog_thread.record_raw = False
            eog_thread.save_raw_data(os.path.join(RESULTS_DIR, "main_task_raw_signals.csv"))

            trials = []
            save_results(trials, calibration_params) # M: saving of thresholds etc in save_results (csv-file)
            
            # Display completion message
            window.fill(BG_COLOR)
            completion_surf = font.render("Test skipped! DOUBLE BLINK to exit or wait 20 secs.", True, WHITE)
            window.blit(completion_surf, (WIDTH // 2 - completion_surf.get_width() // 2, HEIGHT // 2))
            pygame.display.flip()
            test_skipped = True

            break

        pygame.event.pump()
        time.sleep(0.01)

            # # Wait for SPACEBAR to exit
            # spacebar_pressed(eog_thread, window, font, message)
            # calib_and_test_completed = True
            # pygame.quit()

    if test_skipped:
        #M: Clear signals again to check for final exit-double-blink
        print("Clearing old signals before exit wait...")
        cleared = 0
        while not eog_thread.signal.empty():
            try:
                eog_thread.signal.get_nowait()
                cleared += 1
            except:
                break
        print(f"Test: Cleared {cleared} old signals")
        
        # Wait for double blink or 20s to exit
        last_blink_time = None #M: resetting after first double blink detection
        rest_start_time = time.time()
        test_skipped = False

        while time.time() - rest_start_time < 20.0:
            is_double, last_blink_time, cooldown_end_time = utils.check_double_blink(eog_thread, last_blink_time, cooldown_end_time)
            if is_double:
                print(f'Double blink detected. Exiting!')
                pygame.quit()
                calib_and_test_completed = True
                
                return eog_thread

            pygame.event.pump()
            time.sleep(0.01)

        pygame.quit()
        print(f'Double blink detected. Exiting!')
        calib_and_test_completed = True
        return eog_thread
        
    # # Wait for user to start test (#M in utils: spacebar_pressed function with double blink to skip test (quit game))
    # # if spacebar not pressed within 100 s delay, exit main function, else: center_pos = ... (go on with test)
    # if not spacebar_pressed(window, font, "Calibration complete! Press SPACEBAR to start the test. Double blink quick to skip test") == True: 
    #     return
    
       # Define target sequences
    center_pos = [WIDTH // 2, HEIGHT // 2]
    # Define dot radii based on window size
    DOT_RADIUS_STATIC = int(min(actual_width, actual_height) * 0.02)
    DOT_RADIUS_ACTIVE = int(min(actual_width, actual_height) * 0.05)
    sequence = [
        ("center", center_pos),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("center", center_pos),
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),
        ("center", center_pos),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
    ]

    # Initialize task state
    step_index = 0
    dot_pos = sequence[0][1]
    step_start = time.time()
    step_max_h = -float('inf')
    step_min_h = float('inf')
    step_max_v = -float('inf')
    step_min_v = float('inf')
    eog_thread.raw_log = []
    eog_thread.record_raw = True

    # Initialize scoring
    trials = []
    running_correct = 0
    running_total = 0
    step_captured = False
    current_expected = expected_from_name(sequence[step_index][0])
    step_detections = []
    direction_stats = {"left": {"correct": 0, "false": 0},
                       "right": {"correct": 0, "false": 0},
                       "up": {"correct": 0, "false": 0},
                       "down": {"correct": 0, "false": 0},}
    running = True

    try:
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            current_time = time.time()

            # Track max and min H/V values
            # if len(eog.latest_H) > 0 and len(eog.latest_V) > 0:
            #     current_max_h = np.max(np.abs(eog.latest_H))
            #     current_min_h = np.min(eog.latest_H)
            #     current_max_v = np.max(np.abs(eog.latest_V))
            #     current_min_v = np.min(eog.latest_V)
            #     step_max_h = max(step_max_h, current_max_h)
            #     step_min_h = min(step_min_h, current_min_h)
            #     step_max_v = max(step_max_v, current_max_v)
            #     step_min_v = min(step_min_v, current_min_v)

            # Initialize detection variables at the start of each loop iteration
            first_h_det = None
            first_v_det = None
            in_cooldown = (current_time - step_start) < ACCURACY_COOLDOWN

            # Process all detections in the queue
            while eog_thread.out_queue:
                det = eog_thread.out_queue.popleft()
                if det.is_horizontal and first_h_det is None:
                    first_h_det = det
                elif not det.is_horizontal and first_v_det is None:
                    first_v_det = det

                if not in_cooldown and not det.is_blink:
                    step_detections.append(det)

            # Handle step advancement
            if (current_time - step_start) >= STEP_DURATION:
                target = sequence[step_index][0]
                is_directional = target in ("left", "right", "up", "down")

                if is_directional:
                    correct_count = sum(1 for d in step_detections if d.direction == target)
                    false_count = sum(1 for d in step_detections if d.direction != target)
                    direction_stats[target]["correct"] += correct_count
                    direction_stats[target]["false"] += false_count

                # Determine if the step is correct
                is_correct = False

                # Check horizontal movement if expected
                if current_expected["expected_h"] is not None and first_h_det is not None:
                    if current_expected["expected_h"] == first_h_det.direction:
                        is_correct = True
                # Check vertical movement if expected
                if current_expected["expected_v"] is not None and first_v_det is not None:
                    if current_expected["expected_v"] == first_v_det.direction:
                        is_correct = True
                # Special case: if we got a detection but none was expected
                if current_expected["expected_h"] is None and first_h_det is not None:
                    is_correct = False
                if current_expected["expected_v"] is None and first_v_det is not None:
                    is_correct = False

                # Log the trial
                trial_data = {
                    "step_index": step_index,
                    "target_name": sequence[step_index][0],
                    "expected_h": current_expected["expected_h"],
                    "expected_v": current_expected["expected_v"],
                    "detected_h": first_h_det.direction if first_h_det is not None else None,
                    "detected_v": first_v_det.direction if first_v_det is not None else None,
                    "is_blink": first_h_det.is_blink if first_h_det is not None and first_h_det.is_blink else
                            (first_v_det.is_blink if first_v_det is not None and first_v_det.is_blink else False),
                    "correct": is_correct if (first_h_det is not None or first_v_det is not None) else
                            (current_expected["expected_h"] is None and current_expected["expected_v"] is None),
                }
                trials.append(trial_data)

                # Plot the detection window for this step (M: for the test)
                plot_detection_window(
                    eog_reader=eog_thread,
                    step_index=step_index,
                    target_name=sequence[step_index][0],
                    expected_direction=current_expected,
                    detection=first_h_det if first_h_det is not None else first_v_det,
                    calibration_params=calibration_params
                )

                # Update counters
                running_total += 1
                if first_h_det is not None or first_v_det is not None:
                    running_correct += int(is_correct)

                # Advance to next step
                step_index += 1
                if step_index >= len(sequence):  #M: EXIT TEST and jump to finally if last sequence done
                    running = False
                    break

                # Reset for next step
                pygame.time.wait(int(500))  # brief pause between steps
                dot_pos = sequence[step_index][1]
                step_start = time.time()
                step_max_h = -float('inf')
                step_min_h = float('inf')
                step_max_v = -float('inf')
                step_min_v = float('inf')
                current_expected = expected_from_name(sequence[step_index][0])

                # Clear the detection queue before the next step
                while eog_thread.out_queue:
                    eog_thread.out_queue.popleft()

            # Draw the interface
            window.fill(BG_COLOR)
            for name, pos in sequence:
                pygame.draw.circle(window, RED, pos, DOT_RADIUS_STATIC)
            pygame.draw.circle(window, BLUE, dot_pos, DOT_RADIUS_ACTIVE)

            # Draw overlays
            def dir_acc_str(d): #direction_accuracy_string
                c = direction_stats[d]["correct"]
                f = direction_stats[d]["false"]
                total = c + f
                pct = (c / total * 100) if total > 0 else 0.0
                return f"{d}: {c}correct {f}false ({pct:.0f}%)"

            acc = (running_correct / running_total * 100.0) if running_total > 0 else 0.0
            overlay_lines = [
                f"Step {step_index+1}/{len(sequence)} | Target: {sequence[step_index][0]} | "
                f"H: {first_h_det.direction if first_h_det is not None else 'None'}, "
                f"V: {first_v_det.direction if first_v_det is not None else 'None'}",
                f"{dir_acc_str('left')}  {dir_acc_str('right')}  {dir_acc_str('up')}  {dir_acc_str('down')}",
                f"Old score: {running_correct}/{running_total} ({acc:.1f}%)"
            ]
            y = 10
            for line in overlay_lines:
                surf = font.render(line, True, WHITE)
                window.blit(surf, (10, y))
                y += 32

            pygame.display.flip()
            clock.tick(FS)


    finally:
#        eog_thread.stop()
        eog_thread.record_raw = False
        eog_thread.save_raw_data(os.path.join(RESULTS_DIR, "main_task_raw_signals.csv"))
        save_results(trials, calibration_params) # M: saving of thresholds etc in save_results (csv-file)

        # Display completion message  
        window.fill(BG_COLOR)
        completion_surf = font.render("Task complete! DOUBLE BLINK to exit or wait 20 secs.", True, WHITE)
        window.blit(completion_surf, (WIDTH // 2 - completion_surf.get_width() // 2, HEIGHT // 2))
        pygame.display.flip()

        # # Wait for SPACEBAR to exit
        # spacebar_pressed(eog_thread, window, font, message)
        # calib_and_test_completed = True
        # pygame.quit()

        # Wait for double blink or 20s to exit
        rest_start_time = time.time()

        while time.time() - rest_start_time < 20.0:
            is_double, last_blink_time, cooldown_end_time = utils.check_double_blink(eog_thread, last_blink_time, cooldown_end_time)
            if is_double:
                pygame.quit()
                calib_and_test_completed = True
                break

            pygame.event.pump()
            time.sleep(0.01)
            
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
                "blink_threshold": f"Blink: {calibration_params['blink_threshold']:.4f}"
            })

        #M: Add accuracy row
            if direction_stats:
                writer.writerow({})
                writer.writerow({"target_name": "DIRECTION ACCURACY"})
                for d, stats in direction_stats.items():
                    total = stats["correct"] + stats["false"]
                    pct = (stats["correct"] / total * 100) if total > 0 else 0.0
                    writer.writerow({
                        "target_name": d,
                        "expected_h": f"Correct: {stats['correct']}",
                        "expected_v": f"False: {stats['false']}",
                        "correct": f"{pct:.1f}%"
                    })

        print(f"Successfully saved results to {out_path}")
        return True
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# #starting EOG for AFTER testing
#     print(f"Restarting EOG Reader for live detection...")
#     eog_new = EOGReader(det_queue)
#     eog_new.calibration_params = calibration_params
#     eog_new.start()

    return eog_thread

#if __name__ == "__main__":
    main()
