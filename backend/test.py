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
from utils import spacebar_pressed, expected_from_name, plot_detection_window, save_results
import utils
# Create a shared, date-stamped results folder
from datetime import datetime
import os
import json

RESULTS_DIR = os.path.join("results", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(RESULTS_DIR, exist_ok=True)

#START MAIN-Function
#M: async def main():
def main():
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

    # Initialize EOG reader
    det_queue = collections.deque(maxlen=50)
    eog_thread = EOGReader(det_queue) #creating an instance of EOGReader class with det_queue as argument (used in the __init__ method (--> variable self.out_queue IS det_queue for this EOGReader instance (for eog_thread).)
    #M added:
    #M:MAYBE back in: eog.eventLoop = asyncio.get_event_loop() #M: Websocket Setup
    #await eog.connect_to_webapp() #M: verbindet & hält Verbindung zu app.py
    eog_thread.start() #M: start eog_reader (thread) with default calibration_params (default thresholds, etc.)

    # Run calibration

    eog_thread.raw_log = []
    eog_thread.record_raw = True
    calibration_params = run_calibration(eog_thread, window, font, clock, WIDTH, HEIGHT) #runs function with parameters in brackets and saves outcome as "(test.)calibration_params" (eog.calibration_params not changed yet!)
    eog_thread.record_raw = False
    eog_thread.save_raw_data(os.path.join(RESULTS_DIR, "calibration_raw_signals.csv"))
    #samples, timestamps = eog_thread.inlet.pull_chunk(timeout=0.01)
    #eog_thread.calibration_params = calibration_params # Update calibration params in EOG Reader from default to new
    #M: idea for saved csv instead of live: from utils import startOfBreakingTime, endOfBreakingTime) "while startOfBreakingTime is not 0: get startOfBreakingTime" - startofBreakingTime and save in csv alongside raw data"
    
    eog_thread.out_queue.clear()
    eog_thread.raw_log = []
    eog_thread.record_raw = True
    blink_calibration_results = run_blink_calibration(eog_thread, window, font, clock, calibration_params, WIDTH, HEIGHT)
    eog_thread.record_raw = False
    eog_thread.save_raw_data(os.path.join(RESULTS_DIR, "blink_calibration_raw_signals.csv"))
    calibration_params['blink_threshold'] = blink_calibration_results['blink_threshold']
    calibration_params_file = os.path.join(RESULTS_DIR, "calibration_parameters.json") # saving calibration parameters to json file
    with open(calibration_params_file, 'w') as f:
        json.dump(calibration_params, f, indent=4)
    print(f"Saved calibration parameters to {calibration_params_file}")
    eog_thread.calibration_params = calibration_params # Update calibration params in EOG Reader from default to new

    print(f"\nCalibration complete:")
    print(f"Baselines: {calibration_params['baselines']}")
    print(f"Thresholds: {calibration_params['thresholds']}")
    print(f"Channel norm factors: {calibration_params['channel_norm_factors']}")
    print(f"Alpha: {calibration_params['alpha']:.4f}")

 # Create a function to display the rest screen with options
    def show_rest_screen(skip_option=False):
        window.fill(BG_COLOR)
        rest_surf = font.render("Calibration complete! Test starts in 5 seconds", True, WHITE)
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
    redo_last_steps = False
    last_blink_time = None

    while time.time() - rest_start_time < 5.0:
        is_double, last_blink_time = utils.check_double_blink(last_blink_time)
        if is_double:
            pygame.quit()
            print(f'Utils/Test: double blink detected. Skip test.')
            break

        pygame.event.pump()
        time.sleep(0.01)
        
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
    running = True

    try:
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            now = time.time()

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

            # Process all detections in the queue
            while det_queue:
                det = det_queue.popleft()
                if det.is_horizontal and first_h_det is None:
                    first_h_det = det
                elif not det.is_horizontal and first_v_det is None:
                    first_v_det = det

            # Handle step advancement
            if (now - step_start) >= STEP_DURATION:
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
                if step_index >= len(sequence):
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
                while det_queue:
                    det_queue.popleft()

            # Draw the interface
            window.fill(BG_COLOR)
            for name, pos in sequence:
                pygame.draw.circle(window, RED, pos, DOT_RADIUS_STATIC)
            pygame.draw.circle(window, BLUE, dot_pos, DOT_RADIUS_ACTIVE)

            # Draw overlays
            acc = (running_correct / running_total * 100.0) if running_total > 0 else 0.0
            overlay_lines = [
                f"Step {step_index+1}/{len(sequence)} | Target: {sequence[step_index][0]} | "
                f"H det: {first_h_det.direction if 'first_h_det' in locals() and first_h_det is not None else 'None'}, "
                f"V det: {first_v_det.direction if 'first_v_det' in locals() and first_v_det is not None else 'None'}",
                f"Max H: {step_max_h:.2f}, Max V: {step_max_v:.2f}",
                f"Score: {running_correct}/{running_total} ({acc:.1f}%)"
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
        completion_surf = font.render("Task complete! Press SPACEBAR to exit.", True, WHITE)
        window.blit(completion_surf, (WIDTH // 2 - completion_surf.get_width() // 2, HEIGHT // 2))
        pygame.display.flip()

        # Wait for SPACEBAR to exit
        spacebar_pressed(window, font, "Task complete! Press SPACEBAR to exit.")
        calib_and_test_completed = True
        pygame.quit()

# #starting EOG for AFTER testing
#     print(f"Restarting EOG Reader for live detection...")
#     eog_new = EOGReader(det_queue)
#     eog_new.calibration_params = calibration_params
#     eog_new.start()

    return eog_thread

#if __name__ == "__main__":
    main()
