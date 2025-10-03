import pygame
import time
import threading
import collections
import numpy as np
from config import *
from eog_reader import EOGReader
from calibration import calculate_normalized_baseline, calculate_channel_norm_factor, calculate_direction_thresholds, plot_calibration_signals
from utils import wait_for_spacebar, expected_from_name, plot_detection_window, save_results
from signal_processing_wavelet import process_signal

def run_calibration(eog_reader, window, font, calibration_sequence, clock):
    """Run calibration to determine baselines and thresholds"""
    # Data structure to store raw signals for each direction
    calibration_data = {
        "left": {"ch1": [], "ch2": [], "ch3": [], "ch8": []},
        "right": {"ch1": [], "ch2": [], "ch3": [], "ch8": []},
        "up": {"ch1": [], "ch2": [], "ch3": [], "ch8": []},
        "down": {"ch1": [], "ch2": [], "ch3": [], "ch8": []},
        "center": {"ch1": [], "ch2": [], "ch3": [], "ch8": []}
    }

    # Clear the detection queue
    eog_reader.out_queue.clear()

    # Instructions for calibration
    if not wait_for_spacebar(window, font, "Press SPACEBAR to begin calibration..."):
        return {
            "baselines": {"H": 0, "V": 0},
            "thresholds": {"left": 0.1, "right": 0.1, "up": 0.1, "down": 0.1},
            "channel_norm_factors": {"ch1": 1, "ch2": 1, "ch3": 1, "ch8": 1}
        }

    # Record data for each target
    for i, (target_name, pos) in enumerate(calibration_sequence):
        target_key = target_name.lower()

        # Add a rest step every 8 targets
        if i > 0 and i % 8 == 0:
            window.fill(BG_COLOR)
            rest_surf = font.render("Rest your eyes. Press SPACEBAR to continue.", True, BLACK)
            window.blit(rest_surf, (WIDTH // 2 - rest_surf.get_width() // 2, HEIGHT // 2))
            pygame.display.flip()
            if not wait_for_spacebar(window, font, "Rest your eyes. Press SPACEBAR to continue..."):
                return {
                    "baselines": {"H": 0, "V": 0},
                    "thresholds": {"left": 0.1, "right": 0.1, "up": 0.1, "down": 0.1},
                    "channel_norm_factors": {"ch1": 1, "ch2": 1, "ch3": 1, "ch8": 1}
                }

        # Show the target
        window.fill(BG_COLOR)
        for name, p in calibration_sequence:
            pygame.draw.circle(window, RED, p, DOT_RADIUS_STATIC)
        pygame.draw.circle(window, BLUE, pos, DOT_RADIUS_ACTIVE)
        instruction = f"Looking at {target_name}..."
        surf = font.render(instruction, True, BLACK)
        window.blit(surf, (10, 50))
        pygame.display.flip()

        # Record EOG data for 3 seconds
        start_time = time.time()
        end_time = start_time + 3.0
        target_ch1, target_ch2, target_ch3, target_ch8 = [], [], [], []
        print(f"Recording {target_name} for 3 seconds...")
        sample_count = 0
        while time.time() < end_time:
            sample, _ = eog_reader.inlet.pull_sample(timeout=0.01)
            if sample is not None:
                sample_count += 1
                target_ch1.append(sample[0])
                target_ch2.append(sample[1])
                target_ch3.append(sample[2])
                target_ch8.append(sample[7])
            pygame.event.pump()
            clock.tick(60)

        print(f"Recorded {len(target_ch1)} samples for {target_name}")

        # Store the raw channels for this target
        if target_ch1 and target_ch2 and target_ch3 and target_ch8:
            calibration_data[target_key]["ch1"].extend(target_ch1)
            calibration_data[target_key]["ch2"].extend(target_ch2)
            calibration_data[target_key]["ch3"].extend(target_ch3)
            calibration_data[target_key]["ch8"].extend(target_ch8)
        else:
            print(f"WARNING: No samples collected for {target_name}!")

    # Calculate channel normalization factors
    channel_norm_factors = {
        "ch1": calculate_channel_norm_factor(calibration_data, "ch1"),
        "ch2": calculate_channel_norm_factor(calibration_data, "ch2"),
        "ch3": calculate_channel_norm_factor(calibration_data, "ch3"),
        "ch8": calculate_channel_norm_factor(calibration_data, "ch8")
    }

    print(f"\nChannel normalization factors: "
          f"ch1={channel_norm_factors['ch1']:.2f}, ch2={channel_norm_factors['ch2']:.2f}, "
          f"ch3={channel_norm_factors['ch3']:.2f}, ch8={channel_norm_factors['ch8']:.2f}")

    # Calculate baselines from normalized signals
    H_baseline = calculate_normalized_baseline(calibration_data, channel_norm_factors, "H")
    V_baseline = calculate_normalized_baseline(calibration_data, channel_norm_factors, "V")
    baselines = {"H": H_baseline, "V": V_baseline}

    # Calculate direction-specific thresholds
    thresholds = calculate_direction_thresholds(calibration_data, channel_norm_factors, baselines)

    # Format thresholds for return
    formatted_thresholds = {
        "left": thresholds.get("left", 0.05),
        "right": thresholds.get("right", 0.05),
        "up": thresholds.get("up", 0.05),
        "down": thresholds.get("down", 0.05)
    }

    plot_calibration_signals(calibration_data, channel_norm_factors, baselines, formatted_thresholds)
    
    return {
        "baselines": baselines,
        "thresholds": formatted_thresholds,
        "channel_norm_factors": channel_norm_factors
    }

def main():
    """Main application function"""
    # Initialize Pygame
    pygame.init()
    window = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Static Jumps + EOG Accuracy Test")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 32)

    # Define target sequences
    center_pos = [WIDTH // 2, HEIGHT // 2]
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

    calibration_sequence = [
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
        ("center", center_pos),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("center", center_pos),
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)]),
        ("center", center_pos),
        ("left", [int(0.05 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("right", [int(0.95 * WIDTH), HEIGHT // 2]),
        ("center", center_pos),
        ("up", [WIDTH // 2, int(0.05 * HEIGHT)]),
        ("center", center_pos),
        ("down", [WIDTH // 2, int(0.95 * HEIGHT)])
    ]

    # Initialize EOG reader
    det_queue = collections.deque(maxlen=50)
    eog = EOGReader(det_queue)
    eog.start()

    # Run calibration
    calibration_params = run_calibration(eog, window, font, calibration_sequence, clock)
    eog.calibration_params = calibration_params

    print(f"\nCalibration complete:")
    print(f"Baselines: {calibration_params['baselines']}")
    print(f"Thresholds: {calibration_params['thresholds']}")
    print(f"Channel norm factors: {calibration_params['channel_norm_factors']}")

    # Wait for user to start test
    if not wait_for_spacebar(window, font, "Calibration complete! Press SPACEBAR to start the test..."):
        return

    # Initialize task state
    step_index = 0
    dot_pos = sequence[0][1]
    step_start = time.time()
    step_max_h = -float('inf')
    step_min_h = float('inf')
    step_max_v = -float('inf')
    step_min_v = float('inf')

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
            if len(eog.latest_H) > 0 and len(eog.latest_V) > 0:
                current_max_h = np.max(np.abs(eog.latest_H))
                current_min_h = np.min(eog.latest_H)
                current_max_v = np.max(np.abs(eog.latest_V))
                current_min_v = np.min(eog.latest_V)
                step_max_h = max(step_max_h, current_max_h)
                step_min_h = min(step_min_h, current_min_h)
                step_max_v = max(step_max_v, current_max_v)
                step_min_v = min(step_min_v, current_min_v)

            # Handle step advancement
            if (now - step_start) >= STEP_DURATION:
                # Plot the detection window for this step before advancing
                plot_detection_window(
                    eog_reader=eog,
                    step_index=step_index,
                    target_name=sequence[step_index][0],
                    expected_direction=current_expected,
                    detection=None,
                    calibration_params=calibration_params
                )

                # Finalize scoring for previous step if no detection captured
                if not step_captured:
                    is_correct = (current_expected["expected_h"] is None and current_expected["expected_v"] is None)
                    running_total += 1
                    running_correct += int(is_correct)
                    trials.append({
                        "step_index": step_index,
                        "target_name": sequence[step_index][0],
                        "expected_h": current_expected["expected_h"],
                        "expected_v": current_expected["expected_v"],
                        "detected_h": None,
                        "detected_v": None,
                        "ts_detected_h": None,
                        "ts_detected_v": None,
                        "correct": is_correct,
                        "h_value_h": 0,
                        "v_value_h": 0,
                        "h_value_v": 0,
                        "v_value_v": 0,
                        "h_velocity_h": 0,
                        "v_velocity_h": 0,
                        "h_velocity_v": 0,
                        "v_velocity_v": 0,
                        "h_value_max": step_max_h,
                        "h_value_min": step_min_h,
                        "v_value_max": step_max_v,
                        "v_value_min": step_min_v,
                        "h_threshold_left": calibration_params['thresholds']['left'],
                        "h_threshold_right": calibration_params['thresholds']['right'],
                        "v_threshold_up": calibration_params['thresholds']['up'],
                        "v_threshold_down": calibration_params['thresholds']['down']
                    })

                # Advance to next step
                step_index += 1
                if step_index >= len(sequence):
                    running = False
                    break

                dot_pos = sequence[step_index][1]
                step_start = now
                step_captured = False
                current_expected = expected_from_name(sequence[step_index][0])
                step_max_h = -float('inf')
                step_min_h = float('inf')
                step_max_v = -float('inf')
                step_min_v = float('inf')

            # Handle detection window
            if not step_captured and (now - step_start) <= RESPONSE_WINDOW:
                first_h_det = None
                first_v_det = None

                # Process all detections in the queue
                while det_queue:
                    det = det_queue.popleft()
                    if det.is_horizontal and first_h_det is None:
                        first_h_det = det
                    elif not det.is_horizontal and first_v_det is None:
                        first_v_det = det

                # If we have at least one detection, mark step as captured
                if first_h_det is not None or first_v_det is not None:
                    step_captured = True
                    running_total += 1

                    # Plot the detection window with the detection marked
                    if first_h_det is not None:
                        plot_detection_window(
                            eog_reader=eog,
                            step_index=step_index,
                            target_name=sequence[step_index][0],
                            expected_direction=current_expected,
                            detection=first_h_det,
                            calibration_params=calibration_params
                        )
                    elif first_v_det is not None:
                        plot_detection_window(
                            eog_reader=eog,
                            step_index=step_index,
                            target_name=sequence[step_index][0],
                            expected_direction=current_expected,
                            detection=first_v_det,
                            calibration_params=calibration_params
                        )

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

                    running_correct += int(is_correct)

                    # Log both detections to trials
                    trials.append({
                        "step_index": step_index,
                        "target_name": sequence[step_index][0],
                        "expected_h": current_expected["expected_h"],
                        "expected_v": current_expected["expected_v"],
                        "detected_h": first_h_det.direction if first_h_det is not None else None,
                        "detected_v": first_v_det.direction if first_v_det is not None else None,
                        "ts_detected_h": float(first_h_det.ts) if first_h_det is not None else None,
                        "ts_detected_v": float(first_v_det.ts) if first_v_det is not None else None,
                        "correct": is_correct,
                        "h_value_h": first_h_det.h_value if first_h_det is not None else 0,
                        "v_value_h": first_h_det.v_value if first_h_det is not None else 0,
                        "h_value_v": first_v_det.h_value if first_v_det is not None else 0,
                        "v_value_v": first_v_det.v_value if first_v_det is not None else 0,
                        "h_velocity_h": first_h_det.h_velocity if first_h_det is not None else 0,
                        "v_velocity_h": first_h_det.v_velocity if first_h_det is not None else 0,
                        "h_velocity_v": first_v_det.h_velocity if first_v_det is not None else 0,
                        "v_velocity_v": first_v_det.v_velocity if first_v_det is not None else 0,
                        "h_value_max": step_max_h,
                        "h_value_min": step_min_h,
                        "v_value_max": step_max_v,
                        "v_value_min": step_min_v,
                        "h_threshold_left": calibration_params['thresholds']['left'],
                        "h_threshold_right": calibration_params['thresholds']['right'],
                        "v_threshold_up": calibration_params['thresholds']['up'],
                        "v_threshold_down": calibration_params['thresholds']['down']
                    })

            # Draw the interface
            window.fill(BG_COLOR)
            for name, pos in sequence:
                pygame.draw.circle(window, RED, pos, DOT_RADIUS_STATIC)
            pygame.draw.circle(window, BLUE, dot_pos, DOT_RADIUS_ACTIVE)

            # Draw center cross
            cx, cy = WIDTH // 2, HEIGHT // 2
            pygame.draw.line(window, BLACK, (cx - CENTER_CROSS, cy), (cx + CENTER_CROSS, cy), 3)
            pygame.draw.line(window, BLACK, (cx, cy - CENTER_CROSS), (cx, cy + CENTER_CROSS), 3)

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
                surf = font.render(line, True, BLACK)
                window.blit(surf, (10, y))
                y += 32

            pygame.display.flip()
            clock.tick(60)

    finally:
        eog.stop()
        save_results(trials, calibration_params)

        # Display completion message
        window.fill(BG_COLOR)
        completion_surf = font.render("Task complete! Press SPACEBAR to exit.", True, BLACK)
        window.blit(completion_surf, (WIDTH // 2 - completion_surf.get_width() // 2, HEIGHT // 2))
        pygame.display.flip()

        # Wait for SPACEBAR to exit
        wait_for_spacebar(window, font, "Task complete! Press SPACEBAR to exit.")
        pygame.quit()

if __name__ == "__main__":
    main()
