import pygame
import random
import time
import collections
import numpy as np
from collections import deque
from pylsl import StreamInlet, resolve_byprop
from scipy import signal as sig

# ==================== EOG Settings ====================
FS = 250
BUFFER_DURATION = 5
MAX_SAMPLES = FS * BUFFER_DURATION
TOTAL_CHANNELS = 8

# We read all channels then compute H/V from specific ones:
# H = ch1 - ch3 (indices 0 and 2), V = ch8 - ch2 (indices 7 and 1)
# (Matches your script.)
# Filtering
LOWCUT = 0.4
HIGHCUT = 40
FILTER_ORDER = 4

# Eye movement detection
PEAK_DISTANCE = 125
H_THRESH = 95
V_THRESH = 50
MERGE_WINDOW = 500  # in samples

# Refining variables
MIN_CONFIDENCE = 200
GLOBAL_COOLDOWN = 1.2    # seconds between ANY two detections
DETECTION_INTERVAL = 1.5 # seconds between detection passes

# ==================== Snake Settings ====================
CELL = 20                  # pixel size of one grid cell
GRID_W, GRID_H = 30, 22    # grid size (columns x rows)
SPEED = 2                  # grid steps per second
MARGIN = 2                 # grid line thickness (0=off)
FONT_NAME = "consolas"

# Frame (reactive border)
FRAME_THICK = 20
FRAME_IDLE = (55, 55, 55)
FRAME_FLASH = (255, 210, 90)
FLASH_MS = 150

# Derived sizes
PLAY_W, PLAY_H = GRID_W * CELL, GRID_H * CELL
BORDER = FRAME_THICK
WIN_W, WIN_H = PLAY_W + 2 * BORDER, PLAY_H + 2 * BORDER

# Colors
BG = (18, 18, 18)
GRID = (28, 28, 28)
SNAKE = (0, 200, 120)
HEAD = (0, 230, 150)
FOOD = (230, 70, 70)
TEXT = (230, 230, 230)
GHOST = (120, 120, 120)

# Frame sides
TOP, RIGHT, BOTTOM, LEFT = "TOP", "RIGHT", "BOTTOM", "LEFT"

# ==================== EOG Helpers ====================
def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    return sig.butter(order, [low, high], btype='band')

def bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    # zero-phase to avoid phase lag
    return sig.filtfilt(b, a, data, padlen=min(3*max(len(b), len(a)), len(data)-1))

def detect_eye_movements(signal_arr, timestamps):
    """
    signal_arr: shape (N, 2) with columns [H, V]
    timestamps: shape (N,)
    returns: list of (timestamp, direction) with direction in ['left','right','up','down']
    """
    horizontal = signal_arr[:, 0]
    vertical = signal_arr[:, 1]

    h_pos, _ = sig.find_peaks(horizontal, distance=PEAK_DISTANCE, height=H_THRESH)
    h_neg, _ = sig.find_peaks(-horizontal, distance=PEAK_DISTANCE, height=H_THRESH)
    v_pos, _ = sig.find_peaks(vertical, distance=PEAK_DISTANCE, height=V_THRESH)
    v_neg, _ = sig.find_peaks(-vertical, distance=PEAK_DISTANCE, height=V_THRESH)

    h_peaks = sorted([(i, 'pos', abs(horizontal[i])) for i in h_pos] +
                     [(i, 'neg', abs(horizontal[i])) for i in h_neg])
    v_peaks = sorted([(i, 'pos', abs(vertical[i])) for i in v_pos] +
                     [(i, 'neg', abs(vertical[i])) for i in v_neg])

    raw_movements = []
    # Vertical pairings
    for i in range(len(v_peaks) - 1):
        idx1, type1, amp1 = v_peaks[i]
        idx2, type2, amp2 = v_peaks[i + 1]
        if 0 < idx2 - idx1 < PEAK_DISTANCE:
            if type1 == 'pos' and type2 == 'neg':
                raw_movements.append((idx1, 'down', max(amp1, amp2)))
            elif type1 == 'neg' and type2 == 'pos':
                raw_movements.append((idx1, 'up', max(amp1, amp2)))

    # Horizontal pairings
    for i in range(len(h_peaks) - 1):
        idx1, type1, amp1 = h_peaks[i]
        idx2, type2, amp2 = h_peaks[i + 1]
        if 0 < idx2 - idx1 < PEAK_DISTANCE:
            if type1 == 'pos' and type2 == 'neg':
                raw_movements.append((idx1, 'left', max(amp1, amp2)))
            elif type1 == 'neg' and type2 == 'pos':
                raw_movements.append((idx1, 'right', max(amp1, amp2)))

    raw_movements.sort(key=lambda x: x[0])

    # Merge & keep strongest within MERGE_WINDOW
    filtered_movements = []
    i = 0
    while i < len(raw_movements):
        group = [raw_movements[i]]
        j = i + 1
        while j < len(raw_movements) and raw_movements[j][0] - raw_movements[i][0] <= MERGE_WINDOW:
            group.append(raw_movements[j])
            j += 1
        peak = max(group, key=lambda x: x[2])
        filtered_movements.append((timestamps[peak[0]], peak[1]))
        i = j

    return filtered_movements

# ==================== Snake Helpers ====================
def rand_empty_cell(occupied, w, h):
    all_cells = set((x, y) for x in range(w) for y in range(h))
    free = list(all_cells - set(occupied))
    return random.choice(free) if free else None

def draw_grid(surf):
    if MARGIN <= 0:
        return
    for x in range(1, GRID_W):
        X = BORDER + x * CELL
        pygame.draw.line(surf, GRID, (X, BORDER), (X, BORDER + PLAY_H), MARGIN)
    for y in range(1, GRID_H):
        Y = BORDER + y * CELL
        pygame.draw.line(surf, GRID, (BORDER, Y), (BORDER + PLAY_W, Y), MARGIN)

def draw_cell(surf, pos, color):
    x, y = pos
    rect = pygame.Rect(BORDER + x * CELL, BORDER + y * CELL, CELL, CELL)
    pygame.draw.rect(surf, color, rect)

def render_text(surf, txt, size, pos, center=False, color=TEXT):
    font = pygame.font.SysFont(FONT_NAME, size)
    img = font.render(txt, True, color)
    if center:
        surf.blit(img, img.get_rect(center=pos))
    else:
        surf.blit(img, pos)

def draw_frame(surf, flashes):
    def color_for(side):
        t = flashes.get(side, 0)
        if t <= 0:
            return FRAME_IDLE
        alpha = t / FLASH_MS
        r = int(FRAME_IDLE[0] + (FRAME_FLASH[0] - FRAME_IDLE[0]) * alpha)
        g = int(FRAME_IDLE[1] + (FRAME_FLASH[1] - FRAME_IDLE[1]) * alpha)
        b = int(FRAME_IDLE[2] + (FRAME_FLASH[2] - FRAME_IDLE[2]) * alpha)
        return (r, g, b)

    pygame.draw.rect(surf, color_for(TOP), (0, 0, WIN_W, BORDER))
    pygame.draw.rect(surf, color_for(BOTTOM), (0, WIN_H - BORDER, WIN_W, BORDER))
    pygame.draw.rect(surf, color_for(LEFT), (0, 0, BORDER, WIN_H))
    pygame.draw.rect(surf, color_for(RIGHT), (WIN_W - BORDER, 0, BORDER, WIN_H))

# ==================== Main ====================
def main():
    # ---------- Pygame ----------
    pygame.init()
    pygame.display.set_caption("Snake (EOG Controlled)")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()

    # ---------- EOG LSL Setup ----------
    print("Looking for LSL stream 'Explore_8441_ExG'...")
    streams = resolve_byprop('name', 'Explore_8441_ExG', timeout=5.0)
    if not streams:
        raise RuntimeError("Explore_8441_ExG stream not found")
    inlet = StreamInlet(streams[0])
    print("Connected to stream.")

    # Rolling buffers
    channel_buffers = [collections.deque(maxlen=MAX_SAMPLES) for _ in range(TOTAL_CHANNELS)]
    time_buffer = collections.deque(maxlen=MAX_SAMPLES)
    start_time = time.time()
    last_detection_pass = 0.0
    last_any_movement_time = -np.inf

    # ---------- Game State ----------
    def reset():
        start = (GRID_W // 2, GRID_H // 2)
        body = deque([start, (start[0]-1, start[1]), (start[0]-2, start[1])])
        direction = (1, 0)  # initial move right
        score = 0
        highscore = state.get("high", 0)
        food = rand_empty_cell(body, GRID_W, GRID_H)
        return body, direction, food, score, highscore, False, False

    flashes = {TOP: 0, RIGHT: 0, BOTTOM: 0, LEFT: 0}
    def trigger_flash(side):
        flashes[side] = FLASH_MS

    state = {}
    snake, direction, food, score, highscore, paused, dead = reset()
    next_dir = direction
    move_interval = 1.0 / SPEED
    move_timer = 0.0

    # HUD (debug)
    hud_text = "EOG: waiting..."
    hud_conf = 0.0

    print("Streaming, detecting, and playing...")

    while True:
        dt = clock.tick(60) / 1000.0
        now_game = time.time() - start_time

        # ---- Drain LSL samples without blocking ----
        # pull as many as are available right now
        while True:
            sample, ts = inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            # relative time for buffer
            rel_t = time.time() - start_time
            time_buffer.append(rel_t)
            # push all 8 channels
            for i in range(TOTAL_CHANNELS):
                channel_buffers[i].append(sample[i])

        # ---- Handle events (only P/R/Esc) ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE,):
                    pygame.quit()
                    return
                if event.key in (pygame.K_p,):
                    if not dead:
                        paused = not paused
                if event.key in (pygame.K_r,):
                    snake, direction, food, score, highscore, paused, dead = reset()
                    next_dir = direction
                    move_timer = 0.0
                    hud_text = "EOG: waiting..."
                    hud_conf = 0.0
                    for k in flashes:
                        flashes[k] = 0

        # ---- EOG Detection pass (periodic) ----
        # Only attempt detection every DETECTION_INTERVAL seconds if we have enough data
        if (now_game - last_detection_pass > DETECTION_INTERVAL) and (len(time_buffer) >= MAX_SAMPLES):
            last_detection_pass = now_game

            times = np.array(time_buffer)
            ch1 = np.array(channel_buffers[0])
            ch2 = np.array(channel_buffers[1])
            ch3 = np.array(channel_buffers[2])
            ch8 = np.array(channel_buffers[7])

            # Compute H, V
            H = ch1 - ch3
            V = ch8 - ch2

            # Filter
            try:
                H_filt = bandpass_filter(H, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
                V_filt = bandpass_filter(V, LOWCUT, HIGHCUT, FS, FILTER_ORDER)
            except ValueError:
                # Not enough samples for filtfilt padding yet
                H_filt, V_filt = H, V

            sig_array = np.stack((H_filt, V_filt), axis=-1)
            movements = detect_eye_movements(sig_array, times)

            if movements:
                latest_time, direction_str = movements[-1]
                # map to index for confidence
                latest_idx = np.searchsorted(times, latest_time)
                if latest_idx >= len(times):
                    latest_idx = len(times) - 1

                # Confidence per axis
                confidence = (abs(sig_array[latest_idx, 0]) if direction_str in ['left', 'right']
                              else abs(sig_array[latest_idx, 1]))

                # Global cooldown & confidence gate
                if (confidence >= MIN_CONFIDENCE) and ((now_game - last_any_movement_time) > GLOBAL_COOLDOWN):
                    # Enforce axis gating based on current direction
                    dx, dy = direction
                    # If moving horizontally, only accept up/down
                    if dx != 0 and direction_str in ['up', 'down']:
                        proposed = (0, -1) if direction_str == 'up' else (0, 1)
                        # prevent reversing into yourself (not needed here for axis swap, but keep symmetric)
                        if not (proposed[0] == -direction[0] and proposed[1] == -direction[1]):
                            next_dir = proposed
                            if direction_str == 'up':
                                trigger_flash(TOP)
                            else:
                                trigger_flash(BOTTOM)
                            hud_text = f"EOG: {direction_str}"
                            hud_conf = float(confidence)
                            last_any_movement_time = now_game
                            # print for console visibility too
                            print(f"[{latest_time:.2f}s] Direction: {direction_str} | Confidence: {confidence:.1f}")

                    # If moving vertically, only accept left/right
                    elif dy != 0 and direction_str in ['left', 'right']:
                        proposed = (-1, 0) if direction_str == 'left' else (1, 0)
                        if not (proposed[0] == -direction[0] and proposed[1] == -direction[1]):
                            next_dir = proposed
                            if direction_str == 'left':
                                trigger_flash(LEFT)
                            else:
                                trigger_flash(RIGHT)
                            hud_text = f"EOG: {direction_str}"
                            hud_conf = float(confidence)
                            last_any_movement_time = now_game
                            print(f"[{latest_time:.2f}s] Direction: {direction_str} | Confidence: {confidence:.1f}")
                    else:
                        # Ignored due to axis gating
                        pass

        # ---- Update flash timers ----
        dec = int(dt * 1000)
        for k in (TOP, RIGHT, BOTTOM, LEFT):
            flashes[k] = max(0, flashes[k] - dec)

        # ---- Game update ----
        # Prevent reversing into yourself (also covers EOG proposing same axis reverse)
        if (next_dir[0] != -direction[0]) or (next_dir[1] != -direction[1]):
            direction = next_dir

        if not paused and not dead:
            move_timer += dt
            while move_timer >= move_interval:
                move_timer -= move_interval
                head_x, head_y = snake[0]
                dx, dy = direction
                new_head = ((head_x + dx) % GRID_W, (head_y + dy) % GRID_H)

                # Self collision
                if new_head in snake:
                    dead = True
                    highscore = max(highscore, score)
                    state["high"] = highscore
                    break

                snake.appendleft(new_head)

                if new_head == food:
                    score += 1
                    food = rand_empty_cell(snake, GRID_W, GRID_H)
                else:
                    snake.pop()

        # ---- Draw ----
        screen.fill(BG)

        # Playfield background
        pygame.draw.rect(screen, BG, (BORDER, BORDER, PLAY_W, PLAY_H))

        draw_grid(screen)

        if food:
            draw_cell(screen, food, FOOD)

        for i, seg in enumerate(snake):
            draw_cell(screen, seg, HEAD if i == 0 else SNAKE)

        # UI
        render_text(screen, f"Score: {score}", 22, (BORDER + 8, BORDER + 6))
        render_text(screen, f"Best: {max(highscore, score)}", 22, (WIN_W - BORDER - 130, BORDER + 6))
        # HUD with last detection
        render_text(screen, f"{hud_text}  ({hud_conf:.0f})", 18, (BORDER + 8, WIN_H - BORDER - 26), color=GHOST)

        if paused and not dead:
            render_text(screen, "PAUSED", 36, (WIN_W // 2, WIN_H // 2 - 10), center=True, color=GHOST)
            render_text(screen, "Press P to resume", 22, (WIN_W // 2, WIN_H // 2 + 22), center=True, color=GHOST)

        if dead:
            render_text(screen, "GAME OVER", 42, (WIN_W // 2, WIN_H // 2 - 20), center=True)
            render_text(screen, "R: restart   Esc: quit", 22, (WIN_W // 2, WIN_H // 2 + 18), center=True, color=GHOST)

        # Border last (sits on top)
        draw_frame(screen, flashes)

        pygame.display.flip()

if __name__ == "__main__":
    main()
