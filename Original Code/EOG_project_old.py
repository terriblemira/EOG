# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 17:17:27 2025

@author: Benice & Loe
"""

import numpy as np
import pandas as pd
from scipy import signal as sig
import plotly.graph_objs as go
import plotly.io as pio
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

X = pd.read_csv("full_v1_ExG.csv")
# removing the star and the end of the signal, not useful
X=X.loc[2000:19750] #full_v1_ExG.csv file 

X1 = X['TimeStamp']
X1 = X1-X1.iloc[0]
X2 = X.drop(columns=["TimeStamp"])
print(f"Total duration : {X1.iloc[-1]:.2f} secondes")
# print(X1.head()) 

# Getting rid of most of the noise by substracting and so gathering the horizontal and vertical signals
Y = pd.DataFrame({
    'H': X2['ch1'] - X2['ch3'],
    'V': X2['ch8'] - X2['ch2']})


# plt.figure()
# plt.title('Raw EOG recording - 50%')
# for col in X2:
#     plt.plot(X1, X2[col], label=col) 
# plt.legend()
# plt.grid(1)
# plt.show()


# plt.figure()
# plt.title('Subs EOG recording')
# for col in Y:
#     plt.plot(X1, Y[col], label=col) 
# plt.legend()
# plt.grid(1)
# plt.show()

# Filter Features
fs = 250
lowcut = 0.4
highcut = 40

#%% bandpass filter setting - interactive viewing

def butter_bandpass(lowcut, highcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = sig.filtfilt(b, a, data)
    return y

## Plotly figure
fig = go.Figure()

Y_filtered = Y.copy()
for col in Y.columns:
    Y_filtered[col] = bandpass_filter(Y[col], lowcut, highcut, fs, 4) 

    fig.add_trace(go.Scatter(
        x=X1,
        y=Y_filtered[col],
        mode='lines',
        name=col,
        ))
    
## Interactive graph
fig.update_layout(
    title='Filtered EOG',
    xaxis_title='Time(s)',
    yaxis_title='Amplitude(μV)',
    legend_title='channels',
    hovermode='x unified',
    xaxis=dict(
        rangeslider=dict(visible=True), ## time slider
        type='linear'
        )
    )
pio.renderers.default = 'browser'
fig.show()

#%% PSD to detect blink and interpolation to remove blinks

## Features
window_duration = 2  # seconds
overlap = 0        # seconds
threshold_psd = 3000  # µV^2
signal = Y_filtered['V'].values
timestamps = X1.values
blinking_idx = []


## Sample frequency
fs = 1 / np.mean(np.diff(timestamps))
window_size = int(window_duration * fs)
step_size = int((window_duration - overlap) * fs)
  
## Windows segmentation
for i, start in enumerate(range(0, len(signal) - window_size + 1, step_size)):
    end = start + window_size
    segment = signal[start:end]
    segment_timestamps = timestamps[start:end]

    ## Welch analysis
    frequencies, psd = sig.welch(segment, fs=fs, nperseg=min(1024, len(segment)))

    ## --- New block: improved detection ---
    above_thresh = psd > threshold_psd

    if np.any(above_thresh):
        ## Find continuous bands above the threshold
        freq_step = frequencies[1] - frequencies[0]  # frequency resolution
        start_idx = None
        for j, val in enumerate(above_thresh):
            if val and start_idx is None:
                start_idx = j
            elif not val and start_idx is not None:
                width_hz = (j - start_idx) * freq_step
                if width_hz >= 1.0:
                    break  ## A band >1 Hz detected 
                start_idx = None
        else:
            ## If loop ends without break, no significant blink
            continue

        ## If broadband > 1Hz above threshold: save the window 
        buffer = int(0.3 * fs)
        clean_start = max(start - buffer, 0)
        clean_end = min(end + buffer, len(signal))
        blinking_idx.append((clean_start, clean_end))


## Merge overlapping intervals
merged_indices = []
for start, end in blinking_idx:
    if not merged_indices:
        merged_indices.append([start, end])
    elif start <= merged_indices[-1][1]:
        merged_indices[-1][1] = max(merged_indices[-1][1], end)
    else:
        merged_indices.append([start, end])
        
## Interpolation on segments containing blinks
Vsignal_cleaned = signal.copy()
Hsignal_cleaned =Y_filtered['H'].values.copy()

for start_idx, end_idx in merged_indices:
    if start_idx > 0 and end_idx < len(signal)-1:
        Vsignal_cleaned[start_idx:end_idx] = np.interp(
            np.arange(start_idx, end_idx),
            [start_idx - 1, end_idx],
            [Vsignal_cleaned[start_idx - 1], Vsignal_cleaned[end_idx]]
        )
       
for start_idx, end_idx in merged_indices:
      if start_idx > 0 and end_idx < len(signal) - 1:
          Hsignal_cleaned[start_idx:end_idx] = np.interp(
              np.arange(start_idx, end_idx),
              [start_idx - 1, end_idx],
              [Hsignal_cleaned[start_idx - 1], Hsignal_cleaned[end_idx]]
          )       


## Interactive viewing
fig = go.Figure()

Y_cleaned = tableau = pd.DataFrame({
    'H': Hsignal_cleaned,
    'V': Vsignal_cleaned})

for col in Y_cleaned.columns:
    fig.add_trace(go.Scatter(
            x=X1,
            y=Y_cleaned[col],
            mode='lines',
            name=col,
            ))
## Interactive features
fig.update_layout(
    title='Cleaned EOG',
    xaxis_title='Time(s)',
    yaxis_title='Amplitude(μV)',
    legend_title='channels',
    hovermode='x unified',
    xaxis=dict(
        rangeslider=dict(visible=True), ## time slider
        type='linear'
        )
    )
pio.renderers.default = 'browser'
fig.show()



#%% Detecting gaze direction 

def detect_eye_movements(signal, timestamps, distance=125, h_threshold=115,v_threshold = 60, merge_window=500):
    horizontal = signal.iloc[:, 0]  # horizontal channels
    vertical   = signal.iloc[:, 1]  # vertical channels

    # Detection of positive and negative peaks
    h_peaks_pos, _ = sig.find_peaks(horizontal, distance=distance, height=h_threshold)
    h_peaks_neg, _ = sig.find_peaks(-horizontal, distance=distance, height=h_threshold)
    v_peaks_pos, _ = sig.find_peaks(vertical, distance=distance, height=v_threshold)
    v_peaks_neg, _ = sig.find_peaks(-vertical, distance=distance, height=v_threshold)
   
    v_peaks = sorted([(i, 'pos', abs(vertical[i])) for i in v_peaks_pos] + 
                      [(i, 'neg', abs(vertical[i])) for i in v_peaks_neg])
    h_peaks = sorted([(i, 'pos', abs(horizontal[i])) for i in h_peaks_pos] + 
                      [(i, 'neg', abs(horizontal[i])) for i in h_peaks_neg])

    raw_movements = []

    # Vertical analysis
    for i in range(len(v_peaks) - 1):
        idx1, type1, amp1 = v_peaks[i]
        idx2, type2, amp2 = v_peaks[i + 1]
        if 0 < idx2 - idx1 < distance:
            if type1 == 'pos' and type2 == 'neg':
                raw_movements.append((idx1, 'down', max(amp1, amp2)))
            elif type1 == 'neg' and type2 == 'pos':
                raw_movements.append((idx1, 'up', max(amp1, amp2)))

    # Horizontal analysis
    for i in range(len(h_peaks) - 1):
        idx1, type1, amp1 = h_peaks[i]
        idx2, type2, amp2 = h_peaks[i + 1]
        if 0 < idx2 - idx1 < distance:
            if type1 == 'pos' and type2 == 'neg':
                raw_movements.append((idx1, 'left', max(amp1, amp2)))
            elif type1 == 'neg' and type2 == 'pos':
                raw_movements.append((idx1, 'right', max(amp1, amp2)))

    raw_movements.sort(key=lambda x: x[0])
    filtered_movements = []
    i = 0
    while i < len(raw_movements):
        current = raw_movements[i]
        group = [current]
        j = i + 1
        while j < len(raw_movements) and raw_movements[j][0] - current[0] <= merge_window:
            group.append(raw_movements[j])
            j += 1
        higher_peak = max(group, key=lambda x: x[2])
        filtered_movements.append( (timestamps[higher_peak[0]], higher_peak[0], higher_peak[2], higher_peak[1]) )
        i = j

    return filtered_movements

movements = detect_eye_movements(Y_cleaned, timestamps)

#%% Reporting gaze direction on a 4 bands-grid 


class EyeMovementGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Interface EOG")
        self.root.geometry("600x600")

        self.DEFAULT_COLOR = "lightgray"
        self.ACTIVE_COLOR = "green"

        # Setting the different areas
        self.top = tk.Frame(root, bg=self.DEFAULT_COLOR, height=100)
        self.bottom = tk.Frame(root, bg=self.DEFAULT_COLOR, height=100)
        self.left = tk.Frame(root, bg=self.DEFAULT_COLOR, width=100)
        self.right = tk.Frame(root, bg=self.DEFAULT_COLOR, width=100)
        self.center = tk.Frame(root, bg="white")

        # Positions
        self.top.pack(side="top", fill="x")
        self.bottom.pack(side="bottom", fill="x")
        self.left.pack(side="left", fill="y")
        self.right.pack(side="right", fill="y")
        self.center.pack(expand=True, fill="both")

    def reset_colors(self):
        self.top.config(bg=self.DEFAULT_COLOR)
        self.bottom.config(bg=self.DEFAULT_COLOR)
        self.left.config(bg=self.DEFAULT_COLOR)
        self.right.config(bg=self.DEFAULT_COLOR)

    def highlight(self, direction):
        self.reset_colors()
        if direction == "up":
            self.top.config(bg=self.ACTIVE_COLOR)
        elif direction == "down":
            self.bottom.config(bg=self.ACTIVE_COLOR)
        elif direction == "left":
            self.left.config(bg=self.ACTIVE_COLOR)
        elif direction == "right":
            self.right.config(bg=self.ACTIVE_COLOR)

    def play_movements(self, movements, delay_ms=500):
        # Sequential playback of movements
        def step(index):
            if index < len(movements):
                _, direction = movements[index]
                self.highlight(direction)
                self.root.after(delay_ms, lambda: step(index + 1))
            # else:
            #     print("All the movements have been read.")
            #     self.root.quit()
        step(0)
        
# #Creating a list containing the detect movements in the signal      
# movements = detect_eye_movements(Y_cleaned)


# # === In console  ====================
# # Launch interface
# root = tk.Tk()
# app = EyeMovementGUI(root)

# # Playback detected movements after 1s
# root.after(1000, lambda: app.play_movements(movements, delay_ms=800))
# root.mainloop()



#%%#%% gaze range detection (OOP version)  - threshold & 9-grid with dot

class EyeTrackingGUI:
    def __init__(self, root, signal_df, fs, movements):
        #Initializing each content 
        self.root = root
        self.signal_df = signal_df
        self.fs = fs
        self.movements = movements
        self.current_index = 0
        self.point_pos = [1, 1]  # Initial position
        self.cell_size = 100

        self.root.title("Suivi du regard (3x3) + Signal")

        self.canvas = self.create_grid_canvas()
        self.point_id = self.draw_point(self.point_pos)

        self.fig, self.ax = self.create_matplotlib_figure()
        self.canvas_fig = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_fig.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.root.after(800, self.animate)

    # creating 9 squared-grid with a dot following the gaze direction
    def create_grid_canvas(self):
        canvas = tk.Canvas(self.root, width=3 * self.cell_size, height=3 * self.cell_size, bg="white")
        canvas.pack()
        for i in range(4):
            canvas.create_line(i * self.cell_size, 0, i * self.cell_size, 3 * self.cell_size)
            canvas.create_line(0, i * self.cell_size, 3 * self.cell_size, i * self.cell_size)
        return canvas

    def draw_point(self, pos):
        x = pos[0] * self.cell_size + self.cell_size // 2
        y = pos[1] * self.cell_size + self.cell_size // 2
        r = 7
        return self.canvas.create_oval(x - r, y - r, x + r, y + r, fill='red')

    def move_point(self, old_pos, new_pos):
        dx = (new_pos[0] - old_pos[0]) * self.cell_size
        dy = (new_pos[1] - old_pos[1]) * self.cell_size
        self.canvas.move(self.point_id, dx, dy)

    # plotting the graph with the signal and vertical asympotes on every detected movement with the direction written
    def create_matplotlib_figure(self):
        fig = Figure(figsize=(8, 3), dpi=100)
        ax = fig.add_subplot(111)
        t = np.arange(len(self.signal_df)) / self.fs
        ax.plot(t, self.signal_df['H'], label='Horizontal (H)', color='blue')
        ax.plot(t, self.signal_df['V'], label='Vertical (V)', color='green')

        colors = {'up': 'green', 'down': 'red', 'left': 'blue', 'right': 'orange'}
        for time, idx, amplitude, direction in self.movements:
            ax.axvline(x=idx / self.fs, color=colors.get(direction, 'black'), linestyle='--', alpha=0.5)
            ax.text(idx / self.fs,
                    max(self.signal_df['H'].max(), self.signal_df['V'].max()) * 0.9,
                    f"{direction}\n{int(amplitude)}",
                    rotation=90, verticalalignment='top', fontsize=7, color=colors.get(direction, 'black'))
        ax.set_title("EOG signal with eye movements")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend(fontsize=8)
        return fig, ax

    # updating the plot each new detected movement
    def update_matplotlib(self):
        self.ax.clear()
        t = np.arange(len(self.signal_df)) / self.fs
        self.ax.plot(t, self.signal_df['H'], label='Horizontal (H)', color='blue')
        self.ax.plot(t, self.signal_df['V'], label='Vertical (V)', color='green')

        colors = {'up': 'green', 'down': 'red', 'left': 'blue', 'right': 'orange'}
        for i, (time, idx, amplitude, direction) in enumerate(self.movements):
            alpha = 1.0 if i == self.current_index else 0.1
            self.ax.axvline(x=idx / self.fs, color=colors.get(direction, 'black'), linestyle='--', alpha=alpha)
            if i == self.current_index:
                self.ax.text(idx / self.fs,
                             max(self.signal_df['H'].max(), self.signal_df['V'].max()) * 0.9,
                             f"{direction}\n{int(amplitude)}",
                             rotation=90, verticalalignment='top', fontsize=8,
                             color=colors.get(direction, 'black'))

        self.ax.set_title("EOG signal with eye movements")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")
        self.ax.legend(fontsize=8)
        self.fig.canvas.draw_idle()

    # defining the dot change of position depending of the movement range
    def sorting_movements(self, direction, amplitude):
        if direction in ['left', 'right']:
            seuil = 300.0
        elif direction in ['up', 'down']:
            seuil = 110.0
        else:
            seuil = 150.0
        return 'small' if amplitude < seuil else 'large'

    def get_grid_position(self, direction, size, current_pos):
        x, y = current_pos
        if direction == 'left':
            return (0, y) if size == 'large' else (max(0, x - 1), y)
        elif direction == 'right':
            return (2, y) if size == 'large' else (min(2, x + 1), y)
        elif direction == 'down':
            return (x, 2) if size == 'large' else (x, min(2, y + 1))
        elif direction == 'up':
            return (x, 0) if size == 'large' else (x, max(0, y - 1))
        return (1, 1)

    # animating the grid
    def animate(self):
        if self.current_index >= len(self.movements):
            self.move_point(self.point_pos, (1, 1))
            self.canvas.update()
            self.root.after(1500, self.root.quit)
            return

        time, idx, amplitude, direction = self.movements[self.current_index]
        size = self.sorting_movements(direction, amplitude)
        new_pos = self.get_grid_position(direction, size, self.point_pos)

        self.move_point(self.point_pos, new_pos)
        self.point_pos = new_pos
        self.update_matplotlib()

        self.root.title(f"Eye tracking - Movement {self.current_index + 1}/{len(self.movements)} : "
                        f"{direction} ({amplitude:.1f}, {size})")
        self.current_index += 1
        self.root.after(800, self.animate)
        

# #Interface launching for the console 
# root = tk.Tk()
# app = EyeTrackingGUI(root, Y_cleaned, fs, movements)
# root.mainloop()
