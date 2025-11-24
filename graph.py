import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
from pathlib import Path

FS = 50.0 # Sampling frequency (Hz)
LOW, HIGH, ORDER = 0.25, 2.5, 4 # Bandpass filter settings
VM_PEAK_THR_G = 0.005             # Candidate peak threshold (g)
MIN_AMP_G = 0.03 # Minimum amplitude for a step to be accepted after merge/timing filter

MAX_STEP_RATE_SPM = 35 # Max step rate (Steps Per Minute)
MIN_STEP_S = 1.0       # Minimum time between steps (s)
MAX_STEP_S = 4.0       # Maximum time between steps (s)

SHEET_RAW = "IMU" # Excel sheet name for raw data

# Signal Functions
def bandpass_vm(ax, ay, az, fs=FS, low=LOW, high=HIGH, order=ORDER):
    # Vector magnitude, mean-centered
    vm = np.sqrt(ax*ax + ay*ay + az*az)
    vm = vm - np.nanmean(vm)
    if len(vm) < 28:
        return vm
    # Bandpass filter
    sos = butter(order, [low/(fs/2), high/(fs/2)], btype="band", output="sos")
    return sosfiltfilt(sos, vm)

def detect_candidate_peaks(signal, fs=FS, min_height=VM_PEAK_THR_G, min_distance_s=0.1):
    """Find positive local maxima above a low threshold."""
    min_dist = int(min_distance_s * fs)
    peaks = []
    last = -10**9

    for i in range(1, len(signal)-1):
        # Check if peak is above height and is a local maximum
        if signal[i] > min_height and signal[i] > signal[i-1] and signal[i] >= signal[i+1]:
            # Check minimum distance from previous accepted peak
            if i - last >= min_dist:
                peaks.append(i)
                last = i
    return np.array(peaks, dtype=int)

def merge_step_peaks(peaks, signal, fs=FS, max_step_rate_spm=MAX_STEP_RATE_SPM):
    """Group very-close peaks (same-step double-peak) and keep the tallest."""
    if len(peaks) == 0:
        return peaks

    min_period = 60.0 / max_step_rate_spm
    cluster_win = int((min_period/2) * fs) # half-period window for clustering

    out = []
    cluster = [peaks[0]]

    for p in peaks[1:]:
        if p - cluster[-1] <= cluster_win:
            cluster.append(p)
        else:
            # Select the tallest peak in the previous cluster
            best = max(cluster, key=lambda i: signal[i])
            out.append(best)
            cluster = [p]

    # Handle the last cluster
    best = max(cluster, key=lambda i: signal[i])
    out.append(best)
    return np.array(out, dtype=int)

def filter_steps(peaks, signal, fs=FS):
    """
    Applies amplitude and timing constraints to merged peaks.
    """
    if len(peaks) == 0:
        return peaks

    out = []
    last_t = None

    for idx in peaks:
        amp = signal[idx]
        # 1. Amplitude filter
        if amp < MIN_AMP_G:
            continue

        t = idx / fs
        # 2. Timing filter
        if last_t is None:
            out.append(idx)
            last_t = t
        else:
            dt = t - last_t
            if dt < MIN_STEP_S:
                continue # Too fast/close
            # if dt > MAX_STEP_S: accept anyway (allows pauses)
            out.append(idx)
            last_t = t

    return np.array(out, dtype=int)

def detect_steps(vm_f):
    """Wrapper function to detect steps through all stages."""
    cand = detect_candidate_peaks(vm_f)
    merged = merge_step_peaks(cand, vm_f)
    final = filter_steps(merged, vm_f)
    return cand, merged, final

# LOAD FILE
folder = Path(__file__).parent
files = list(folder.glob("imu_seated_march_20251122_130549.xlsx"))

if not files:
    print(" No 'imu_seated_march_*.xlsx' files found in the directory.")
    raise SystemExit

FILE_NAME = max(files, key=lambda f: f.stat().st_mtime)
print(f"📄 Loading file: {FILE_NAME.name}")

df = pd.read_excel(FILE_NAME, sheet_name=SHEET_RAW)
df.columns = df.columns.str.strip() # Clean column names

# Extract and convert columns
t = df["time_ms"].values.astype(float)
ax = df["acc_x_g"].values.astype(float)
ay = df["acc_y_g"].values.astype(float)
az = df["acc_z_g"].values.astype(float)

t_s = t / 1000.0 # Convert milliseconds to seconds

vm_f = bandpass_vm(ax, ay, az)
cand, merged, final = detect_steps(vm_f)
# PLOT
plt.figure(figsize=(14, 6))

plt.plot(t_s, vm_f, label="Filtered VM")

if len(cand):
    plt.plot(t_s[cand], vm_f[cand], "o", label="Candidate Peaks", markersize=4)

if len(merged):
    plt.plot(t_s[merged], vm_f[merged], "^", label="Merged Peaks", markersize=6)

if len(final):
    plt.plot(t_s[final], vm_f[final], "x", label="Final Steps", markersize=10)

# Plot the candidate peak threshold
plt.axhline(VM_PEAK_THR_G, color="grey", linestyle="--", label="Threshold")

plt.xlabel("Time (s)")
plt.ylabel("VM Magnitude (g)")
plt.title("Step Detection on Filtered Signal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()