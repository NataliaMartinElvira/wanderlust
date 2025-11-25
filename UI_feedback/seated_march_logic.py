import time
import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt
import warnings 

# =========================
# CONSTANTS (From user's validated script)
# =========================
FS = 50.0 
LOW, HIGH, ORDER = 0.15, 2.5, 4 
VM_PEAK_THR_G = 0.01   

# CRITICAL ADJUSTMENTS for Rehabilitation Speed
MIN_STEP_S = 0.5         
MAX_STEP_S = 6.0         
MAX_STEP_RATE_SPM = 35   

SINGLE_IMU_ONE_LEG = True
BILATERAL_FACTOR = 2.0
# Using user's threshold. Feedback is triggered if amplitude < 0.03g
COACH_MIN_AMP_G = 0.03   


# =========================
# 1. SIGNAL UTILITIES (Directly adopted from user's validated code)
# =========================

def bandpass_vm(ax, ay, az, fs=FS, low=LOW, high=HIGH, order=ORDER):
    """Calculates filtered Vector Magnitude minus the mean (to remove gravity bias)."""
    vm = np.sqrt(ax*ax + ay*ay + az*az)
    vm = vm - np.nanmean(vm)
    if len(vm) < 28: return vm
    sos = butter(order, [low/(fs/2), high/(fs/2)], btype="band", output="sos")
    return sosfiltfilt(sos, vm)

def detect_candidate_peaks(signal, fs=FS, min_height=VM_PEAK_THR_G, min_distance_s=0.1):
    """Detects preliminary peaks based on height and distance."""
    min_distance = int(min_distance_s * fs)
    peaks, last_i = [], -10**9
    for i in range(1, len(signal)-1):
        if signal[i] > min_height and signal[i] > signal[i-1] and signal[i] >= signal[i+1]:
            if i - last_i >= min_distance:
                peaks.append(i); last_i = i
    return np.array(peaks, dtype=int)

def merge_step_peaks(peaks, signal, fs=FS, max_step_rate_spm=MAX_STEP_RATE_SPM):
    """Merges peaks that occur too closely together."""
    if len(peaks) == 0: return peaks
    min_step_period = 60.0 / max_step_rate_spm       
    cluster_window = int((min_step_period/2.0) * fs) 
    selected = []
    cluster = [peaks[0]]
    for p in peaks[1:]:
        if p - cluster[-1] <= cluster_window:
            cluster.append(p)
        else:
            best = max(cluster, key=lambda idx: signal[idx])
            selected.append(best)
            cluster = [p]
    selected.append(max(cluster, key=lambda idx: signal[idx]))
    return np.array(selected, dtype=int)

def filter_steps_by_timing(peaks, fs=FS, min_step_s=MIN_STEP_S, max_step_s=MAX_STEP_S):
    """Removes steps that violate min/max timing constraints."""
    if len(peaks) == 0: return peaks
    valid, last_t = [], None
    for idx in peaks:
        t_cur = idx / fs
        if last_t is None: valid.append(idx); last_t = t_cur
        else:
            dt_s = t_cur - last_t
            if dt_s < min_step_s: continue
            valid.append(idx); last_t = t_cur
    return np.array(valid, dtype=int)

def detect_steps(vm_f, fs=FS):
    """Full step detection pipeline."""
    peaks_cand = detect_candidate_peaks(vm_f, fs=fs, min_height=VM_PEAK_THR_G, min_distance_s=0.1)
    peaks_merged = merge_step_peaks(peaks_cand, vm_f, fs=fs, max_step_rate_spm=MAX_STEP_RATE_SPM)
    steps = filter_steps_by_timing(peaks_merged, fs=fs, min_step_s=MIN_STEP_S, max_step_s=MAX_STEP_S)
    return steps


def compute_coaching_flags(raw_df):
    """
    Core analysis: Runs the full detection pipeline on the buffer.
    Returns: (n_steps, last_amp, negative_feedback_bool)
    """
    if raw_df.empty or len(raw_df) < 28: 
        return (0, 0.0, False)

    # Data Extraction (Crucial: converting DataFrame columns to NumPy arrays)
    ax = pd.to_numeric(raw_df["acc_x_g"],  errors="coerce").values.astype(float)
    ay = pd.to_numeric(raw_df["acc_y_g"],  errors="coerce").values.astype(float)
    az = pd.to_numeric(raw_df["acc_z_g"],  errors="coerce").values.astype(float)

    vm_f = bandpass_vm(ax, ay, az)
    step_idx = detect_steps(vm_f, fs=FS)
    n_steps = len(step_idx)

    if n_steps == 0: 
        return (0, 0.0, False)

    # Amplitude check is done on the last detected step
    latest_amp = vm_f[step_idx[-1]]
    
    # Negative feedback only triggered by low amplitude (amp_low)
    amp_low = latest_amp < COACH_MIN_AMP_G
    negative_feedback = amp_low
    
    return (n_steps, latest_amp, negative_feedback)


# =========================
# 2. EXERCISE CLASS (Event-Based Logic)
# =========================

class SeatedMarchLogic:
    """
    IMU logic handler for the Seated March and Standing March exercises.
    Manages the data buffer and triggers feedback events based on NEW steps.
    """
    def __init__(self):
        self.last_reported_step_count = 0 
        # Buffer to store live IMU data for windowed analysis
        self.data_buffer = pd.DataFrame(columns=["time_ms", "acc_x_g", "acc_y_g", "acc_z_g"])
        self.max_buffer_size = int(FS * 10) 
        print("[Logic] Seated/Standing March logic loaded.")

    def check_calmness(self, raw_data_dict):
        """
        Used when START_CALIBRATION is received. Clears the buffer.
        """
        self.last_reported_step_count = 0 
        # CRITICAL: Ensure the buffer is empty and ready for a fresh start.
        self.data_buffer = self.data_buffer.iloc[0:0].copy() 
        return True 

    def analyze_performance(self, raw_data_dict):
        """
        Processes real-time data and returns True only when a NEW step is detected 
        AND that step requires NEGATIVE feedback (Low Amplitude).
        
        Returns: True (FEEDBACK:BAD), False (FEEDBACK:GOOD), or None (No Step Event).
        """
        
        # --- CRITICAL FIX: Only run analysis if the state is active exercise ---
        current_state = __import__('imu_main_controller').GLOBAL_STATE['current_exercise']
        if current_state != 'SEATED_MARCH' and current_state != 'STANDING_MARCH':
            # This ensures that during CALIBRATION or NONE, we DO NOT analyze data,
            # thus preventing false feedback during instruction audios.
            return None
        # ----------------------------------------------------------------------
        
        if not raw_data_dict or 'sensor2' not in raw_data_dict:
            return None

        # 1. Update Data Buffer
        new_row = {
            "time_ms": time.time() * 1000.0,
            "acc_x_g": raw_data_dict['sensor2']['accel_x'], 
            "acc_y_g": raw_data_dict['sensor2']['accel_y'],
            "acc_z_g": raw_data_dict['sensor2']['accel_z'],
        }
        
        # Use pd.concat for robustness, suppressing the known FutureWarning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.data_buffer = pd.concat([self.data_buffer, pd.DataFrame([new_row])], ignore_index=True)


        if len(self.data_buffer) > self.max_buffer_size:
            self.data_buffer = self.data_buffer.iloc[-self.max_buffer_size:]
        
        # 2. Run Analysis on the full buffer window
        if len(self.data_buffer) >= 28: 
            
            n_steps, latest_amp, negative_feedback = compute_coaching_flags(self.data_buffer.copy()) 
            
            # 3. CRITICAL: Check for NEW Step Event
            if n_steps > self.last_reported_step_count:
                
                # A new step was detected (the count increased).
                self.last_reported_step_count = n_steps
                
                # 4. Generate Feedback based on the NEW step
                if negative_feedback:
                    # Send FEEDBACK:BAD (Raise leg higher)
                    return True 
                else:
                    # Send FEEDBACK:GOOD (Well done)
                    return False 
            
            return None # No new event
            
        return None # Not enough data in buffer