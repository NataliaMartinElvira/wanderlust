import time
import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt
import warnings 
import sys 
import math # Necesario para time.perf_counter() en Python 3.7+ (aunque time.perf_counter() ya está en time)

# =========================
# CONFIG AND CONSTANTS
# =========================
FS = 50.0 
LOW, HIGH, ORDER = 0.5, 2.0, 4 # Adjusted bandpass filter settings
VM_PEAK_THR_G = 0.015  # CRITICAL: Increased threshold (0.01g -> 0.015g) to filter noise
MAX_STEP_RATE_SPM = 35   
MIN_STEP_S = 2.0         # Restrict feedback frequency to minimum 2 seconds between steps
MAX_STEP_S = 4.0         
COACH_MIN_AMP_G = 0.03   # Threshold for GOOD vs BAD coaching feedback
BILATERAL_FACTOR = 2.0
SINGLE_IMU_ONE_LEG = True

# --- STREAMING/BUFFER CONTROL ---
MAX_BUFFER_SIZE_SECONDS = 2.0  
BATCH_WINDOW_SECONDS = 0.6     
BATCH_WINDOW_FRAMES = int(FS * BATCH_WINDOW_SECONDS)
MAX_BUFFER_SIZE_FRAMES = int(FS * MAX_BUFFER_SIZE_SECONDS) 


# =========================
# 1. SIGNAL UTILITIES (Core Logic)
# =========================

def bandpass_vm(ax, ay, az, fs=FS, low=LOW, high=HIGH, order=ORDER):
    vm = np.sqrt(ax*ax + ay*ay + az*az)
    vm = vm - np.nanmean(vm)
    if len(vm) < 28: return vm
    sos = butter(order, [low/(fs/2), high/(fs/2)], btype="band", output="sos")
    vm_f = sosfiltfilt(sos, vm)
    return vm_f

def detect_candidate_peaks(signal, fs=FS, min_height=VM_PEAK_THR_G, min_distance_s=0.1):
    min_distance = int(min_distance_s * fs)
    peaks, last_i = [], -10**9
    for i in range(1, len(signal)-1):
        if signal[i] > min_height and signal[i] > signal[i-1] and signal[i] >= signal[i+1]:
            if i - last_i >= min_distance:
                peaks.append(i); last_i = i
    return np.array(peaks, dtype=int)

def merge_step_peaks(peaks, signal, fs=FS, max_step_rate_spm=MAX_STEP_RATE_SPM):
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
    peaks_cand = detect_candidate_peaks(vm_f, fs=fs, min_height=VM_PEAK_THR_G, min_distance_s=0.1)
    peaks_merged = merge_step_peaks(peaks_cand, vm_f, fs=fs, max_step_rate_spm=MAX_STEP_RATE_SPM)
    steps = filter_steps_by_timing(peaks_merged, fs=fs, min_step_s=MIN_STEP_S, max_step_s=MAX_STEP_S)
    return steps


def compute_coaching_flags(raw_df):
    """
    Core analysis: Runs the full detection pipeline on the buffer.
    Returns: (n_steps, last_amp, negative_feedback_bool, step_indices_in_batch)
    """
    if raw_df.empty or len(raw_df) < 28: 
        return (0, 0.0, False, np.array([]))

    ax = pd.to_numeric(raw_df["acc_x_g"],  errors="coerce").values.astype(float)
    ay = pd.to_numeric(raw_df["acc_y_g"],  errors="coerce").values.astype(float)
    az = pd.to_numeric(raw_df["acc_z_g"],  errors="coerce").values.astype(float)

    vm_f = bandpass_vm(ax, ay, az)
    step_idx = detect_steps(vm_f, fs=FS)
    n_steps = len(step_idx)

    if n_steps == 0: 
        return (0, 0.0, False, step_idx)

    latest_amp = vm_f[step_idx[-1]]
    
    # NOTE: Negative feedback is based on amplitude being below the COACHING threshold (0.03g), not the PEAK detection threshold (0.015g)
    amp_low = latest_amp < COACH_MIN_AMP_G 
    negative_feedback = amp_low
    
    return (n_steps, latest_amp, negative_feedback, step_idx)


# =========================
# 2. EXERCISE CLASS 
# =========================

class SeatedMarchLogic:
    def __init__(self):
        self.last_analysis_step_count = 0 
        self.total_session_steps = 0 
        self.data_buffer = pd.DataFrame(columns=["time_ms", "acc_x_g", "acc_y_g", "acc_z_g"])
        self.max_buffer_size = MAX_BUFFER_SIZE_FRAMES 
        
        # --- COOLDOWN CONTROL (CRITICAL) ---
        self.last_feedback_time_march = time.perf_counter() 
        self.MIN_SEND_INTERVAL = 1.5 # 1.5s minimum between sending any feedback event
        
        print("[Logic] Seated/Standing March logic loaded.")

    def check_calmness(self, raw_data_dict):
        """ Used when START_CALIBRATION is received. Clears the buffer. """
        self.last_analysis_step_count = 0 
        self.total_session_steps = 0 
        self.data_buffer = self.data_buffer.iloc[0:0].copy() 
        return True 

    def analyze_performance(self, raw_data_dict):
        """
        Processes real-time data and returns True only when a NEW step is detected.
        
        Returns: True (FEEDBACK:BAD), False (FEEDBACK:GOOD), or None (No Step Event).
        """
        
        # Accessing GLOBAL_STATE 
        try:
            current_state = sys.modules['imu_main_controller'].GLOBAL_STATE['current_exercise']
        except (KeyError, AttributeError):
            current_state = 'SEATED_MARCH' 
            
        if current_state != 'SEATED_MARCH' and current_state != 'STANDING_MARCH':
            return None
        
        if 'sensor2' not in raw_data_dict:
            return None

        # 1. Update Data Buffer
        new_row = {
            "time_ms": time.time() * 1000.0,
            "acc_x_g": raw_data_dict['sensor2']['accel_x'], 
            "acc_y_g": raw_data_dict['sensor2']['accel_y'],
            "acc_z_g": raw_data_dict['sensor2']['accel_z'],
        }
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            self.data_buffer = pd.concat([self.data_buffer, pd.DataFrame([new_row])], ignore_index=True)


        # 2. Run Analysis check (only when enough data is present)
        current_len = len(self.data_buffer)

        if current_len >= BATCH_WINDOW_FRAMES: 
            
            analysis_data = self.data_buffer.iloc[-MAX_BUFFER_SIZE_FRAMES:].copy()
            
            n_steps_in_analysis, latest_amp, negative_feedback, step_idx = compute_coaching_flags(analysis_data) 
            
            # 3. CRITICAL: Check for NEW Step Event
            if n_steps_in_analysis > self.last_analysis_step_count:
                
                # --- STEP 3A: CHECK COOLDOWN (PREVENTS AUDIO SPAM) ---
                t_now = time.perf_counter()
                if t_now - self.last_feedback_time_march < self.MIN_SEND_INTERVAL:
                     # Step detected, but too soon to send feedback -> ignore the event
                     return None 
                
                # Step accepted: Update cooldown timer
                self.last_feedback_time_march = t_now 
                
                # --- STEP 3B: PROCESS AND REPORT ---
                
                # Update global step count
                self.total_session_steps += (n_steps_in_analysis - self.last_analysis_step_count)
                
                # Print event (always useful for debugging consistency)
                amp_status = "Amp Low (BAD)" if negative_feedback else "Amp OK (GOOD)"
                print(f"\n[STEP DETECTED] ✅ Total Steps: {self.total_session_steps} | Last Amp: {latest_amp:.3f}g | Status: {amp_status} (THR={COACH_MIN_AMP_G}g)", flush=True)

                feedback_to_send = negative_feedback # True for BAD, False for GOOD
                
                # 4. Buffer Management
                self.last_analysis_step_count = n_steps_in_analysis
                
                if len(step_idx) > 0:
                    last_step_index_in_batch = step_idx[-1]
                    self.data_buffer = self.data_buffer.iloc[last_step_index_in_batch:].reset_index(drop=True)
                    self.last_analysis_step_count = 0 
                
                return feedback_to_send
            
            # 5. Limit buffer size (if no step was detected but buffer is too large)
            if current_len > self.max_buffer_size:
                self.data_buffer = self.data_buffer.iloc[-self.max_buffer_size:].reset_index(drop=True)
                self.last_analysis_step_count = 0
            
            return None 
            
        return None