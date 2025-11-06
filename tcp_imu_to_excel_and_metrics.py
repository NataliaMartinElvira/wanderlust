# tcp_imu_to_excel_and_metrics.py
# Connects to ESP32 Feather V2 AP (IMU_Logger / imu12345) at 192.168.4.1:3333
# Logs raw IMU to Excel (sheet "IMU") and adds seated-march metrics ("reps", "summary").
# Versión corregida: cuenta solo una repetición por cada ciclo completo (subida + bajada).

import socket, time, datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt

# ---- CONFIG ----
HOST = "192.168.4.1"
PORT = 3333
SHEET_RAW = "IMU"
SHEET_REPS = "reps"
SHEET_SUM  = "summary"
HEADERS = [
    "time_ms","acc_x_g","acc_y_g","acc_z_g",
    "pitch_deg","roll_deg","gyr_x_dps","gyr_y_dps","gyr_z_dps"
]

SAVE_EVERY_SECONDS = 2
SAVE_EVERY_ROWS = 200
RECONNECT_DELAY_S = 2

# ---- DETECTION PARAMS ----
FS = 50.0  # frecuencia de muestreo
LOW, HIGH, ORDER = 0.25, 2.5, 4
VM_PEAK_THR_G = 0.04
MIN_DIST_S = 0.5  # separación mínima entre repeticiones (segundos)


# ---- UTILIDADES ----
def now_ts():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


# ---- FILTRADO VECTOR MAGNITUD ----
def bandpass_vm(ax, ay, az, fs=FS, low=LOW, high=HIGH, order=ORDER):
    vm = np.sqrt(ax*ax + ay*ay + az*az)
    vm = vm - np.nanmean(vm)
    if vm.size < 28:
        return vm
    sos = butter(order, [low/(fs/2), high/(fs/2)], btype="band", output="sos")
    return sosfiltfilt(sos, vm)


# ---- DETECCIÓN DE PICOS ----
def detect_peaks(signal, fs=FS, min_height=VM_PEAK_THR_G, min_distance_s=MIN_DIST_S):
    """
    Detecta picos positivos del vector de aceleración (solo subidas de pierna).
    Evita contar las bajadas como repeticiones adicionales.
    """
    min_distance = int(min_distance_s * fs)
    peaks = []
    last_i = -10**9
    for i in range(1, len(signal)-1):
        # pico positivo
        if signal[i] > min_height and signal[i] > signal[i-1] and signal[i] >= signal[i+1]:
            if i - last_i >= min_distance:
                peaks.append(i)
                last_i = i
    return np.array(peaks, dtype=int)


# ---- PERIODICIDAD ----
def periodicity_score(signal, fs=FS, min_p=0.4, max_p=2.0):
    x = signal - np.mean(signal)
    if len(x) < 5:
        return 0.0
    ac = np.correlate(x, x, mode="full")
    ac = ac[ac.size//2:]
    ac = ac / (ac[0] + 1e-12)
    Lmin = int(min_p*fs)
    Lmax = min(len(ac)-1, int(max_p*fs))
    if Lmax <= Lmin:
        return 0.0
    return float(np.max(ac[Lmin:Lmax]))


# ---- CÁLCULO DE MÉTRICAS ----
def compute_metrics(raw_df):
    if raw_df.empty or len(raw_df) < 30:
        return pd.DataFrame(columns=["t_ms","amp_g"]), {
            "n_reps":0,"cadence_spm":0,"mean_amp_g":0,"sd_amp_g":0,
            "ipi_mean_s":0,"ipi_sd_s":0,"periodicity":0
        }

    t = raw_df["time_ms"].values.astype(float)
    ax = raw_df["acc_x_g"].values.astype(float)
    ay = raw_df["acc_y_g"].values.astype(float)
    az = raw_df["acc_z_g"].values.astype(float)

    vm_f = bandpass_vm(ax, ay, az)
    peaks = detect_peaks(vm_f)

    # ✅ Ajuste: cada ciclo completo (subida+bajada) = 1 repetición
    if len(peaks) > 1:
        peaks = peaks[::2]

    rep_times = t[peaks] if len(peaks) else np.array([])
    amp = vm_f[peaks] if len(peaks) else np.array([])

    cadence = len(peaks) / ((t[-1]-t[0]) / 60000.0) if len(t) > 1 else 0.0
    ipi = np.diff(rep_times) / 1000.0 if len(rep_times) > 1 else np.array([])

    reps_df = pd.DataFrame({"t_ms": rep_times, "amp_g": amp})
    metrics = {
        "n_reps": int(len(peaks)),
        "cadence_spm": float(cadence),
        "mean_amp_g": float(np.mean(amp)) if len(amp) else 0.0,
        "sd_amp_g": float(np.std(amp)) if len(amp) else 0.0,
        "ipi_mean_s": float(np.mean(ipi)) if len(ipi) else 0.0,
        "ipi_sd_s": float(np.std(ipi)) if len(ipi) else 0.0,
        "periodicity": float(periodicity_score(vm_f))
    }
    return reps_df, metrics


# ---- GUARDADO EN EXCEL ----
def save_excel(out_path, raw_df):
    reps_df, m = compute_metrics(raw_df)
    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as w:
        raw_df.to_excel(w, index=False, sheet_name=SHEET_RAW)
        reps_df.to_excel(w, index=False, sheet_name=SHEET_REPS)
        pd.DataFrame([{
            "total_reps": m["n_reps"],
            "cadence_spm": m["cadence_spm"],
            "mean_amp_g": m["mean_amp_g"],
            "sd_amp_g": m["sd_amp_g"],
            "ipi_mean_s": m["ipi_mean_s"],
            "ipi_sd_s": m["ipi_sd_s"],
            "periodicity": m["periodicity"]
        }]).to_excel(w, index=False, sheet_name=SHEET_SUM)
    return len(reps_df), m


# ---- CONEXIÓN TCP ----
def connect():
    while True:
        try:
            print(f"Connecting to {HOST}:{PORT} …")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(10)
            s.connect((HOST, PORT))
            s.settimeout(None)
            print("Connected.")
            return s
        except Exception as e:
            print(f"Connect failed: {e}. Retrying in 2s…")
            time.sleep(2)


# ---- MAIN ----
def main():
    out_path = Path(__file__).parent / f"imu_seated_march_{now_ts()}.xlsx"
    print(f"Saving raw+metrics to: {out_path}")

    rows = []
    buffer = ""
    last_save = time.time()
    header_seen = False

    s = connect()
    try:
        while True:
            data = s.recv(2048)
            if not data:
                raise ConnectionError("Remote closed")
            buffer += data.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line: continue
                parts = [p.strip() for p in line.split(",")]

                if not header_seen:
                    if parts == HEADERS:
                        header_seen = True
                        print("Header received.")
                        continue
                    else:
                        header_seen = True

                if len(parts) != len(HEADERS):
                    continue
                rows.append(parts)

                if (time.time() - last_save >= SAVE_EVERY_SECONDS) or (len(rows) >= SAVE_EVERY_ROWS):
                    df_new = pd.DataFrame(rows, columns=HEADERS)
                    for c in HEADERS:
                        df_new[c] = pd.to_numeric(df_new[c], errors="coerce")
                    if out_path.exists():
                        try:
                            df_exist = pd.read_excel(out_path, sheet_name=SHEET_RAW, engine="openpyxl")
                        except Exception:
                            df_exist = pd.DataFrame(columns=HEADERS)
                        raw_df = pd.concat([df_exist, df_new], ignore_index=True)
                    else:
                        raw_df = df_new
                    reps_count, m = save_excel(out_path, raw_df)
                    rows.clear()
                    last_save = time.time()
                    print(f"Saved: rows={len(raw_df)} | reps={reps_count} | cadence={m['cadence_spm']:.1f} spm | amp={m['mean_amp_g']:.3f} g | rhythm_SD={m['ipi_sd_s']:.2f} s | periodicity={m['periodicity']:.2f}")
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        try:
            s.close()
        except:
            pass
        if rows:
            df_new = pd.DataFrame(rows, columns=HEADERS)
            for c in HEADERS:
                df_new[c] = pd.to_numeric(df_new[c], errors="coerce")
            if out_path.exists():
                try:
                    df_exist = pd.read_excel(out_path, sheet_name=SHEET_RAW, engine="openpyxl")
                except Exception:
                    df_exist = pd.DataFrame(columns=HEADERS)
                raw_df = pd.concat([df_exist, df_new], ignore_index=True)
            else:
                raw_df = df_new
            save_excel(out_path, raw_df)
        print(f"Excel file: {out_path}")


if __name__ == "__main__":
    main()
