# live_classify.py
# Usage:
#   python live_classify.py --host 192.168.4.1 --port 3333 --model model.pkl --fs 10 --win 2.0 --step 1.0
#
# Prints a prediction every 'step' seconds, and logs to live_preds.csv

import argparse, socket, time, datetime as dt
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.signal import butter, filtfilt, welch

HEADERS = ["time_ms","acc_x_g","acc_y_g","acc_z_g","pitch_deg","roll_deg","gyr_x_dps","gyr_y_dps","gyr_z_dps"]
NUM_COLS = HEADERS[1:]  # all but time

def butter_lowpass(cut, fs, order=3):
    from scipy.signal import butter
    b, a = butter(order, cut/(fs/2.0), btype="low")
    return b, a

def magnitude(x, y, z): return np.sqrt(x*x + y*y + z*z)
def zcr(x): return float(((x[:-1]*x[1:])<0).sum())/max(1,len(x)-1)
def iqr(x): return float(np.percentile(x,75)-np.percentile(x,25))
def mad(x): return float(np.median(np.abs(x-np.median(x))))
def rms(x): return float(np.sqrt(np.mean(np.square(x))))
def safe_dom_freq(x, fs):
    from scipy.signal import welch
    f, pxx = welch(x, fs=fs, nperseg=min(len(x), max(8, int(fs))))
    if len(pxx)==0 or np.allclose(pxx,0): return 0.0
    return float(f[np.argmax(pxx)])
def spec_energy(x):
    X = np.fft.rfft(x-np.mean(x)); P=np.abs(X)**2; d=np.sum(P)
    return float(np.sum(P/d)) if d>0 else 0.0

def extract_window_feats(win_df, fs):
    feats = {}
    acc_mag = magnitude(win_df["acc_x_g"], win_df["acc_y_g"], win_df["acc_z_g"])
    gyr_mag = magnitude(win_df["gyr_x_dps"], win_df["gyr_y_dps"], win_df["gyr_z_dps"])
    series = {c: win_df[c].values.astype(float) for c in NUM_COLS}
    series["acc_mag"] = acc_mag; series["gyr_mag"] = gyr_mag

    for name, x in series.items():
        feats[f"{name}_mean"]=float(np.mean(x))
        feats[f"{name}_std"]=float(np.std(x, ddof=1)) if len(x)>1 else 0.0
        feats[f"{name}_min"]=float(np.min(x))
        feats[f"{name}_max"]=float(np.max(x))
        feats[f"{name}_rms"]=rms(x)
        feats[f"{name}_iqr"]=iqr(x)
        feats[f"{name}_mad"]=mad(x)
        feats[f"{name}_zcr"]=zcr(x)
        feats[f"{name}_domf"]=safe_dom_freq(x, fs)
        feats[f"{name}_spen"]=spec_energy(x)
    feats["acc_sma"]=float(np.mean(np.abs(series["acc_x_g"])+np.abs(series["acc_y_g"])+np.abs(series["acc_z_g"])))
    feats["gyr_sma"]=float(np.mean(np.abs(series["gyr_x_dps"])+np.abs(series["gyr_y_dps"])+np.abs(series["gyr_z_dps"])))
    return feats

def preprocess_resample(buf_df, fs, lp_cut=4.0, order=4):
    import numpy as np
    import pandas as pd
    from scipy.signal import butter, filtfilt, lfilter

    df = buf_df.copy()

    NUM_COLS = ["acc_x_g","acc_y_g","acc_z_g","gyr_x_dps","gyr_y_dps","gyr_z_dps","pitch_deg","roll_deg"]

    # Ensure numeric & sorted
    for c in ["time_ms"] + NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time_ms"]).sort_values("time_ms")
    if len(df) < 3:
        return None

    # seconds since start
    t = (df["time_ms"].values - df["time_ms"].values[0]) / 1000.0
    if t[-1] <= 0:
        return None

    # Uniform grid at fs
    grid = np.arange(0.0, t[-1], 1.0 / fs)
    if len(grid) < 3:
        return None

    out = {"time_s": grid}
    for c in NUM_COLS:
        out[c] = np.interp(grid, t, df[c].values.astype(float))
    out = pd.DataFrame(out)

    # Low-pass
    b, a = butter(order, lp_cut / (fs / 2.0), btype="low")

    # filtfilt needs n > padlen; default padlen = 3*(max(len(a), len(b)) - 1)
    default_padlen = 3 * (max(len(a), len(b)) - 1)

    n = len(out)
    if n <= default_padlen + 1:
        # Not enough samples yet — tell caller to wait a bit more
        # (this avoids the crash entirely)
        return None

    # Guard pad length so it's < n-1
    safe_padlen = min(default_padlen, n - 2)

    # Apply filter safely
    for c in NUM_COLS:
        try:
            out[c] = filtfilt(b, a, out[c], padlen=safe_padlen)
        except ValueError:
            # Paranoid fallback: use causal filter if something odd happens
            out[c] = lfilter(b, a, out[c])

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.4.1")
    ap.add_argument("--port", type=int, default=3333)
    ap.add_argument("--model", default="model.pkl")
    ap.add_argument("--fs", type=float, default=10.0)
    ap.add_argument("--win", type=float, default=2.0)
    ap.add_argument("--step", type=float, default=1.0)
    args = ap.parse_args()

    obj = joblib.load(args.model)
    clf = obj["model"]; feature_list = obj["features"]

    # Rolling buffer (a few extra seconds so resampling works)
    max_buf_sec = max(10.0, args.win*3)
    rows = []
    last_pred_time = time.time()
    preds_log = Path("live_preds.csv")
    if not preds_log.exists():
        pd.DataFrame(columns=["timestamp","pred","proba"]).to_csv(preds_log, index=False)

    print(f"Connecting to {args.host}:{args.port} …")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.connect((args.host,args.port))
    s.settimeout(None)
    print("Connected. Waiting for data… (Ctrl+C to stop)")

    buffer = ""; header_seen=False

    try:
        while True:
            data = s.recv(1024)
            if not data: break
            buffer += data.decode("utf-8","ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n",1)
                line = line.strip()
                if not line: continue
                parts = [p.strip() for p in line.split(",")]
                # header
                if not header_seen:
                    if parts == HEADERS:
                        header_seen=True; continue
                    else:
                        header_seen=True  # join mid-stream
                if len(parts)!=len(HEADERS): continue

                row = dict(zip(HEADERS, parts))
                rows.append(row)

                # Keep only last max_buf_sec seconds (approximate by count)
                # At 10 Hz, 10 * max_buf_sec rows is enough
                max_rows = int(args.fs * max_buf_sec)
                if len(rows) > max_rows:
                    rows = rows[-max_rows:]

                # Predict every step seconds
                if time.time()-last_pred_time >= args.step:
                    df = pd.DataFrame(rows)
                    proc = preprocess_resample(df, fs=args.fs)
                    if proc is None or len(proc) < int(args.win*args.fs):
                        last_pred_time = time.time()
                        continue
                    W = int(args.win*args.fs)
                    win = proc.iloc[-W:]
                    feats = extract_window_feats(win, fs=args.fs)
                    # Align features with training feature order and fill missing with 0
                    x = pd.DataFrame([feats])[feature_list].fillna(0.0)
                    proba = clf.predict_proba(x)[0]
                    pred = clf.classes_[int(np.argmax(proba))]
                    conf = float(np.max(proba))
                    ts = dt.datetime.now().strftime("%H:%M:%S")
                    print(f"[{ts}] Pred: {pred}  (p={conf:.2f})")
                    pd.DataFrame([{"timestamp":ts,"pred":pred,"proba":conf}]).to_csv(
                        preds_log, mode="a", header=False, index=False
                    )
                    last_pred_time = time.time()

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        try: s.close()
        except: pass

if __name__ == "__main__":
    main()
