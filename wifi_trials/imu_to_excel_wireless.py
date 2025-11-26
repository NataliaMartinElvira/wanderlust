# tcp_imu_to_excel_wireless.py
# pip install pandas openpyxl

# After upload, power the ESP32 from a battery/USB power bank.
# On your computer, join Wi-Fi IMU_Logger.
# Run: python tcp_imu_to_excel_wireless.py
# Press Ctrl+C to stop; file saves in the script folder.

import socket
import time
import datetime as dt
from pathlib import Path
import pandas as pd

HOST = "192.168.4.1"   # fixed AP IP
PORT = 3333
SHEET_NAME = "IMU"
HEADERS = [
    "time_ms","acc_x_g","acc_y_g","acc_z_g",
    "pitch_deg","roll_deg","gyr_x_dps","gyr_y_dps","gyr_z_dps"
]

SAVE_EVERY_SECONDS = 2
SAVE_EVERY_ROWS = 200
RECONNECT_DELAY_S = 2

def now_ts():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def save_buffer(out_path, sheet_name, headers, buffered_rows):
    if not buffered_rows:
        return 0

    new_df = pd.DataFrame(buffered_rows, columns=headers)

    # Explicit, future-proof numeric casting (no errors="ignore")
    numeric_cols = [
        "time_ms","acc_x_g","acc_y_g","acc_z_g",
        "pitch_deg","roll_deg","gyr_x_dps","gyr_y_dps","gyr_z_dps"
    ]
    for col in numeric_cols:
        try:
            # time_ms is integer-like; others float-like
            if col == "time_ms":
                new_df[col] = pd.to_numeric(new_df[col], downcast="integer")
            else:
                new_df[col] = pd.to_numeric(new_df[col])
        except Exception:
            # If a bad value slips in, keep the original text so the row isn't lost
            pass

    # Load existing (if present), append, rewrite
    if out_path.exists():
        try:
            existing = pd.read_excel(out_path, sheet_name=sheet_name, engine="openpyxl")
        except Exception:
            existing = pd.DataFrame(columns=headers)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    with pd.ExcelWriter(out_path, engine="openpyxl", mode="w") as writer:
        combined.to_excel(writer, index=False, sheet_name=sheet_name)

    return len(new_df)

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
            print(f"Connect failed: {e}. Retrying in {RECONNECT_DELAY_S}s…")
            time.sleep(RECONNECT_DELAY_S)

def main():
    script_dir = Path(__file__).parent
    out_path = script_dir / f"imu_data_{now_ts()}.xlsx"
    print(f"Saving to: {out_path}")

    total_saved = 0
    rows = []
    buffer = ""
    last_save = time.time()
    header_seen = False

    s = connect()

    try:
        while True:
            try:
                data = s.recv(1024)
                if not data:
                    raise ConnectionError("Remote closed")
                buffer += data.decode("utf-8", errors="ignore")

                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue

                    parts = [p.strip() for p in line.split(",")]

                    # First valid line should be header from ESP
                    if not header_seen:
                        if parts == HEADERS:
                            header_seen = True
                            print("Header received.")
                            continue
                        else:
                            # Might have joined mid-stream; treat as data if lengths match
                            header_seen = True

                    if len(parts) != len(HEADERS):
                        # skip noise
                        continue

                    rows.append(parts)

                    if len(rows) % 50 == 0:
                        print(f"Buffered: {len(rows)} | Total saved: {total_saved}")

                    if (time.time() - last_save >= SAVE_EVERY_SECONDS) or (len(rows) >= SAVE_EVERY_ROWS):
                        total_saved += save_buffer(out_path, SHEET_NAME, HEADERS, rows)
                        rows.clear()
                        last_save = time.time()

            except (ConnectionError, OSError):
                print("Lost connection. Flushing buffer and reconnecting…")
                if rows:
                    total_saved += save_buffer(out_path, SHEET_NAME, HEADERS, rows)
                    rows.clear()
                s.close()
                time.sleep(RECONNECT_DELAY_S)
                s = connect()
                header_seen = False
                buffer = ""

    except KeyboardInterrupt:
        print("\nStopping…")

    finally:
        try:
            s.close()
        except Exception:
            pass
        if rows:
            total_saved += save_buffer(out_path, SHEET_NAME, HEADERS, rows)
        print(f"Done. Total rows saved: {total_saved}")
        print(f"Excel file: {out_path}")

if __name__ == "__main__":
    main()
