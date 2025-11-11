import serial
import time

# --- Serial Port Configuration ---
# Ensure the port name and speed match the Arduino settings
SERIAL_PORT = 'COM6' 
BAUD_RATE = 115200

print(f"Attempting to connect to {SERIAL_PORT} at {BAUD_RATE} baud...")

try:
    # Open the serial connection
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) # Wait for the connection to establish

    print("Connection successful. Receiving data...")
    print("-" * 30)

    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            
            if line and "Data: " in line:
                data_part = line.split("Data: ")[-1]
                print(f"Data Received: {data_part}")
                
                if ":" in data_part:
                    identifier, raw_values = data_part.split(":")
                    
                    # --- NEW LOGIC FOR HANDLING V1 and V2 ---
                    
                    if identifier == "V1_ACCEL":
                        # V1 sends 6 values (A:X,Y,Z, B:X,Y,Z)
                        all_values = list(map(float, raw_values.split(',')))
                        if len(all_values) == 6:
                            xA, yA, zA, xB, yB, zB = all_values
                            print(f"  > ID: {identifier} (2 Sensors)")
                            print(f"    Sensor A: X={xA:.2f}, Y={yA:.2f}, Z={zA:.2f}")
                            print(f"    Sensor B: X={xB:.2f}, Y={yB:.2f}, Z={zB:.2f}")
                        else:
                            print(f"  > ERROR: V1 received {len(all_values)} values, expected 6.")

                    elif identifier == "V2_ACCEL":
                        # V2 sends 3 values (X,Y,Z)
                        try:
                            x, y, z = map(float, raw_values.split(','))
                            print(f"  > ID: {identifier} (1 Sensor): X={x:.2f}, Y={y:.2f}, Z={z:.2f}")
                        except ValueError:
                            print(f"  > ERROR: V2 data format incorrect.")

except serial.SerialException as e:
    print(f"Serial connection error: {e}")
    print("Make sure the COM6 port is not open in the Arduino IDE or any other program.")
except KeyboardInterrupt:
    print("\nStopping the script.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")