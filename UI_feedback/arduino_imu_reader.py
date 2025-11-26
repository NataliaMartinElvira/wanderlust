import time
import serial
import numpy as np

# --- CONFIG ---
# This is the expected format of the V1_ACCEL packet (9 float values total)
LEN_V1_PACKET = 9 
# This is the expected format of the V2_ACCEL packet (18 float values total, 9 per IMU)
LEN_V2_PACKET = 18

# Sensor Keys for V1 (Single IMU: Leg)
LEG_ID = 'sensor2' 

# Sensor Keys for V2 (Dual IMU: Torso/Pelvis)
UPPER_ID = 'IMU_CH3' 
PELVIS_ID = 'IMU_CH0'

class ArduinoIMUReader:
    """
    Class to manage the serial connection and parse data from IMUs based on packet ID.
    Handles V1 (single IMU, 9 values) and V2 (dual IMU, 18 values).
    """
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.is_connected = False

    def connect_imus(self):
        """Attempts to establish the real serial connection."""
        print(f"[IMU] Conectando al Arduino Receptor en {self.port}...")
        try:
            # Set a timeout to prevent blocking the main thread indefinitely
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=0.5) 
            time.sleep(2) 
            self.serial_connection.flushInput() # Clear old data from buffer
            
            self.is_connected = True
            print("[IMU] Conexión serial exitosa. Preparado para leer V1/V2...")
        except Exception as e:
            print(f"[IMU ERROR] No se pudo conectar al Arduino Receptor en {self.port}: {e}")
            self.is_connected = False

    def _parse_imu_data(self, float_values, start_index):
        """Helper to extract 9 values from the float list and map them to keys."""
        # The indices are based on the full HEADERS list provided by the user
        # 0: time_ms, 1: acc_x, 2: acc_y, 3: acc_z, 4: pitch, 5: roll, 6: gyr_x, 7: gyr_y, 8: gyr_z
        
        return {
            'accel_x': float_values[start_index + 1], 
            'accel_y': float_values[start_index + 2], 
            'accel_z': float_values[start_index + 3], 
            'pitch_deg': float_values[start_index + 4],
            'roll_deg': float_values[start_index + 5],
            'gyr_x': float_values[start_index + 6],
            'gyr_y': float_values[start_index + 7],
            'gyr_z': float_values[start_index + 8],
        }


    def read_data(self):
        """
        Reads one line of IMU data and parses it based on the packet ID (V1 or V2).
        Returns a dict with sensor keys (e.g., {'sensor2': {...}}) or None.
        """
        if not self.is_connected:
            return None

        try:
            line_bytes = self.serial_connection.readline() 
            if not line_bytes: return None 
                
            line = line_bytes.decode('utf-8', errors='replace').strip()
            
            if not line: return None
            
            # --- Identify Packet Type ---
            
            packet_type = None
            if line.startswith("V1_ACCEL:"):
                clean_line = line.replace("V1_ACCEL:", "")
                packet_type = 1
            elif line.startswith("V2_ACCEL:"):
                clean_line = line.replace("V2_ACCEL:", "")
                packet_type = 2
            else:
                return None # Skip unhandled lines
            
            parts = clean_line.split(',')
            
            # Validate Length
            expected_len = LEN_V1_PACKET if packet_type == 1 else LEN_V2_PACKET
            if len(parts) != expected_len:
                # print(f"[IMU PARSE ERROR] Length mismatch V{packet_type}. Expected {expected_len}, got {len(parts)}. Line: {line}")
                return None

            float_values = [float(p.strip()) for p in parts]
            
            # --- Structure Data based on Type ---
            
            if packet_type == 1:
                # V1: Single IMU (Pierna)
                return { LEG_ID: self._parse_imu_data(float_values, 0) }
                
            elif packet_type == 2:
                # V2: Dual IMU (Torso/Pelvis)
                # Assuming the first 9 values are UPPER, and the second 9 are PELVIS
                return {
                    UPPER_ID: self._parse_imu_data(float_values, 0),    # Indices 0 to 8
                    PELVIS_ID: self._parse_imu_data(float_values, 9)   # Indices 9 to 17
                }
            
        except ValueError:
            # print(f"[IMU PARSE ERROR] Failed to convert data to float. Line: {line}")
            return None
        except Exception:
            return None

    def disconnect_imus(self):
        """Closes the serial connection."""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("[IMU] Conexión serial cerrada.")
        self.is_connected = False