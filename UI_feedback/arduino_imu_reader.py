import time
import serial
import numpy as np

class ArduinoIMUReader:
    """
    Class to manage the serial connection with the Arduino and read data from the IMUs.
    This version expects and parses the V1_ACCEL packet format (9 values).
    """
    def __init__(self, port, baudrate):
        self.port = port
        self.baudrate = baudrate
        self.serial_connection = None
        self.is_connected = False
        self.EXPECTED_DATA_LEN = 9 
        self.TARGET_SENSOR_KEY = 'sensor2' 

    def connect_imus(self):
        """Attempts to establish the real serial connection."""
        print(f"[IMU] Conectando al Arduino Receptor en {self.port}...")
        try:
            self.serial_connection = serial.Serial(self.port, self.baudrate, timeout=0.5) 
            time.sleep(2) 
            self.serial_connection.flushInput() 
            
            self.is_connected = True
            print("[IMU] Conexión serial exitosa. Esperando paquetes V1_ACCEL...")
        except Exception as e:
            print(f"[IMU ERROR] No se pudo conectar al Arduino Receptor en {self.port}: {e}")
            self.is_connected = False

    def read_data(self):
        """
        Reads one line of IMU data, parses the V1_ACCEL packet, and returns a dictionary.
        Prints only on errors.
        """
        if not self.is_connected:
            print("[IMU - read_data ERROR] Intento de lecimtura sin conexión establecida.")
            return None

        try:
            line_bytes = self.serial_connection.readline()
            #print("Line_bytes after serial connection:", line_bytes)
            if not line_bytes:
                return None 
            #print("Line bytes received:", line_bytes)
                
            line = line_bytes.decode('utf-8', errors='replace').strip()
            
            if not line.startswith("V1_ACCEL:"):
                return None
            
            clean_line = line.replace("V1_ACCEL:", "")
            parts = clean_line.split(',')
            
            if len(parts) != self.EXPECTED_DATA_LEN:
                # Print only on error
                print(f"[IMU PARSE ERROR] Length mismatch. Line: {line}")
                return None

            float_values = [float(p.strip()) for p in parts]
            
            raw_data_dict = {
                self.TARGET_SENSOR_KEY: {
                    'accel_x': float_values[1], 
                    'accel_y': float_values[2], 
                    'accel_z': float_values[3], 
                    'angle_x': float_values[4]
                }
            }
            
            return raw_data_dict
            
        except ValueError:
            print(f"[IMU PARSE ERROR] Failed to convert data to float. Line: {line}")
            return None
        except Exception:
            return None

    def disconnect_imus(self):
        """Closes the serial connection."""
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
            print("[IMU] Conexión serial cerrada.")
        self.is_connected = False