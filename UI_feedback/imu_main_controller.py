import socket
import threading
import time
from queue import Queue
import importlib
import sys

# --- COMMUNICATION CONFIGURATION WITH UNITY ---
UNITY_IP = '127.0.0.1'
UNITY_FEEDBACK_PORT = 5001     # Python SENDS FEEDBACK here (Client)
PYTHON_COMMAND_PORT = 5004     # Python RECEIVES COMMANDS here (Server)

# --- GLOBAL STATE ---
GLOBAL_STATE = {
    'current_exercise': 'NONE', 
    'stop_threads': False
}

# Queue for safely sending commands from main loop to Unity Sender Thread
tcp_send_queue = Queue() 

# Map exercise names (received from Unity) to their corresponding logic classes
try:
    from seated_march_logic import SeatedMarchLogic
    from trunk_rotation_logic import TrunkRotationLogic 
    from exercise_logic_template import ExerciseLogicTemplate 
except ImportError as e:
    print(f"Error: Could not import required logic modules. Details: {e}")
    sys.exit(1)

LOGIC_MAP = {
    'SEATED_MARCH': SeatedMarchLogic,
    'TRUNK_ROTATION': TrunkRotationLogic, 
    'STANDING_MARCH': SeatedMarchLogic, 
    'CALIBRATION': ExerciseLogicTemplate, 
    'NONE': ExerciseLogicTemplate 
}

# --- TCP CLIENT (SENDER) ---

class TcpClientThread(threading.Thread):
    """ Handles connection and sending of feedback to Unity (Python is Client). """
    def __init__(self, ip, port, state):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.state = state
        self.socket = None
        self.is_connected = False

    def run(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.ip, self.port))
            self.is_connected = True
            print(f"[TCP CLIENT] Connection to Unity Feedback Server ({self.port}) established.")
        except ConnectionRefusedError:
            print(f"[TCP CLIENT ERROR] Connection refused on {self.port}. Ensure Unity is in Play Mode.")
            self.state['stop_threads'] = True
            return

        while not self.state['stop_threads']:
            try:
                if not tcp_send_queue.empty():
                    message = tcp_send_queue.get()
                    if message:
                        self.socket.sendall(message.encode('utf-8'))
                
                time.sleep(0.01)

            except socket.error as e:
                if e.errno == 32: 
                    print("[TCP CLIENT ERROR] Socket error: Broken pipe. Disconnected.")
                else:
                    print(f"[TCP CLIENT ERROR] Socket error: {e}. Disconnected.")
                self.is_connected = False
                self.state['stop_threads'] = True
                break
        
        if self.socket: self.socket.close()
        print("[TCP CLIENT] Client thread closed.")

# --- TCP SERVER (RECEIVER) ---

class TcpListenerThread(threading.Thread):
    """ Handles listening for state commands sent FROM Unity (Python is Server). """
    def __init__(self, ip, port, state):
        threading.Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.state = state
        self.listener = None
        
    def run(self):
        try:
            self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.listener.bind((self.ip, self.port))
            self.listener.listen(1)
            print(f"[TCP SERVER] Listening for Unity commands on {self.ip}:{self.port}...")

            while not self.state['stop_threads']:
                try:
                    self.listener.settimeout(0.5) 
                    conn, addr = self.listener.accept()
                    
                    with conn:
                        data = conn.recv(1024).decode('utf-8')
                        if data:
                            self.process_command(data.strip())
                except socket.timeout:
                    continue
                except Exception as e:
                    if not self.state['stop_threads']:
                        print(f"[TCP SERVER ERROR] Error during connection: {e}")
                    break
                    
        except Exception as e:
            print(f"[TCP SERVER ERROR] Listener setup failed: {e}")
            self.state['stop_threads'] = True
        finally:
            if self.listener: self.listener.close()
            print("[TCP SERVER] Listener thread closed.")

    def process_command(self, command):
        """ Parses commands received from Unity. """
        parts = command.split(':')
        cmd_type = parts[0].strip().upper()
        
        if cmd_type == 'SET_EXERCISE' and len(parts) > 1:
            exercise = parts[1].strip().upper()
            if exercise in LOGIC_MAP:
                GLOBAL_STATE['current_exercise'] = exercise
                tcp_send_queue.put("RESET_STEP_COUNTER\n") 
                print(f"[TCP SERVER] State change received: SET_EXERCISE to {exercise}")
            else:
                print(f"[TCP SERVER WARNING] Unknown exercise logic requested: {exercise}")
        
        elif cmd_type == 'START_CALIBRATION':
            # NOTE: We keep this command handler in case Unity still sends it, 
            # but the main loop handles the state switch to CALIBRATION which now only runs the template.
            GLOBAL_STATE['current_exercise'] = 'CALIBRATION'
            tcp_send_queue.put("RESET_STEP_COUNTER\n") 
            print("[TCP SERVER] Command received: START_CALIBRATION (Resetting buffer)")
            
        else:
            print(f"[TCP SERVER WARNING] Unhandled command received: {command}")


def send_to_unity(command):
    """ Helper function to safely put commands into the sending queue. """
    if isinstance(command, str) and command.strip():
        tcp_send_queue.put(command.strip() + '\n')
    else:
        print("[ERROR] Attempted to send empty command.")

def main_loop():
    """ 
    Main program loop that reads IMUs and applies exercise logic based on Unity state.
    """
    try:
        from arduino_imu_reader import ArduinoIMUReader 
    except ImportError:
        print("[ERROR] arduino_imu_reader not found. Cannot proceed without IMU data source.")
        return

    # 1. INITIALIZE ARDUINO/IMU
    imu_reader = ArduinoIMUReader(port='/dev/tty.usbserial-58550220231', baudrate=115200) 
    imu_reader.connect_imus()
    
    # 2. START TCP THREADS 
    tcp_send_thread = TcpClientThread(UNITY_IP, UNITY_FEEDBACK_PORT, GLOBAL_STATE)
    tcp_listen_thread = TcpListenerThread(UNITY_IP, PYTHON_COMMAND_PORT, GLOBAL_STATE)
    
    tcp_send_thread.start()
    tcp_listen_thread.start()

    while not tcp_send_thread.is_connected and not GLOBAL_STATE['stop_threads']:
        time.sleep(0.5)

    if not tcp_send_thread.is_connected:
        print("Application could not connect to Unity. Exiting.")
        return

    # 3. LOGIC INSTANCE MANAGEMENT
    current_logic_key = 'NONE'
    
    sys.modules['imu_main_controller'] = sys.modules[__name__]

    current_logic = LOGIC_MAP[current_logic_key]() 
    
    print("\n--- MAIN LOOP STARTED ---")

    # --- CONTROL AND EXECUTION LOOP ---
    while not GLOBAL_STATE['stop_threads']:
        
        # Check if the required logic module has changed
        if GLOBAL_STATE['current_exercise'] != current_logic_key:
            current_logic_key = GLOBAL_STATE['current_exercise']
            current_logic = LOGIC_MAP.get(current_logic_key, LOGIC_MAP['NONE'])()
            print(f"[MAIN] Switched active logic to: {current_logic_key}")


        # 3.1. READ RAW IMU DATA
        raw_data = imu_reader.read_data() 
        if raw_data is None:
            time.sleep(0.005)
            continue
        
        # 3.2. CRITICAL: IDLE STATE CHECK
        
        ACTIVE_EXERCISES = ['SEATED_MARCH', 'STANDING_MARCH', 'TRUNK_ROTATION']
        
        if GLOBAL_STATE['current_exercise'] == 'NONE':
            time.sleep(0.005)
            continue

        # 3.3. ACTIVE EXERCISE PHASE
        
        # A) Explicit CALIBRATION handling (REMOVED BRANCH FOR CALIBRATION)
        # NOTE: If Unity sends START_CALIBRATION, the state changes to 'CALIBRATION'
        # and the code below will call current_logic.check_calmness.
        
        if GLOBAL_STATE['current_exercise'] == 'CALIBRATION':
            current_logic.check_calmness(raw_data)
            time.sleep(0.005)
            continue
        
        # B) PERFORMANCE ANALYSIS (Only for exercise states)
        if GLOBAL_STATE['current_exercise'] in ACTIVE_EXERCISES:
            
            feedback_result = current_logic.analyze_performance(raw_data)
            
            if feedback_result is not None:
                
                feedback_is_bad = feedback_result
                
                if current_logic_key == 'TRUNK_ROTATION':
                    # TRUNK_ROTATION logic handles its own sending via _send_feedback.
                    pass 
                
                else: # SEATED/STANDING MARCH
                    if feedback_is_bad:
                        send_to_unity("FEEDBACK:BAD") 
                    else:
                        send_to_unity("FEEDBACK:GOOD") 
        
        
        time.sleep(0.005) 
        
    # --- CLEANUP ---
    print("\nShutting down the system.")
    GLOBAL_STATE['stop_threads'] = True
    imu_reader.disconnect_imus()
    tcp_send_thread.join() 
    tcp_listen_thread.join()

if __name__ == "__main__":
    main_loop()