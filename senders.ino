#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h> // Include Wire.h for I2C communication (for most accelerometers)
#include <ESP.h> // <-- ¡Asegúrate de que esto esté arriba!

// --- ACCELEROMETER LIBRARIES (Adjust these based on your specific sensor, e.g., MPU6050) ---
// You will likely need to install libraries for your sensor via the Arduino Library Manager.
// Example: #include <Adafruit_MPU6050.h>
// Example: Adafruit_MPU6050 mpu;
// ------------------------------------------------------------------------------------------

// --- Configuration (MUST MATCH YOUR HOTSPOT AND RECEIVER) ---
const char* ssid = "iPhone de Laura";          // Your hotspot name
const char* password = "ponceorozcolaura";      // Your hotspot password
// !!! Receiver IP Address from the Huzzah32 V2 Serial Monitor: 172.20.10.2 !!!
const char* receiverIP = "172.20.10.2";
unsigned int remoteUdpPort = 8888;             // Must match the Receiver's port

// --- Unique Identifier (CHANGE THIS FOR EACH BOARD) ---
// Board V1:
// const char* boardIdentifier = "V1_ACCEL";
// Board V2:
const char* boardIdentifier = "V2_ACCEL"; // <-- Change this line for each board

// --- Global Variables ---
WiFiUDP Udp;

// ------------------------------------------------------------------------------------------
// !!! IMPORTANT: Replace this function with your actual accelerometer reading code !!!
// ------------------------------------------------------------------------------------------
String readAccelerometerData() {
  // 1. **(Initialise Sensor)** If your sensor needs it (e.g., mpu.getEvent()), put that here.
  
  // 2. **(Read Data)** Replace these lines with your sensor's actual X, Y, and Z readings.
  // The goal is to get the three float or integer values.
  
  // --- PLACEHOLDER CODE (Replace me!) ---
  // Using random numbers for demonstration. Replace with actual sensor data.
  float x = random(100, 200) / 100.0; 
  float y = random(10, 50) / 100.0;
  float z = random(-1000, -900) / 100.0;
  // --- Sensor B Readings ---
  // Replace these lines with actual readings from ACCEL B
  //float xB = random(200, 300) / 100.0; 
  //float yB = random(60, 100) / 100.0;
  //float zB = random(-1100, -1000) / 100.0;
  // --- END PLACEHOLDER CODE ---

  // 3. **(Format String)** Concatenate the data into a comma-separated string.
  // This format (e.g., "1.52,0.34,-9.81") is easy for the Receiver to parse.
  return String(x) + "," + String(y) + "," + String(z);

  //return String(xA) + "," + String(yA) + "," + String(zA) + "," + String(xB) + "," + String(yB) + "," + String(zB);
}
// ------------------------------------------------------------------------------------------

void setup() {
  Serial.begin(115200);
  //Wire.begin(); // Initialize I2C communication

  // --- Initialize your accelerometer sensor here (e.g., mpu.begin()) ---
  // Reemplaza esta sección con el código real de inicialización del sensor.
  // ---------------------------------------------------------------------

  // Connect to Wi-Fi
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  // --- Bucle de Conexión Robusto con Prevención de WDT ---
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(500); 
    Serial.print(".");
    yield(); // Permite que el Watchdog Timer se ejecute
    
    attempts++;
    if (attempts > 60) { // Timeout de 30 segundos (60 * 500ms)
        Serial.println("\nConnection failed after 30s. Restarting board...");
        ESP.restart(); // Reinicia la placa para intentar la conexión de nuevo
    }
  }
  // --- Fin del Bucle de Conexión ---

  Serial.println("\nWiFi connected.");
  Serial.print("Sender IP address: ");
  Serial.println(WiFi.localIP());
}

void loop() {
  // 1. Get the sensor data
  String accelData = readAccelerometerData();

  // 2. Construct the full message: Identifier:X,Y,Z
  // Example: V1_ACCEL:1.52,0.34,-9.81
  String fullMessage = String(boardIdentifier) + ":" + accelData;

  // 3. Send the UDP packet
  Udp.beginPacket(receiverIP, remoteUdpPort);
  Udp.print(fullMessage);
  Udp.endPacket();

  Serial.print("Sent message to ");
  Serial.print(receiverIP);
  Serial.print(": ");
  Serial.println(fullMessage);

  delay(50); // Adjust this delay to control the data stream rate (50ms = 20 samples/sec)
}
