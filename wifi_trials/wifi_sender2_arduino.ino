#include <esp_now.h>
#include <WiFi.h>
#include <Wire.h>
#include <ICM20948_WE.h>

// ------------------------------------------------------------
// HARDWARE SETTINGS (HUZZAH32 + PCA9546A)
// ------------------------------------------------------------
#define NODE_ID 2
#define SDA_PIN 23
#define SCL_PIN 22

// PCA9546A Multiplexer Address
#define PCA9546A_ADDR 0x70 

// IMU Address (Both are 0x69, separated by Mux)
#define ICM20948_ADDR 0x69

// Create Objects
ICM20948_WE imu_ch0(ICM20948_ADDR); // Connected to Mux Channel 0
ICM20948_WE imu_ch3(ICM20948_ADDR); // Connected to Mux Channel 3

// Receiver MAC address (MUST MATCH YOUR RECEIVER)
uint8_t broadcastAddress[] = {0x10, 0x06, 0x1C, 0x18, 0x69, 0x00};

// ------------------------------------------------------------
// V2 DATA STRUCTURE (Large Packet for 2 IMUs)
// ------------------------------------------------------------
typedef struct struct_message_v2 {
  int id;
  unsigned long timestamp;
  // IMU 1 (from CH0)
  float acc1_x, acc1_y, acc1_z;
  float pitch1, roll1;
  float gyr1_x, gyr1_y, gyr1_z;
  // IMU 2 (from CH3)
  float acc2_x, acc2_y, acc2_z;
  float pitch2, roll2;
  float gyr2_x, gyr2_y, gyr2_z;
} struct_message_v2;

struct_message_v2 myData;

// ------------------------------------------------------------
// MULTIPLEXER HELPER FUNCTIONS
// ------------------------------------------------------------
static inline bool pcaSelect(uint8_t channel) {
  if (channel > 3) return false;
  Wire.beginTransmission(PCA9546A_ADDR);
  Wire.write(1 << channel);
  if (Wire.endTransmission() != 0) return false;
  delay(2); 
  return true;
}

static inline void pcaDisableAll() {
  Wire.beginTransmission(PCA9546A_ADDR);
  Wire.write(0x00);
  Wire.endTransmission();
}

// ------------------------------------------------------------
// ESP-NOW CALLBACK
// ------------------------------------------------------------
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  // Optional debug
  // if (status != ESP_NOW_SEND_SUCCESS) Serial.println("Send Fail");
}

// ------------------------------------------------------------
// SETUP
// ------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  delay(500);

  // 1. Init I2C
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000); // Fast Mode

  // 2. Init ESP-NOW
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("ESP-NOW Init Failed");
    return;
  }
  esp_now_register_send_cb(OnDataSent);

  // 3. Register Peer
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }

  // 4. Init IMU on Channel 0
  pcaSelect(0);
  if (!imu_ch0.init()) {
    Serial.println("IMU CH0 Error!");
  } else {
    Serial.println("IMU CH0 OK");
  }

  // 5. Init IMU on Channel 3
  pcaSelect(3);
  if (!imu_ch3.init()) {
    Serial.println("IMU CH3 Error!");
  } else {
    Serial.println("IMU CH3 OK");
  }
  
  pcaDisableAll();
}

// ------------------------------------------------------------
// MAIN LOOP
// ------------------------------------------------------------
void loop() {
  unsigned long t = millis();

  // --- READ SENSOR 1 (CH0) ---
  pcaSelect(0);
  imu_ch0.readSensor();
  xyzFloat acc1; imu_ch0.getGValues(&acc1);
  xyzFloat gyr1; imu_ch0.getGyrValues(&gyr1);
  float pitch1 = imu_ch0.getPitch();
  float roll1  = imu_ch0.getRoll();

  // --- READ SENSOR 2 (CH3) ---
  pcaSelect(3);
  imu_ch3.readSensor();
  xyzFloat acc2; imu_ch3.getGValues(&acc2);
  xyzFloat gyr2; imu_ch3.getGyrValues(&gyr2);
  float pitch2 = imu_ch3.getPitch();
  float roll2  = imu_ch3.getRoll();

  pcaDisableAll(); // Good practice to release bus

  // --- PACK DATA ---
  myData.id = NODE_ID;
  myData.timestamp = t;

  // Fill IMU 1
  myData.acc1_x = acc1.x;
  myData.acc1_y = acc1.y;
  myData.acc1_z = acc1.z;
  myData.pitch1 = pitch1;
  myData.roll1  = roll1;
  myData.gyr1_x = gyr1.x;
  myData.gyr1_y = gyr1.y;
  myData.gyr1_z = gyr1.z;

  // Fill IMU 2
  myData.acc2_x = acc2.x;
  myData.acc2_y = acc2.y;
  myData.acc2_z = acc2.z;
  myData.pitch2 = pitch2;
  myData.roll2  = roll2;
  myData.gyr2_x = gyr2.x;
  myData.gyr2_y = gyr2.y;
  myData.gyr2_z = gyr2.z;

  // --- SEND ---
  // The receiver will detect the larger packet size and treat it as V2
  esp_now_send(broadcastAddress, (uint8_t *) &myData, sizeof(myData));

  // Optional: Print for local debug
  
  Serial.print("Sent V2 Packet. Pitch1: ");
  Serial.print(pitch1);
  Serial.print(" Pitch2: ");
  Serial.println(pitch2);
  

  delay(50); // 20Hz
}
