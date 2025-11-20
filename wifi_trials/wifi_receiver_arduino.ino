#include <esp_now.h>
#include <WiFi.h>
//receiver
// --- Define BOTH Structures ---
typedef struct struct_v1 {
  int id; 
  unsigned long timestamp;
  float acc_x, acc_y, acc_z;
  float pitch, roll;
  float gyr_x, gyr_y, gyr_z;
} struct_v1;

typedef struct struct_v2 {
  int id;
  unsigned long timestamp;
  float acc1_x, acc1_y, acc1_z;
  float pitch1, roll1;
  float gyr1_x, gyr1_y, gyr1_z;
  float acc2_x, acc2_y, acc2_z;
  float pitch2, roll2;
  float gyr2_x, gyr2_y, gyr2_z;
} struct_v2;

struct_v1 dataV1;
struct_v2 dataV2;

// --- Callback ---
void OnDataRecv(const uint8_t * mac, const uint8_t *incomingData, int len) {
  
  // CHECK IF IT IS V1 (Small packet)
  if (len == sizeof(struct_v1)) {
    memcpy(&dataV1, incomingData, sizeof(dataV1));
    
    Serial.print("V1_ACCEL:"); // <--- IDENTIFIER 1
    Serial.print(dataV1.timestamp); Serial.print(",");
    Serial.print(dataV1.acc_x, 3); Serial.print(",");
    Serial.print(dataV1.acc_y, 3); Serial.print(",");
    Serial.print(dataV1.acc_z, 3); Serial.print(",");
    Serial.print(dataV1.pitch, 2); Serial.print(",");
    Serial.print(dataV1.roll, 2); Serial.print(",");
    Serial.print(dataV1.gyr_x, 3); Serial.print(",");
    Serial.print(dataV1.gyr_y, 3); Serial.print(",");
    Serial.println(dataV1.gyr_z, 3);
  }

  // CHECK IF IT IS V2 (Large packet)
  else if (len == sizeof(struct_v2)) {
    memcpy(&dataV2, incomingData, sizeof(dataV2));
    
    Serial.print("V2_ACCEL:"); // <--- IDENTIFIER 2
    Serial.print(dataV2.timestamp); Serial.print(",");
    // Print IMU 1
    Serial.print(dataV2.acc1_x, 3); Serial.print(",");
    Serial.print(dataV2.acc1_y, 3); Serial.print(",");
    Serial.print(dataV2.acc1_z, 3); Serial.print(",");
    Serial.print(dataV2.pitch1, 2); Serial.print(",");
    Serial.print(dataV2.roll1, 2); Serial.print(",");
    Serial.print(dataV2.gyr1_x, 3); Serial.print(",");
    Serial.print(dataV2.gyr1_y, 3); Serial.print(",");
    Serial.print(dataV2.gyr1_z, 3); Serial.print(",");
    // Print IMU 2
    Serial.print(dataV2.acc2_x, 3); Serial.print(",");
    Serial.print(dataV2.acc2_y, 3); Serial.print(",");
    Serial.print(dataV2.acc2_z, 3); Serial.print(",");
    Serial.print(dataV2.pitch2, 2); Serial.print(",");
    Serial.print(dataV2.roll2, 2); Serial.print(",");
    Serial.print(dataV2.gyr2_x, 3); Serial.print(",");
    Serial.print(dataV2.gyr2_y, 3); Serial.print(",");
    Serial.println(dataV2.gyr2_z, 3);
  }
}

void setup() {
  Serial.begin(115200);
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) return;
  esp_now_register_recv_cb(OnDataRecv);
}

void loop() {}
