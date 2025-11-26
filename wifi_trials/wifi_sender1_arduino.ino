#include <esp_now.h>
#include <WiFi.h>
#include <ICM20948_WE.h>
#include <Wire.h> 
#include "freertos/FreeRTOS.h"
#include "freertos/task.h" 
//sender1

// CUSTOM SETTINGS

#define NODE_ID 1  // This is Board 1

// Receiver MAC address (MUST MATCH YOUR RECEIVER BOARD)
uint8_t broadcastAddress[] = {0x10, 0x06, 0x1C, 0x18, 0x69, 0x00};

// I2C Pins (ESP32-PICO)
#define I2C_SDA_PIN 22
#define I2C_SCL_PIN 20


// V1 DATA STRUCTURE (Small Packet for 1 IMU)

typedef struct struct_message_v1 {
    int id; 
    unsigned long timestamp;
    float acc_x, acc_y, acc_z;
    float pitch, roll;
    float gyr_x, gyr_y, gyr_z;
} struct_message_v1;

struct_message_v1 myData;

// IMU Object
#define ICM20948_ADDR 0x68
ICM20948_WE myIMU = ICM20948_WE(ICM20948_ADDR);


// ESP-NOW SEND CALLBACK

void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
    
    if (status != ESP_NOW_SEND_SUCCESS) {
        // Serial.println("ESP-NOW send error"); 
    }
}


// IMU TASK (Runs on Core 1)

void imu_task(void *pvParameters) {

    Serial.println("[CORE 1] Initializing I2C...");
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, 400000);

    delay(200);

    Serial.println("[CORE 1] Initializing ICM20948...");
    if (!myIMU.init()) {
        Serial.println("[CORE 1] FATAL: IMU not detected!");
        vTaskDelete(NULL);
    }

    Serial.println("[CORE 1] IMU OK. Streaming...");

    while (1) {
        unsigned long t = millis();

        // 1. Read Sensor
        myIMU.readSensor();

        // 2. Get Data 
        xyzFloat acc;
        myIMU.getGValues(&acc); 

        xyzFloat gyr;
        myIMU.getGyrValues(&gyr); 

        float pitch = myIMU.getPitch();
        float roll = myIMU.getRoll();

        // 3. Fill V1 Structure
        myData.id = NODE_ID;
        myData.timestamp = t;
        
        myData.acc_x = acc.x;
        myData.acc_y = acc.y;
        myData.acc_z = acc.z;
        
        myData.pitch = pitch;
        myData.roll = roll;
        
        myData.gyr_x = gyr.x;
        myData.gyr_y = gyr.y;
        myData.gyr_z = gyr.z;

        // 4. Send Data
        
        esp_now_send(broadcastAddress, (uint8_t*)&myData, sizeof(myData));

        
        static unsigned long lastPrint = 0;
        if (millis() - lastPrint > 200) {
             Serial.print("V1 Sending | Pitch: ");
             Serial.println(pitch);
             lastPrint = millis();
        }

        vTaskDelay(pdMS_TO_TICKS(50)); // 20 Hz
    }
}

// ------------------------------------------------------------
// SETUP
// ------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println("[CORE 0] Starting ESP-NOW Sender 1...");

    WiFi.mode(WIFI_STA);

    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP-NOW init failed!");
        return;
    }

    esp_now_register_send_cb(OnDataSent);

    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, broadcastAddress, 6);
    peerInfo.channel = 0;
    peerInfo.encrypt = false;

    if (esp_now_add_peer(&peerInfo) != ESP_OK) {
        Serial.println("Failed to add peer!");
        return;
    }

    Serial.println("[CORE 0] ESP-NOW OK. Starting IMU task...");

    xTaskCreatePinnedToCore(
        imu_task,
        "IMU_Task",
        4096,
        NULL,
        5,
        NULL,
        1 // Core 1
    );
}

void loop() {
    vTaskDelay(pdMS_TO_TICKS(10));
}
