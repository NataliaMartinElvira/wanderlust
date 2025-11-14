#include <esp_now.h>
#include <WiFi.h>
#include <ICM20948_WE.h>
#include <Wire.h> 
// FreeRTOS includes for stable delay
#include "freertos/FreeRTOS.h"
#include "freertos/task.h" 

// --------------------------------------------------------------------------
// !!! IMPORTANT: CUSTOMIZE THESE VALUES !!!
// --------------------------------------------------------------------------
// 1. Set a unique ID for this sensor. (CHANGE THIS to 2 for the second board)
#define NODE_ID 1 

// 2. MAC Address of the Receiver Node 
uint8_t broadcastAddress[] = {0x10, 0x06, 0x1C, 0x18, 0x69, 0x00};

// 3. Adafruit Feather I2C Pins (MUST use 21/22)
#define I2C_SDA_PIN 21 
#define I2C_SCL_PIN 22 
// --------------------------------------------------------------------------

// EXPANDED DATA STRUCTURE (Must match the Receiver)
typedef struct struct_message {
    int id;
    unsigned long timestamp;
    float acc_x;
    float acc_y;
    float acc_z;
    float pitch;
    float roll;
    float gyr_x;
    float gyr_y;
    float gyr_z;
} struct_message;

struct_message myData;
 
// ICM20948 IMU SETUP
#define ICM20948_ADDR 0x68
ICM20948_WE myIMU = ICM20948_WE(ICM20948_ADDR); 

// Callback function executed when data is sent
void OnDataSent(const wifi_tx_info_t *info, esp_now_send_status_t status) {
    if (status == ESP_NOW_SEND_SUCCESS) {
        // Success
    } else {
        Serial.println("Error sending data.");
    }
}
 
void setup() {
    Serial.begin(115200);
    
    // Safety measure for Core Watchdog (prevents setup timeout)
    disableCore0WDT(); 
    
    randomSeed(analogRead(0)); 
  
    // ---------------------- IMU INITIALIZATION ------------------------
    Serial.println("Starting I2C and IMU init...");
    
    // CRITICAL FIX: Initialize I2C with EXPLICIT PINS (21 for SDA, 22 for SCL)
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN); 
    delay(100); 
    
    if (!myIMU.init()) {
        Serial.println("FATAL: ICM20948 not found. Check wiring (Pin 21/22) or address.");
        while(1) {
             delay(100); 
        } 
    }
    Serial.println("ICM20948 Initialized successfully.");
    // ------------------------------------------------------------------
  
    // Wi-Fi and ESP-NOW setup
    WiFi.mode(WIFI_STA);
    
    if (esp_now_init() != ESP_OK) {
        Serial.println("FATAL: Error initializing ESP-NOW");
        return;
    }
  
    // Register the send callback function
    esp_now_register_send_cb(OnDataSent);
  
    // Register the Receiver (Peer)
    esp_now_peer_info_t peerInfo = {}; 
    memcpy(peerInfo.peer_addr, broadcastAddress, 6);
    peerInfo.channel = 0;  
    peerInfo.encrypt = false;
    
    if (esp_now_add_peer(&peerInfo) != ESP_OK){
        Serial.println("FATAL: Failed to add peer. Check MAC address.");
        return;
    }
    
    Serial.print("Sender Node ");
    Serial.print(NODE_ID);
    Serial.println(" initialized and peer added. Starting data transmission.");
}
 
void loop() {
    // --- START SENSOR READING AND DATA PACKAGING ---
    
    unsigned long timestamp_ms = millis();
    myIMU.readSensor();
    
    // Get sensor values 
    xyzFloat acc;
    xyzFloat gyr;
    myIMU.getGValues(&acc);   
    myIMU.getGyrValues(&gyr); 
    float pitch = myIMU.getPitch();
    // **FIX 1: Use the dot operator (.)**
    float roll = myIMU.getRoll(); 
    
    // 2. Load data into the struct
    myData.id = NODE_ID;
    myData.timestamp = timestamp_ms;
    myData.acc_x = acc.x;
    myData.acc_y = acc.y;
    myData.acc_z = acc.z;
    myData.pitch = pitch;
    myData.roll = roll;
    myData.gyr_x = gyr.x;
    myData.gyr_y = gyr.y;
    myData.gyr_z = gyr.z;
    
    // --- END SENSOR READING AND DATA PACKAGING ---

    // 1. Print a sample of the data to the sender's Serial Monitor for verification
    Serial.print("Sending Node ");
    Serial.print(NODE_ID);
    Serial.print(" - Pitch: ");
    Serial.println(myData.pitch);

    // 2. Send the data to the registered peer (the Receiver)
    esp_now_send(broadcastAddress, (uint8_t *) &myData, sizeof(myData));
    
    // **FIX 2: Use the correct uppercase macro name**
    vTaskDelay(pdMS_TO_TICKS(50)); 
}
