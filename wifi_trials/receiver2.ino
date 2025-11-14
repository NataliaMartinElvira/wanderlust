#include <esp_now.h>
#include <WiFi.h>

// --------------------------------------------------------------------------
// EXPANDED DATA STRUCTURE (MUST match the Sender)
// --------------------------------------------------------------------------
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

struct_message incomingReadings;

// Callback function executed when data is received (using the updated signature)
void OnDataRecv(const esp_now_recv_info * info, const uint8_t *incomingData, int len) {
    // Get the sender's MAC address
    const uint8_t * mac_addr = info->src_addr;
    
    // Copy the received data into the struct
    memcpy(&incomingReadings, incomingData, sizeof(incomingReadings));
  
    // Convert MAC address to a printable string
    char macStr[18];
    snprintf(macStr, sizeof(macStr), "%02x:%02x:%02x:%02x:%02x:%02x",
             mac_addr[0], mac_addr[1], mac_addr[2], mac_addr[3], mac_addr[4], mac_addr[5]);
  
    // Print the received data to the Serial Monitor (PC)
    Serial.print("Data from Node: ");
    Serial.print(incomingReadings.id);
    Serial.print(" | Time: ");
    Serial.print(incomingReadings.timestamp);
    Serial.print("ms | Acc(X,Y,Z): ");
    Serial.print(incomingReadings.acc_x, 3);
    Serial.print(", ");
    Serial.print(incomingReadings.acc_y, 3);
    Serial.print(", ");
    Serial.print(incomingReadings.acc_z, 3);
    Serial.print(" | Pitch/Roll: ");
    Serial.print(incomingReadings.pitch, 2);
    Serial.print(", ");
    Serial.print(incomingReadings.roll, 2);
    Serial.print(" | Gyr(X,Y,Z): ");
    Serial.print(incomingReadings.gyr_x, 3);
    Serial.print(", ");
    Serial.print(incomingReadings.gyr_y, 3);
    Serial.print(", ");
    Serial.println(incomingReadings.gyr_z, 3);
}
 
void setup() {
    Serial.begin(115200);
    
    // Set device as a Wi-Fi Station
    WiFi.mode(WIFI_STA);
    
    // Initialize ESP-NOW
    if (esp_now_init() != ESP_OK) {
        Serial.println("Error initializing ESP-NOW");
        return;
    }
    
    // Register the callback function to receive data
    esp_now_register_recv_cb(OnDataRecv);
    
    Serial.println("Receiver Node Ready. Waiting for data...");
}
 
void loop() {
    // Data reception is handled by the OnDataRecv callback
}
