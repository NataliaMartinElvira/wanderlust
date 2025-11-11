#include <WiFi.h>
#include <WiFiUdp.h>

//Receiver IP address: 172.20.10.2



// --- Configuration ---
const char* ssid = "iPhone de Laura";          // Replace with your Wi-Fi name
const char* password = "ponceorozcolaura";  // Replace with your Wi-Fi password
unsigned int localUdpPort = 8888;             // Local port to listen on

// --- Global Variables ---
WiFiUDP Udp;
char incomingPacket[255]; // Buffer for incoming data

void setup() {
  Serial.begin(115200);
  delay(100);

  // Connect to Wi-Fi
  Serial.print("Connecting to ");
  Serial.println(ssid);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected.");
  Serial.print("Receiver IP address: ");
  Serial.println(WiFi.localIP());

  // Start listening for UDP packets
  Udp.begin(localUdpPort);
  Serial.print("Listening on UDP port ");
  Serial.println(localUdpPort);
}

void loop() {
  int packetSize = Udp.parsePacket();

  if (packetSize) {
    // Read the packet into the buffer
    int len = Udp.read(incomingPacket, 255);
    if (len > 0) {
      incomingPacket[len] = 0; // Null-terminate the string
    }

    // Print the sender's IP and the received data
    Serial.print("Packet from ");
    Serial.print(Udp.remoteIP());
    Serial.print(":");
    Serial.print(Udp.remotePort());
    Serial.print(" - Data: ");
    Serial.println(incomingPacket);
  }
}
