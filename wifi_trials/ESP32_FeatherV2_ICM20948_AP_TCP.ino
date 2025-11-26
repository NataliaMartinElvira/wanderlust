
/*
  ESP32 Feather V2 + ICM-20948  (Wi‑Fi SoftAP + TCP CSV stream @ ~50 Hz)
  - SSID: IMU_Logger    PASS: imu12345
  - IP:   192.168.4.1   PORT: 3333
  - CSV header (first line after connect):
    time_ms,acc_x_g,acc_y_g,acc_z_g,pitch_deg,roll_deg,gyr_x_dps,gyr_y_dps,gyr_z_dps

  Notes:
  * Uses ICM20948_WE library (set address to 0x68 or 0x69 as per your breakout).
  * Feather ESP32 V2 I2C pins are printed on the board silkscreen (SDA/SCL).
    If needed, call Wire.begin(SDA_PIN, SCL_PIN). Here we rely on default Wire.begin().
  * Stream rate target ~50 Hz (DT_MS = 20).
*/

#include <WiFi.h>
#include <Wire.h>
#include <ICM20948_WE.h>

// ---- Adjust if your board uses 0x69 ----
#define ICM20948_ADDR 0x68
ICM20948_WE myIMU(ICM20948_ADDR);

// ---- Wi‑Fi AP ----
const char* AP_SSID = "IMU_Leg_Logger";
const char* AP_PASS = "imu12345";    // >= 8 chars
const uint16_t TCP_PORT = 3333;
WiFiServer server(TCP_PORT);
WiFiClient client;

// ---- timing ----
const uint32_t DT_MS = 20;  // ~50 Hz
uint32_t last_ms = 0;

void setup() {
  Serial.begin(115200);
  delay(200);

  // ---- Wi‑Fi AP ----
  WiFi.mode(WIFI_AP);
  bool ok = WiFi.softAP(AP_SSID, AP_PASS, 6, 0, 1);
  if (ok) {
    Serial.print("AP up  SSID: "); Serial.print(AP_SSID);
    Serial.print("  PASS: "); Serial.print(AP_PASS);
    Serial.print("  IP: ");   Serial.println(WiFi.softAPIP()); // 192.168.4.1
  } else {
    Serial.println("WARN: softAP failed");
  }
  server.begin();
  server.setNoDelay(true);

  // ---- I2C / IMU ----
  // If your Feather labels SDA/SCL differently, you may do: Wire.begin(SDA_PIN, SCL_PIN);
  Wire.begin();
  if (!myIMU.init()) {
    Serial.println("ERR: ICM20948 not responding (check I2C wiring/addr)");
  }

  // Ranges
  myIMU.setAccRange(ICM20948_ACC_RANGE_2G);
  myIMU.setGyrRange(ICM20948_GYRO_RANGE_250);

  // DLPF and sample dividers (keep modest bandwidth, aim ~50 Hz)
  myIMU.setAccDLPF(ICM20948_DLPF_6);
  myIMU.setGyrDLPF(ICM20948_DLPF_6);
  myIMU.setAccSampleRateDivider(10);
  myIMU.setGyrSampleRateDivider(10);
}

void loop() {
  // Accept client and send CSV header
  if (!client || !client.connected()) {
    client = server.available();
    if (client && client.connected()) {
      client.println("time_ms,acc_x_g,acc_y_g,acc_z_g,pitch_deg,roll_deg,gyr_x_dps,gyr_y_dps,gyr_z_dps");
      Serial.println("Client connected.");
    }
    delay(10);
    return;
  }

  uint32_t now = millis();
  if (now - last_ms >= DT_MS) {
    last_ms = now;

    // Read sensors
    myIMU.readSensor();
    xyzFloat acc; myIMU.getGValues(&acc);
    xyzFloat gyr; myIMU.getGyrValues(&gyr);
    float pitch = myIMU.getPitch();
    float roll  = myIMU.getRoll();

    // Stream one CSV row
    client.print((unsigned long)now); client.print(",");
    client.print(acc.x, 3); client.print(",");
    client.print(acc.y, 3); client.print(",");
    client.print(acc.z, 3); client.print(",");
    client.print(pitch, 2); client.print(",");
    client.print(roll, 2);  client.print(",");
    client.print(gyr.x, 3); client.print(",");
    client.print(gyr.y, 3); client.print(",");
    client.println(gyr.z, 3);

    if (!client.connected()) client.stop();
  }
}
