#include <WiFi.h>
#include <Wire.h>
#include <ICM20948_WE.h>

#define ICM20948_ADDR 0x68   // If your board uses 0x69, change this.
ICM20948_WE myIMU(ICM20948_ADDR);

const char* AP_SSID = "IMU_Logger";
const char* AP_PASS = "imu12345";          // >= 8 chars
const uint16_t TCP_PORT = 3333;

WiFiServer server(TCP_PORT);
WiFiClient client;

void setup() {
  // Start Serial only for optional debugging during upload. Not required later.
  Serial.begin(115200);
  delay(200);

  // ---- Wi-Fi AP (fixed IP 192.168.4.1) ----
  WiFi.mode(WIFI_AP);
  // Optional: lock to channel 6, max 1 client
  bool ok = WiFi.softAP(AP_SSID, AP_PASS, 6, 0, 1);
  if (ok) {
    IPAddress ip = WiFi.softAPIP();
    Serial.print("AP up. SSID: "); Serial.print(AP_SSID);
    Serial.print("  PASS: "); Serial.print(AP_PASS);
    Serial.print("  IP: "); Serial.println(ip); // 192.168.4.1
  } else {
    Serial.println("WARN: softAP failed");
  }

  server.begin();
  server.setNoDelay(true);

  // ---- IMU ----
  Wire.begin();
  if (!myIMU.init()) {
    Serial.println("ERR: ICM20948 not responding (check I2C wiring/addr)");
    // Keep running so you can still connect and see the error
  }

  myIMU.setAccRange(ICM20948_ACC_RANGE_2G);
  myIMU.setAccDLPF(ICM20948_DLPF_6);
  myIMU.setAccSampleRateDivider(10);

  myIMU.setGyrRange(ICM20948_GYRO_RANGE_250);
  myIMU.setGyrDLPF(ICM20948_DLPF_6);
  myIMU.setGyrSampleRateDivider(10);
}

void loop() {
  // Accept a client if none is connected
  if (!client || !client.connected()) {
    client = server.available();
    if (client && client.connected()) {
      // Send CSV header once per connection
      client.println("time_ms,acc_x_g,acc_y_g,acc_z_g,pitch_deg,roll_deg,gyr_x_dps,gyr_y_dps,gyr_z_dps");
    }
    delay(10);
    return;
  }

  // Read sensors and stream a CSV row every 100 ms
  myIMU.readSensor();
  xyzFloat acc; myIMU.getGValues(&acc);
  xyzFloat gyr; myIMU.getGyrValues(&gyr);
  float pitch = myIMU.getPitch();
  float roll  = myIMU.getRoll();
  unsigned long t = millis();

  client.print(t); client.print(",");
  client.print(acc.x, 3); client.print(",");
  client.print(acc.y, 3); client.print(",");
  client.print(acc.z, 3); client.print(",");
  client.print(pitch, 2); client.print(",");
  client.print(roll, 2); client.print(",");
  client.print(gyr.x, 3); client.print(",");
  client.print(gyr.y, 3); client.print(",");
  client.println(gyr.z, 3);

  if (!client.connected()) client.stop();

  delay(100);  // ~10 Hz
}
