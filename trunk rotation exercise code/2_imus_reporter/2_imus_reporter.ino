#include <WiFi.h>
#include <Wire.h>
#include <ICM20948_WE.h>

/*** ---------- HUZZAH32 I2C pins ---------- ***/
#define SDA_PIN 23
#define SCL_PIN 22

/*** ---------- PCA9546A (4 channels) ---------- ***/
#define PCA9546A_ADDR 0x70   // A0..A2 = GND
static inline bool pcaSelect(uint8_t channel) {
  if (channel > 3) return false;
  Wire.beginTransmission(PCA9546A_ADDR);
  Wire.write(1 << channel);
  if (Wire.endTransmission() != 0) return false;
  delay(2);  // small settle
  return true;
}
static inline void pcaDisableAll() {
  Wire.beginTransmission(PCA9546A_ADDR);
  Wire.write(0x00);
  Wire.endTransmission();
}

/*** ---------- IMUs ---------- ***/
// Both IMUs strapped AD0 = HIGH -> 0x69 (OK behind mux)
// CH0 uses channel 0; CH3 uses channel 3
#define ICM20948_ADDR_CH0 0x69
#define ICM20948_ADDR_CH3 0x69

ICM20948_WE imu_ch0(ICM20948_ADDR_CH0);   // PCA channel 0
ICM20948_WE imu_ch3(ICM20948_ADDR_CH3);   // PCA channel 3

/*** ---------- WiFi AP / TCP ---------- ***/
const char* AP_SSID = "IMU_Logger";
const char* AP_PASS = "imu12345";
const uint16_t TCP_PORT = 3333;

WiFiServer server(TCP_PORT);
WiFiClient client;

/*** ---------- Helpers ---------- ***/
bool initIMU(ICM20948_WE& imu, uint8_t ch, uint8_t addr, const char* tag) {
  if (!pcaSelect(ch)) {
    Serial.printf("ERR: PCA select CH%u failed for %s\n", ch, tag);
    return false;
  }
  delay(2);
  if (!imu.init()) {
    Serial.printf("ERR: %s not responding (I2C addr 0x%02X on CH%u)\n", tag, addr, ch);
    pcaDisableAll();
    return false;
  }

  imu.setAccRange(ICM20948_ACC_RANGE_2G);
  imu.setAccDLPF(ICM20948_DLPF_6);
  imu.setAccSampleRateDivider(10);

  imu.setGyrRange(ICM20948_GYRO_RANGE_250);
  imu.setGyrDLPF(ICM20948_DLPF_6);
  imu.setGyrSampleRateDivider(10);

  Serial.printf("%s initialized on CH%u @0x%02X\n", tag, ch, addr);
  pcaDisableAll();
  return true;
}

void startAP() {
  WiFi.persistent(false);
  WiFi.disconnect(true, true);
  delay(200);

  WiFi.mode(WIFI_AP);
  WiFi.setSleep(false);
  WiFi.softAPConfig(IPAddress(192,168,4,1),
                    IPAddress(192,168,4,1),
                    IPAddress(255,255,255,0));

  bool ok = WiFi.softAP(AP_SSID, AP_PASS, /*channel*/6, /*hidden*/0, /*max_conn*/4);
  Serial.printf("softAP(): %s, SSID=%s, PASS=%s, IP=%s, CH=%d\n",
                ok ? "OK" : "FAIL",
                AP_SSID, AP_PASS,
                WiFi.softAPIP().toString().c_str(),
                WiFi.channel());
  WiFi.setTxPower(WIFI_POWER_19_5dBm);

  server.begin();
  server.setNoDelay(true);
  Serial.println("TCP server listening on :3333");
}

/*** ---------- Optional: quick channel scanner at boot ---------- ***/
void scanBus(const char* tag) {
  for (uint8_t a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.printf("%s: found 0x%02X\n", tag, a);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(200);

  startAP();

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(100000);

  Wire.beginTransmission(PCA9546A_ADDR);
  if (Wire.endTransmission() != 0) {
    Serial.printf("ERR: No PCA9546A at 0x%02X. Check wiring/power/ADDR pins.\n", PCA9546A_ADDR);
  } else {
    Serial.printf("PCA9546A detected at 0x%02X\n", PCA9546A_ADDR);
  }

  // Clean scans with channels disabled between steps
  pcaDisableAll();
  Serial.println("Scan UPSTREAM (expect only 0x70):");
  scanBus("UP");

  if (pcaSelect(0)) { Serial.println("Scan CH0 (your IMU on 0x69):"); scanBus("CH0"); pcaDisableAll(); }
  if (pcaSelect(3)) { Serial.println("Scan CH3 (your IMU on 0x69):"); scanBus("CH3"); pcaDisableAll(); }

  bool ok0 = initIMU(imu_ch0, 0, ICM20948_ADDR_CH0, "IMU_CH0");
  bool ok3 = initIMU(imu_ch3, 3, ICM20948_ADDR_CH3, "IMU_CH3");
  if (!ok0 || !ok3) {
    Serial.println("WARN: One or both IMUs failed to init. Continuing so you can still connect and see logs.");
  }

  // Wire.setClock(400000); // optional once stable
}

void loop() {
  if (!client || !client.connected()) {
    WiFiClient c = server.available();
    if (c && c.connected()) {
      client = c;
      Serial.println("Client connected");
      client.println("imu_id,time_ms,acc_x_g,acc_y_g,acc_z_g,pitch_deg,roll_deg,gyr_x_dps,gyr_y_dps,gyr_z_dps");
    }
    delay(5);
    return;
  }

  const unsigned long t = millis();

  // Read CH0
  if (pcaSelect(0)) {
    imu_ch0.readSensor();
    xyzFloat acc; imu_ch0.getGValues(&acc);
    xyzFloat gyr; imu_ch0.getGyrValues(&gyr);
    float pitch = imu_ch0.getPitch();
    float roll  = imu_ch0.getRoll();

    client.print("IMU_CH0,");
    client.print(t); client.print(",");
    client.print(acc.x, 3); client.print(",");
    client.print(acc.y, 3); client.print(",");
    client.print(acc.z, 3); client.print(",");
    client.print(pitch, 2); client.print(",");
    client.print(roll, 2); client.print(",");
    client.print(gyr.x, 3); client.print(",");
    client.print(gyr.y, 3); client.print(",");
    client.println(gyr.z, 3);
  }

  // Read CH3
  if (pcaSelect(3)) {
    imu_ch3.readSensor();
    xyzFloat acc; imu_ch3.getGValues(&acc);
    xyzFloat gyr; imu_ch3.getGyrValues(&gyr);
    float pitch = imu_ch3.getPitch();
    float roll  = imu_ch3.getRoll();

    client.print("IMU_CH3,");
    client.print(t); client.print(",");
    client.print(acc.x, 3); client.print(",");
    client.print(acc.y, 3); client.print(",");
    client.print(acc.z, 3); client.print(",");
    client.print(pitch, 2); client.print(",");
    client.print(roll, 2); client.print(",");
    client.print(gyr.x, 3); client.print(",");
    client.print(gyr.y, 3); client.print(",");
    client.println(gyr.z, 3);
  }

  if (!client.connected()) {
    client.stop();
    Serial.println("Client disconnected");
  }

  delay(100);
}
