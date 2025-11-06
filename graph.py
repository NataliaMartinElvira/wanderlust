import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt

# ---- CONFIG (Matching Original Script) ----
FS = 50.0        
LOW, HIGH, ORDER = 0.25, 2.5, 4
VM_PEAK_THR_G = 0.04
MIN_DIST_S = 0.5

# ---- UTILIDADES (Functions bandpass_vm and detect_peaks are the same) ----
def bandpass_vm(ax, ay, az, fs=FS, low=LOW, high=HIGH, order=ORDER):
    vm = np.sqrt(ax*ax + ay*ay + az*az)
    vm = vm - np.nanmean(vm)
    sos = butter(order, [low/(fs/2), high/(fs/2)], btype="band", output="sos")
    return sosfiltfilt(sos, vm)

def detect_peaks(signal, fs=FS, min_height=VM_PEAK_THR_G, min_distance_s=MIN_DIST_S):
    min_distance = int(min_distance_s * fs)
    peaks = []
    last_i = -10**9
    for i in range(1, len(signal)-1): 
        if signal[i] > min_height and signal[i] > signal[i-1] and signal[i] >= signal[i+1]:
            if i - last_i >= min_distance:
                peaks.append(i)
                last_i = i
    return np.array(peaks, dtype=int)

# --- ANÁLISIS PRINCIPAL Y PLOTEO ---

# 1. Cargar Datos
FILE_NAME = 'imu_seated_march.csv' 

try:
    # **AJUSTES CLAVE AQUÍ:**
    # 1. sep=';': Indica que el separador de columnas es el punto y coma.
    # 2. decimal=',': Indica que el separador decimal es la coma.
    # 3. encoding='utf-8': Volvemos al estándar, ya que el problema anterior era del decimal.
    
    df = pd.read_csv(
        FILE_NAME, 
        sep=';', 
        decimal=',',
        encoding='utf-8' 
    )
    print(f"Datos cargados exitosamente desde el archivo: {FILE_NAME}")

    # Eliminamos espacios en el encabezado (por si acaso)
    df.columns = df.columns.str.strip() 
    
except Exception as e:
    print(f"Error al cargar el archivo: {e}")
    print("Por favor, revise el nombre del archivo y la ruta.")
    exit()

# 2. Procesar Señal
# Estas líneas ahora deberían funcionar ya que las columnas están limpias y son numéricas.
t = df["time_ms"].values.astype(float)
ax = df["acc_x_g"].values.astype(float)
ay = df["acc_y_g"].values.astype(float)
az = df["acc_z_g"].values.astype(float)

vm_f = bandpass_vm(ax, ay, az)

# 3. Detectar Picos
original_peaks_i = detect_peaks(vm_f)

# 4. Aplicar el Ajuste Controversial (El punto clave del diagnóstico)
corrected_peaks_i = original_peaks_i[::2] if len(original_peaks_i) > 1 else original_peaks_i

# --- PLOTEO ---
plt.figure(figsize=(14, 6))

plt.plot(t/1000.0, vm_f, label='Vector Magnitud Filtrado (VM_f)', color='C0')

# Muestra todos los picos que el algoritmo ENCUENTRA antes del ajuste
plt.plot(t[original_peaks_i]/1000.0, vm_f[original_peaks_i], 
         'o', markersize=5, label='Picos Detectados Originalmente (P1, P2, P3, P4...)', color='red')

# Muestra los picos que el código USA para CONTAR el paso (Cada 2do pico)
plt.plot(t[corrected_peaks_i]/1000.0, vm_f[corrected_peaks_i], 
         'x', markersize=10, label='Pasos Contados Finales (Solo P1, P3, P5...)', color='green', markeredgewidth=2)

plt.axhline(y=VM_PEAK_THR_G, color='grey', linestyle='--', label=f'Umbral de Pico ({VM_PEAK_THR_G} g)')

plt.title('Análisis de Detección de Pasos: ¿Se Pierden Pasos con el Ajuste?')
plt.xlabel('Tiempo (segundos)')
plt.ylabel('Magnitud de Aceleración Dinámica (g)')
plt.legend()
plt.grid(True)
plt.show()