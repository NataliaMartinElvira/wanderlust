import os
import pandas as pd
import numpy as np
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt

# ---- CONFIGURACIÓN (Parámetros Fijos) ----
FS = 50.0        # Frecuencia de muestreo en Hz
LOW, HIGH, ORDER = 0.25, 2.5, 4 # Parámetros del filtro pasa banda
VM_PEAK_THR_G = 0.04 # Umbral de magnitud vectorial para detectar picos (en g)
MIN_DIST_S = 0.5     # Distancia mínima entre picos (en segundos)

# ---- UTILIDADES (Funciones de Procesamiento) ----
def bandpass_vm(ax, ay, az, fs=FS, low=LOW, high=HIGH, order=ORDER):
    """Calcula la magnitud vectorial y le aplica un filtro pasa banda."""
    vm = np.sqrt(ax*ax + ay*ay + az*az)
    vm = vm - np.nanmean(vm)
    sos = butter(order, [low/(fs/2), high/(fs/2)], btype="band", output="sos")
    return sosfiltfilt(sos, vm)

def detect_peaks(signal, fs=FS, min_height=VM_PEAK_THR_G, min_distance_s=MIN_DIST_S):
    """Detecta picos locales que superan el umbral y cumplen la distancia mínima."""
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

FILE_NAME = 'imu_seated_march_20251124_205245.xlsx'

# Cargar (ya sabemos que pd.read_excel funciona en tu caso)
try:
    df = pd.read_excel(FILE_NAME, engine='openpyxl')
    print(f"✅ Leído como Excel con pd.read_excel (openpyxl).")
    print("✅ DataFrame cargado. Columnas:", df.columns.tolist())
except Exception as e:
    print(f"❌ Error al leer el archivo: {e}")
    raise

# Comprobar columnas esperadas
expected_cols = ["time_ms", "acc_x_g", "acc_y_g", "acc_z_g"]
for c in expected_cols:
    if c not in df.columns:
        raise KeyError(f"Falta columna esperada: {c}")

# Extraer vectores (y convertir a float)
t = df["time_ms"].astype(float).values
ax = df["acc_x_g"].astype(float).values
ay = df["acc_y_g"].astype(float).values
az = df["acc_z_g"].astype(float).values

print(f"Registros: {len(t)} muestras")
print(f"Tiempo (ms) -> min: {np.nanmin(t)}, max: {np.nanmax(t)}")
print(f"AX -> min/max: {np.nanmin(ax)}/{np.nanmax(ax)}")
print(f"AY -> min/max: {np.nanmin(ay)}/{np.nanmax(ay)}")
print(f"AZ -> min/max: {np.nanmin(az)}/{np.nanmax(az)}")

# Si t no es monótono o tiene problemas, reconstruir el eje tiempo usando FS
if not np.all(np.diff(t) > 0):
    print("⚠️ El vector time_ms no es monotónico. Se reconstruirá tiempo usando FS.")
    t = np.arange(len(ax)) * (1000.0 / FS)  # ms

# Aplicar filtro
vm_f = bandpass_vm(ax, ay, az)

# Verificar la señal filtrada
if not np.isfinite(vm_f).all():
    print("⚠️ La señal filtrada contiene NaN/Inf. Se tomarán las partes finitas.")
    finite_mask = np.isfinite(vm_f)
else:
    finite_mask = np.ones_like(vm_f, dtype=bool)

print(f"VM filtrada -> min: {np.nanmin(vm_f)}, max: {np.nanmax(vm_f)}")

# Detectar picos
original_peaks_i = detect_peaks(vm_f, fs=FS)
corrected_peaks_i = original_peaks_i  # mismo comportamiento que antes
num_pasos = len(corrected_peaks_i)
print(f"👣 Número de pasos contados: {num_pasos}")

# Preparar gráfico
fig, axplt = plt.subplots(figsize=(14, 6))
time_s = t / 1000.0

# Trazar señal (usar máscara finita para evitar errores)
axplt.plot(time_s[finite_mask], vm_f[finite_mask], label='Vector Magnitud Filtrado (VM_f)')

# Picos detectados (verificar índices en rango)
valid_orig = original_peaks_i[(original_peaks_i >= 0) & (original_peaks_i < len(vm_f))]
valid_corr = corrected_peaks_i[(corrected_peaks_i >= 0) & (corrected_peaks_i < len(vm_f))]

if valid_orig.size > 0:
    axplt.plot(time_s[valid_orig], vm_f[valid_orig], 'o', markersize=5,
               label='Picos Detectados Originalmente (Posibles picos de oscilación)', color='red')
if valid_corr.size > 0:
    axplt.plot(time_s[valid_corr], vm_f[valid_corr], 'x', markersize=10,
               label=f'Pasos Contados Finales (N={num_pasos})', color='green', markeredgewidth=2)

axplt.axhline(y=VM_PEAK_THR_G, linestyle='--', label=f'Umbral de Pico ({VM_PEAK_THR_G} g)')
axplt.set_title('Análisis de Detección de Pasos de Marcha Sentado (VM Filtrada)')
axplt.set_xlabel('Tiempo (segundos)')
axplt.set_ylabel('Magnitud de Aceleración Dinámica (g)')
axplt.legend()
axplt.grid(True)

# Guardar figura a disco (solución para entornos sin GUI)
out_png = "vm_plot.png"
plt.tight_layout()
plt.savefig(out_png, dpi=200, bbox_inches='tight')
print(f"🖼️ Gráfica guardada en: {os.path.abspath(out_png)}")

# Además, intentar mostrar (esto funciona solo si tu entorno tiene soporte GUI)
try:
    plt.show(block=False)
    print("Si tu entorno soporta ventana gráfica, la imagen debería mostrarse ahora. Si no, abre el archivo guardado.")
except Exception as e:
    print(f"Nota: plt.show() falló o no abrió ventana gráfica: {e}")

# Guardar picos a CSV para inspección
peaks_df = pd.DataFrame({
    "idx": valid_corr,
    "time_ms": t[valid_corr],
    "time_s": (t[valid_corr] / 1000.0),
    "vm_value": vm_f[valid_corr]
})
peaks_csv = "peaks.csv"
peaks_df.to_csv(peaks_csv, index=False, sep=';')
print(f"📄 Índices y tiempos de picos guardados en: {os.path.abspath(peaks_csv)}")

# Mensajes finales útiles
print("Hecho. Revisa los archivos 'vm_plot.png' y 'peaks.csv' en el directorio actual.")
