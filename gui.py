import os
import sys
import gc
import glob
import json
import time
import traceback
import psutil
from datetime import datetime
import numpy as np
import pandas as pd

# Configura la GPU (o fuerza CPU con --cpu) y los mensajes de TensorFlow antes de importar TF/Sionna.
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


# Detecta y configura de forma segura los dispositivos GPU disponibles.
def configure_compute_device():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return "CPU", False
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as e:
            print(f"Advertencia al configurar memoria GPU: {e}")
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        device_name = details.get("device_name", gpus[0].name)
    except Exception:
        device_name = gpus[0].name
    return f"GPU: {device_name}", True


COMPUTE_DEVICE_INFO, HAS_GPU = configure_compute_device()
tf.get_logger().setLevel('ERROR')

import sionna

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QCheckBox, QPushButton, QProgressBar, QPlainTextEdit, QFileDialog,
    QMessageBox, QLabel, QSplitter, QScrollArea, QStatusBar
)
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QStandardItemModel

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# =====================================================================
# 0. CONSTANTES Y UTILIDADES DE PERSISTENCIA
# =====================================================================

# Longitudes en bits de cada polinomio CRC soportado por Sionna 5G NR.
CRC_LENGTHS = {
    "CRC24A": 24,
    "CRC24B": 24,
    "CRC24C": 24,
    "CRC16": 16,
    "CRC11": 11,
    "CRC6": 6,
}

# Tasas de código estándar predefinidas (etiqueta, tasa nominal R = K/N).
PREDEFINED_CODE_RATES = [
    ("1/8 (0.125)", 1 / 8),
    ("1/4 (0.250)", 1 / 4),
    ("1/3 (0.333)", 1 / 3),
    ("3/8 (0.375)", 3 / 8),
    ("1/2 (0.500)", 1 / 2),
    ("5/8 (0.625)", 5 / 8),
    ("2/3 (0.667)", 2 / 3),
    ("3/4 (0.750)", 3 / 4),
    ("5/6 (0.833)", 5 / 6),
    ("7/8 (0.875)", 7 / 8),
]

# Valores SNR de diseño recomendados por canal (verificados para R = 0.5).
CHANNEL_SNR_DEFAULTS = {
    "awgn": -1.2,
    "rayleigh": 1.0,
    "rice": -0.3,
    "lognormal": 2.4,
    "suzuki": 5.0,
    "nakagami": 0.3,
    "shadowed_rician": 0.5,
    "impulsive": -1.9,
}

# Canales sin desvanecimiento multiplicativo (h = 1, permiten ecualizador 'no_eq').
NON_FADING_CHANNELS = ("awgn", "impulsive")

# Etiquetas y estilos de marcador compartidos para las curvas BER.
CURVE_UNCODED = "Sin código"
CURVE_POLAR = "Código Polar"
CURVE_LDPC = "Código LDPC 5G NR"
CURVE_MARKERS = {
    CURVE_UNCODED: "o-",
    CURVE_POLAR: "s-",
    CURVE_LDPC: "D-",
}

# Parámetros operativos de Monte Carlo, Auto SNR y gráfica BER.
MC_RAM_LIMIT_PERCENT = 85.0
MC_BACKUP_INTERVAL = 10
MC_CHUNK_SYNC = 50
AUTO_SNR_TARGET_ALPHA = 0.88
AUTO_SNR_BISECTION_STEPS = 12
BER_MIN_YLIM = 1e-6
BER_FLOOR_YLIM = 1e-9

# Claves obligatorias (retrocompatibles) y completas de configuración en respaldos Monte Carlo.
MC_BACKUP_CORE_KEYS = (
    "N", "K", "bitsxsimb", "canal", "bloq_coher",
    "k_factor_db", "sigma_db", "ecualizador", "LLR_clip", "snr_simulacion"
)
MC_BACKUP_KEYS = MC_BACKUP_CORE_KEYS + ("m_nakagami", "imp_prob", "imp_gamma")


# Calcula K a partir de N, la tasa seleccionada y bitsxsimb (ajustando múltiplos cuando aplica).
def compute_info_bits(N, rate_spec, bitsxsimb=1):
    if isinstance(rate_spec, tuple) and len(rate_spec) == 2 and rate_spec[0] == "exact_k":
        return max(1, min(N - 1, int(rate_spec[1])))
    k = int(round(N * float(rate_spec)))
    if bitsxsimb > 1 and N % bitsxsimb == 0 and k % bitsxsimb != 0:
        snapped = int(round(k / bitsxsimb)) * bitsxsimb
        k = max(bitsxsimb, min(N - bitsxsimb, snapped))
    return max(1, min(N - 1, k))


# Formatea una duración en segundos como MM:SS o HH:MM:SS.
def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


# Extrae los metadatos relevantes de Monte Carlo para validar respaldos.
def extract_mc_backup_meta(params):
    return {k: params[k] for k in MC_BACKUP_KEYS if k in params}


# Guarda de forma atómica el progreso Monte Carlo junto con su índice y metadatos.
def save_mc_backup(backup_path, ber_bits, next_index, params):
    tmp_path = f"{backup_path}.tmp.npz"
    meta_json = json.dumps(extract_mc_backup_meta(params), sort_keys=True)
    np.savez(
        tmp_path,
        ber_bits=np.asarray(ber_bits, dtype=np.float64),
        next_index=np.int64(next_index),
        meta_json=np.array(meta_json)
    )
    # Reintentos breves ante bloqueos temporales de archivo en Windows (antivirus/OneDrive).
    for attempt in range(3):
        try:
            os.replace(tmp_path, backup_path)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.05)


# Evalúa los recursos de cómputo y memoria disponibles antes de simular y sugiere ajustes.
def assess_system_resources(params, mode, has_gpu=HAS_GPU, mem_info=None, cpu_count=None,
                            lock_n=False):
    mem = mem_info if mem_info is not None else psutil.virtual_memory()
    avail_gb = mem.available / (1024 ** 3)
    mem_pct = float(mem.percent)
    cores = cpu_count if cpu_count is not None else (os.cpu_count() or 1)

    N = int(params['N'])
    warnings = []
    recommendations = {}

    if mode == "mc":
        lote_mc = int(params['lote_mc'])
        num_iter = int(params['num_iter'])
        total_ops = N * lote_mc * num_iter
        est_batch_gb = (lote_mc * N * 64) / (1024 ** 3)

        if mem_pct >= 75.0 or avail_gb < 2.0:
            warnings.append(
                f"Memoria RAM disponible limitada ({avail_gb:.1f} GB libres, {mem_pct:.0f}% en uso)."
            )
            recommendations['lote_mc'] = min(lote_mc, 500)
            if not lock_n and N > 512:
                recommendations['N'] = 512

        if est_batch_gb > max(0.5, avail_gb * 0.4):
            warnings.append(
                f"El lote paralelo ({lote_mc} × N={N}, ~{est_batch_gb:.2f} GB estimados) "
                "es alto para la RAM libre actual."
            )
            recommendations['lote_mc'] = min(
                recommendations.get('lote_mc', lote_mc), max(100, min(lote_mc // 2, 1000))
            )
            if not lock_n and N > 512:
                recommendations['N'] = min(recommendations.get('N', N), N // 2)

        if not has_gpu and total_ops > 200_000_000:
            warnings.append(
                f"Ejecutando en CPU ({cores} hilos, sin GPU NVIDIA): {N} bit-canales × "
                f"{lote_mc * num_iter:,} muestras puede tardar varias horas."
            )
            recommendations['lote_mc'] = min(recommendations.get('lote_mc', lote_mc), 500)
            recommendations['num_iter'] = min(num_iter, 400)
            if not lock_n and N > 256:
                recommendations['N'] = min(recommendations.get('N', N), 256)

    elif mode == "ber":
        lote_ber = int(params['lote_berplot'])
        ebno_pts = int(params['ebno_rango'])
        list_scl = int(params['list_scl'])
        est_ber_gb = (lote_ber * N * list_scl * 32) / (1024 ** 3)

        if mem_pct >= 75.0 or avail_gb < 2.0 or est_ber_gb > max(0.5, avail_gb * 0.4):
            warnings.append(
                f"Memoria RAM disponible limitada ({avail_gb:.1f} GB libres, {mem_pct:.0f}% en uso) "
                f"para lote BER={lote_ber} con SCL={list_scl}."
            )
            recommendations['lote_berplot'] = max(20, min(lote_ber // 2, 100))
            if list_scl > 4:
                recommendations['list_scl'] = 4

        if not has_gpu and (N >= 1024 or ebno_pts > 15 or list_scl >= 8 or lote_ber > 100):
            warnings.append(
                f"Ejecutando en CPU ({cores} hilos, sin GPU NVIDIA): el barrido BER "
                f"({ebno_pts} puntos, N={N}, SCL={list_scl}) puede ser muy lento."
            )
            recommendations['lote_berplot'] = min(
                recommendations.get('lote_berplot', lote_ber), 100
            )
            recommendations['ebno_rango'] = min(ebno_pts, 11)
            recommendations['list_scl'] = min(recommendations.get('list_scl', list_scl), 4)
            if not lock_n and N >= 1024:
                recommendations['N'] = 512

    # Filtra recomendaciones que no reducen el valor actual.
    recommendations = {k: v for k, v in recommendations.items() if v < params.get(k, v)}
    summary = {
        "has_gpu": has_gpu,
        "avail_gb": avail_gb,
        "mem_pct": mem_pct,
        "cores": cores,
    }
    return warnings, recommendations, summary


# Carga un respaldo Monte Carlo en formato .npz si existe y es válido.
def load_mc_backup(backup_path):
    with np.load(backup_path, allow_pickle=False) as data:
        ber_bits = np.array(data["ber_bits"], dtype=np.float64)
        next_index = int(data["next_index"])
        meta = json.loads(str(data["meta_json"]))
    return ber_bits, next_index, meta


# Construye la ruta estándar para guardar los resultados CSV de BER_bits.
def default_ber_bits_csv_path(params, n_len=None):
    N = int(n_len) if n_len is not None else int(params['N'])
    fecha = datetime.now().strftime("%y%m%d")
    filename = (
        f"{fecha}_{params['canal']}_{params['ecualizador']}_{params['snr_simulacion']}SNR_"
        f"BER_bits_{N}_{params['K']}_{params['num_iter']}iter.csv"
    )
    return os.path.join(f"results_{N}", filename)


# Construye la ruta estándar para exportar las curvas BER vs Eb/N0 en CSV.
def default_ber_curves_csv_path(params):
    N = int(params['N'])
    fecha = datetime.now().strftime("%y%m%d")
    filename = f"{fecha}_{params['canal']}_{params['ecualizador']}_BER_curves_{N}_{params['K']}.csv"
    return os.path.join(f"results_{N}", filename)


# =====================================================================
# 1. MODELOS DE CANAL Y ECUALIZADORES (compilados con XLA)
# =====================================================================

# Genera ruido gaussiano complejo CN(0, no) con la misma forma que x.
def _sample_complex_awgn(shape, no):
    stddev = tf.sqrt(no / 2.0)
    awgn_real = sionna.phy.config.tf_rng.normal(shape, mean=0.0, stddev=stddev)
    awgn_imag = sionna.phy.config.tf_rng.normal(shape, mean=0.0, stddev=stddev)
    return tf.complex(awgn_real, awgn_imag)


# Calcula cuántos bloques de coherencia cubren una secuencia de longitud n_len.
def _fading_repetitions(n_len, bloq_coher):
    return tf.cast(tf.math.floordiv(n_len + bloq_coher - 1, bloq_coher), tf.int32)


# Canal AWGN: ganancia plana unitaria y ruido gaussiano complejo.
@tf.function(jit_compile=True)
def canalAWGN(ebno_db, bitsxsimb, coderate, x):
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    awgn = _sample_complex_awgn(tf.shape(x), no)
    h = tf.complex(1.0, 0.0)
    y = x + awgn
    return y, h, no


# Canal Rayleigh: desvanecimiento independiente por bloques de coherencia.
@tf.function(jit_compile=True)
def canalRayleigh(bloq_coher, ebno_db, bitsxsimb, coderate, x):
    batch_size = tf.shape(x)[0]
    bloq_coher = tf.cast(bloq_coher, tf.int32)
    N = tf.shape(x)[-1]
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    repeticiones = _fading_repetitions(N, bloq_coher)
    awgn = _sample_complex_awgn(tf.shape(x), no)

    h_real = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], stddev=tf.sqrt(0.5), dtype=tf.float32)
    h_imag = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], stddev=tf.sqrt(0.5), dtype=tf.float32)
    h = tf.repeat(tf.complex(h_real, h_imag), repeats=bloq_coher, axis=1)[:, :N]
    y = h * x + awgn
    return y, h, no


# Canal Rice: combina componente de línea de vista y desvanecimiento disperso.
@tf.function(jit_compile=True)
def canalRice(bloq_coher, k_factor_db, ebno_db, bitsxsimb, coderate, x):
    batch_size = tf.shape(x)[0]
    bloq_coher = tf.cast(bloq_coher, tf.int32)
    N = tf.shape(x)[-1]
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    repeticiones = _fading_repetitions(N, bloq_coher)
    awgn = _sample_complex_awgn(tf.shape(x), no)

    k_factor_lin = tf.pow(10.0, tf.cast(k_factor_db, tf.float32) / 10.0)
    h_nlos_real = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], stddev=tf.sqrt(0.5), dtype=tf.float32)
    h_nlos_imag = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], stddev=tf.sqrt(0.5), dtype=tf.float32)
    h_nlos = tf.complex(h_nlos_real, h_nlos_imag)

    phi = sionna.phy.config.tf_rng.uniform(
        [batch_size, repeticiones],
        minval=0.0,
        maxval=tf.constant(2.0 * np.pi, tf.float32)
    )
    h_los = tf.complex(tf.cos(phi), tf.sin(phi))

    a = tf.sqrt(k_factor_lin / (k_factor_lin + 1.0))
    b = tf.sqrt(1.0 / (k_factor_lin + 1.0))
    h_bloque = (
        tf.cast(a, tf.complex64) * tf.cast(h_los, tf.complex64)
        + tf.cast(b, tf.complex64) * tf.cast(h_nlos, tf.complex64)
    )
    h = tf.repeat(h_bloque, repeats=bloq_coher, axis=1)[:, :N]
    y = h * x + awgn
    return y, h, no


# Canal lognormal: aplica desvanecimiento multiplicativo por sombreado.
@tf.function(jit_compile=True)
def canalLognormal(bloq_coher, sigma_db, ebno_db, bitsxsimb, coderate, x):
    batch_size = tf.shape(x)[0]
    bloq_coher = tf.cast(bloq_coher, tf.int32)
    N = tf.shape(x)[-1]
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    repeticiones = _fading_repetitions(N, bloq_coher)
    awgn = _sample_complex_awgn(tf.shape(x), no)

    sigma_lin = tf.cast(tf.math.log(10.0) / 20.0 * sigma_db, tf.float32)
    mu_ln = -tf.square(sigma_lin)
    z = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], mean=mu_ln, stddev=sigma_lin, dtype=tf.float32)
    a = tf.repeat(tf.exp(z), repeats=bloq_coher, axis=1)[:, :N]
    h = tf.complex(a, tf.zeros_like(a))
    y = h * x + awgn
    return y, h, no


# Canal Suzuki: combina desvanecimiento Rayleigh y sombreado lognormal.
@tf.function(jit_compile=True)
def canalRayleighLognormal(bloq_coher, sigma_db, ebno_db, bitsxsimb, coderate, x):
    batch_size = tf.shape(x)[0]
    bloq_coher = tf.cast(bloq_coher, tf.int32)
    N = tf.shape(x)[-1]
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    repeticiones = _fading_repetitions(N, bloq_coher)
    awgn = _sample_complex_awgn(tf.shape(x), no)

    g_real = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], stddev=tf.sqrt(0.5), dtype=tf.float32)
    g_imag = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], stddev=tf.sqrt(0.5), dtype=tf.float32)
    g = tf.complex(g_real, g_imag)

    sigma_lin = tf.cast(tf.math.log(10.0) / 20.0 * sigma_db, tf.float32)
    mu_ln = -tf.square(sigma_lin)
    z = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], mean=mu_ln, stddev=sigma_lin, dtype=tf.float32)
    a_c = tf.complex(tf.exp(z), tf.zeros_like(z))

    h = tf.repeat(g * a_c, repeats=bloq_coher, axis=1)[:, :N]
    y = h * x + awgn
    return y, h, no


# Genera muestras Gamma(m, m) de media unitaria compatibles con XLA (Marsaglia-Tsang).
def _sample_gamma_unit_mean(shape, m_param):
    m_tf = tf.cast(m_param, tf.float32)
    alpha = tf.where(m_tf < 1.0, m_tf + 1.0, m_tf)
    d = alpha - (1.0 / 3.0)
    c = tf.math.rsqrt(9.0 * d)

    accepted = tf.zeros(shape, dtype=tf.bool)
    samples = tf.ones(shape, dtype=tf.float32)

    def cond(acc, _samp):
        return tf.reduce_any(tf.logical_not(acc))

    def body(acc, samp):
        z = sionna.phy.config.tf_rng.normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32)
        u = sionna.phy.config.tf_rng.uniform(shape, minval=1e-7, maxval=1.0, dtype=tf.float32)
        v_base = 1.0 + c * z
        v = v_base * v_base * v_base
        v_safe = tf.maximum(v, 1e-7)
        accept_cond = tf.logical_and(
            v_base > 0.0,
            tf.math.log(u) < (0.5 * z * z + d * (1.0 - v_safe + tf.math.log(v_safe)))
        )
        newly_acc = tf.logical_and(tf.logical_not(acc), accept_cond)
        return tf.logical_or(acc, newly_acc), tf.where(newly_acc, d * v_safe, samp)

    _, samples = tf.while_loop(cond, body, [accepted, samples])

    # Aplica reducción exacta Gamma(m) = Gamma(m + 1) * U^(1/m) cuando 0.5 <= m < 1.
    u_boost = sionna.phy.config.tf_rng.uniform(shape, minval=1e-7, maxval=1.0, dtype=tf.float32)
    samples = tf.where(m_tf < 1.0, samples * tf.pow(u_boost, 1.0 / m_tf), samples)
    return samples / m_tf


# Canal Nakagami-m: envolvente Nakagami normalizada a potencia unitaria (m >= 0.5).
@tf.function(jit_compile=True)
def canalNakagami(bloq_coher, m_nakagami, ebno_db, bitsxsimb, coderate, x):
    batch_size = tf.shape(x)[0]
    bloq_coher = tf.cast(bloq_coher, tf.int32)
    N = tf.shape(x)[-1]
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    repeticiones = _fading_repetitions(N, bloq_coher)
    awgn = _sample_complex_awgn(tf.shape(x), no)

    potencia = _sample_gamma_unit_mean([batch_size, repeticiones], m_nakagami)
    amp = tf.sqrt(tf.maximum(potencia, 1e-12))
    phi = sionna.phy.config.tf_rng.uniform(
        [batch_size, repeticiones],
        minval=0.0,
        maxval=tf.constant(2.0 * np.pi, tf.float32)
    )
    h_bloque = tf.complex(amp * tf.cos(phi), amp * tf.sin(phi))
    h = tf.repeat(h_bloque, repeats=bloq_coher, axis=1)[:, :N]
    y = h * x + awgn
    return y, h, no


# Canal Shadowed-Rician (Abdi / satelital LMS): LOS con sombreado Nakagami-m + NLOS Rayleigh.
@tf.function(jit_compile=True)
def canalShadowedRician(bloq_coher, k_factor_db, m_nakagami, ebno_db, bitsxsimb, coderate, x):
    batch_size = tf.shape(x)[0]
    bloq_coher = tf.cast(bloq_coher, tf.int32)
    N = tf.shape(x)[-1]
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    repeticiones = _fading_repetitions(N, bloq_coher)
    awgn = _sample_complex_awgn(tf.shape(x), no)

    k_lin = tf.pow(10.0, tf.cast(k_factor_db, tf.float32) / 10.0)
    p_los = k_lin / (k_lin + 1.0)
    p_nlos = 1.0 / (k_lin + 1.0)

    sombreado_los = _sample_gamma_unit_mean([batch_size, repeticiones], m_nakagami)
    amp_los = tf.sqrt(tf.maximum(p_los * sombreado_los, 1e-12))
    phi = sionna.phy.config.tf_rng.uniform(
        [batch_size, repeticiones],
        minval=0.0,
        maxval=tf.constant(2.0 * np.pi, tf.float32)
    )
    h_los = tf.complex(amp_los * tf.cos(phi), amp_los * tf.sin(phi))

    g_real = sionna.phy.config.tf_rng.normal(
        [batch_size, repeticiones], stddev=tf.sqrt(0.5 * p_nlos), dtype=tf.float32
    )
    g_imag = sionna.phy.config.tf_rng.normal(
        [batch_size, repeticiones], stddev=tf.sqrt(0.5 * p_nlos), dtype=tf.float32
    )
    h_nlos = tf.complex(g_real, g_imag)

    h = tf.repeat(h_los + h_nlos, repeats=bloq_coher, axis=1)[:, :N]
    y = h * x + awgn
    return y, h, no


# Canal AWGN con ruido impulsivo (mezcla gaussiana Bernoulli-Gaussian de potencia total N0).
@tf.function(jit_compile=True)
def canalImpulsivo(imp_prob, imp_gamma, ebno_db, bitsxsimb, coderate, x):
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    p_tf = tf.cast(imp_prob, tf.float32)
    gamma_tf = tf.cast(imp_gamma, tf.float32)

    # Normaliza varianza de fondo para que la potencia media total de ruido sea exactamente no.
    no_bg = no / (1.0 + p_tf * gamma_tf)
    ruido_bg = _sample_complex_awgn(tf.shape(x), no_bg)
    ruido_imp = _sample_complex_awgn(tf.shape(x), gamma_tf * no_bg)

    u = sionna.phy.config.tf_rng.uniform(tf.shape(x), minval=0.0, maxval=1.0, dtype=tf.float32)
    mascara = tf.cast(u < p_tf, tf.complex64)
    h = tf.complex(1.0, 0.0)
    y = x + ruido_bg + mascara * ruido_imp
    return y, h, no


# Sin ecualización: conserva la señal recibida para canales AWGN.
@tf.function(jit_compile=True)
def sin_ecualizar(y, _h, no):
    return y, no


# Ecualizador ZF: cancela la ganancia del canal.
@tf.function(jit_compile=True)
def zf(y, h, no, eps=1e-12):
    h2 = tf.maximum(tf.math.square(tf.abs(h)), eps)
    x_hat = y * tf.math.conj(h) / tf.cast(h2, tf.complex64)
    no_eff = no / h2
    return x_hat, no_eff


# Ecualizador MMSE: equilibra reducción de interferencia y amplificación de ruido.
@tf.function(jit_compile=True)
def mmse(y, h, no, eps=1e-12):
    g = tf.math.conj(h) / tf.cast(tf.math.square(tf.abs(h)) + no, tf.complex64)
    d = g * h
    d_abs = tf.maximum(tf.abs(d), eps)
    x_hat = g * y / tf.cast(d_abs, tf.complex64)
    no_eff = no / tf.maximum(tf.math.square(tf.abs(h)), eps)
    return x_hat, no_eff


# Aplica el modelo de canal, ecualización, demapeo con recorte LLR y desintercalado.
def apply_channel_and_equalizer(x, ebno_db, coderate, params, common_blocks):
    c = params['canal']
    if c == "awgn":
        y, h, no = canalAWGN(ebno_db, params['bitsxsimb'], coderate, x)
    elif c == "rayleigh":
        y, h, no = canalRayleigh(params['bloq_coher'], ebno_db, params['bitsxsimb'], coderate, x)
    elif c == "rice":
        y, h, no = canalRice(params['bloq_coher'], params['k_factor_db'], ebno_db, params['bitsxsimb'], coderate, x)
    elif c == "lognormal":
        y, h, no = canalLognormal(params['bloq_coher'], params['sigma_db'], ebno_db, params['bitsxsimb'], coderate, x)
    elif c == "suzuki":
        y, h, no = canalRayleighLognormal(
            params['bloq_coher'], params['sigma_db'], ebno_db, params['bitsxsimb'], coderate, x
        )
    elif c == "nakagami":
        y, h, no = canalNakagami(
            params['bloq_coher'], params['m_nakagami'], ebno_db, params['bitsxsimb'], coderate, x
        )
    elif c == "shadowed_rician":
        y, h, no = canalShadowedRician(
            params['bloq_coher'], params['k_factor_db'], params['m_nakagami'],
            ebno_db, params['bitsxsimb'], coderate, x
        )
    elif c == "impulsive":
        y, h, no = canalImpulsivo(
            params['imp_prob'], params['imp_gamma'], ebno_db, params['bitsxsimb'], coderate, x
        )
    else:
        raise ValueError(f"Canal inválido: {c}")

    eq = params['ecualizador']
    if eq == "no_eq":
        x_hat, no_eff = sin_ecualizar(y, h, no)
    elif eq == "zf":
        x_hat, no_eff = zf(y, h, no)
    elif eq == "mmse":
        x_hat, no_eff = mmse(y, h, no)
    else:
        raise ValueError(f"Ecualizador inválido: {eq}")

    llr_prev = tf.clip_by_value(
        common_blocks['demapeador'](x_hat, no_eff),
        -params['LLR_clip'],
        params['LLR_clip']
    )
    return common_blocks['deinterleaver'](llr_prev)


# =====================================================================
# 2. BLOQUES DEL SISTEMA SIONNA
# =====================================================================

# Base común para canal, ecualización y desintercalado de todos los sistemas.
class BaseSystemBlock(sionna.phy.Block):
    def __init__(self, params, common_blocks):
        super().__init__()
        self.p = params
        self.cb = common_blocks

    def apply_channel_and_eq(self, x, ebno_db, coderate):
        return apply_channel_and_equalizer(x, ebno_db, coderate, self.p, self.cb)


# Sistema de referencia sin código de canal.
class SinCodigo(BaseSystemBlock):
    def call(self, batch_size, ebno_db):
        b = self.cb['datos']([batch_size, self.p['K']])
        x = self.cb['mapeador'](self.cb['interleaver'](b))
        llr = self.apply_channel_and_eq(x, ebno_db, coderate=1.0)
        return b, llr


# Sistema con código Polar y decodificación SCL.
class CodigoPolar(BaseSystemBlock):
    def __init__(self, params, common_blocks, bits_congelados):
        super().__init__(params, common_blocks)
        self.polar_codif = sionna.phy.fec.polar.encoding.PolarEncoder(bits_congelados, self.p['N'])
        self.polar_decodif = sionna.phy.fec.polar.decoding.PolarSCLDecoder(
            bits_congelados, self.p['N'], list_size=self.p['list_scl']
        )

    def call(self, batch_size, ebno_db):
        b = self.cb['datos']([batch_size, self.p['K']])
        x = self.cb['mapeador'](self.cb['interleaver'](self.polar_codif(b)))
        llr = self.apply_channel_and_eq(x, ebno_db, coderate=self.p['K'] / self.p['N'])
        return b, self.polar_decodif(llr)


# Sistema con código Polar y verificación CRC para SCL.
class CodigoPolarCRC(BaseSystemBlock):
    def __init__(self, params, common_blocks, bits_congelados):
        super().__init__(params, common_blocks)
        self.polar_codif = sionna.phy.fec.polar.encoding.PolarEncoder(bits_congelados, self.p['N'])
        self.polar_decodif = sionna.phy.fec.polar.decoding.PolarSCLDecoder(
            bits_congelados, self.p['N'], list_size=self.p['list_scl'], crc_degree=self.p['crc_degree']
        )

    def call(self, batch_size, ebno_db):
        K_efec = self.p['K'] - self.cb['crc_long']
        b = self.cb['datos']([batch_size, K_efec])
        bits_codif = self.polar_codif(self.cb['crc_codif'](b))
        x = self.cb['mapeador'](self.cb['interleaver'](bits_codif))
        llr = self.apply_channel_and_eq(x, ebno_db, coderate=self.p['K'] / self.p['N'])
        b_hat = self.polar_decodif(llr)[..., :K_efec]
        return b, b_hat


# Sistema con código LDPC 5G NR y decodificación belief propagation.
class CodigoLDPC(BaseSystemBlock):
    def __init__(self, params, common_blocks):
        super().__init__(params, common_blocks)
        self.ldpc_codif = sionna.phy.fec.ldpc.encoding.LDPC5GEncoder(
            self.p['K'], self.p['N'], self.p['bitsxsimb']
        )
        self.ldpc_decodif = sionna.phy.fec.ldpc.decoding.LDPC5GDecoder(self.ldpc_codif)

    def call(self, batch_size, ebno_db):
        b = self.cb['datos']([batch_size, self.p['K']])
        x = self.cb['mapeador'](self.cb['interleaver'](self.ldpc_codif(b)))
        llr = self.apply_channel_and_eq(x, ebno_db, coderate=self.p['K'] / self.p['N'])
        return b, self.ldpc_decodif(llr)


# Crea los bloques compartidos de datos, mapeo, intercalado y CRC.
def build_common_blocks(p):
    sionna.phy.config.seed = p['seed']
    modulacion = "pam" if p['bitsxsimb'] == 1 else "qam"
    datos = sionna.phy.mapping.BinarySource()
    constelacion = sionna.phy.mapping.Constellation(modulacion, p['bitsxsimb'])
    mapeador = sionna.phy.mapping.Mapper(constellation=constelacion)
    demapeador = sionna.phy.mapping.Demapper("app", constellation=constelacion)
    interleaver = sionna.phy.fec.interleaving.RandomInterleaver(seed=p['seed'], axis=-1)
    deinterleaver = sionna.phy.fec.interleaving.Deinterleaver(interleaver)
    crc_codif = sionna.phy.fec.crc.CRCEncoder(crc_degree=p['crc_degree'])
    return {
        'datos': datos,
        'constelacion': constelacion,
        'mapeador': mapeador,
        'demapeador': demapeador,
        'interleaver': interleaver,
        'deinterleaver': deinterleaver,
        'crc_codif': crc_codif,
        'crc_long': crc_codif.crc_length,
    }


# =====================================================================
# 3. HILOS DE TRABAJO EN SEGUNDO PLANO
# =====================================================================

# Mide la confiabilidad de cada canal de bit Polar mediante Monte Carlo.
class MonteCarloWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, float, float)  # actual, total, transcurrido, eta
    finished_signal = pyqtSignal(np.ndarray, str)
    stopped_signal = pyqtSignal(int, int, str)
    error_signal = pyqtSignal(str, str)  # mensaje_corto, detalle_traceback

    def __init__(self, params, resume_backup=False):
        super().__init__()
        self.p = dict(params)
        self.resume_backup = resume_backup
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def _init_ber_array(self, backup_file, N):
        ber_bits = np.full(N, np.nan, dtype=np.float64)
        start_index = 0
        if self.resume_backup and os.path.exists(backup_file):
            loaded_bits, saved_index, _ = load_mc_backup(backup_file)
            if len(loaded_bits) == N and 0 <= saved_index <= N:
                ber_bits = loaded_bits
                start_index = saved_index
                if start_index < N:
                    self.log_signal.emit(f"Respaldo recuperado. Reanudando desde bit-canal {start_index + 1}/{N}")
                else:
                    self.log_signal.emit(f"Respaldo completo recuperado ({N}/{N}). Finalizando exportación...")
                self.progress_signal.emit(start_index, N, 0.0, 0.0)
        elif not self.resume_backup and os.path.exists(backup_file):
            os.remove(backup_file)
        return ber_bits, start_index

    def _simulate_bit_channel(self, bit_idx, N, cb, snr_sim, coderate_sim):
        lote_mc = self.p['lote_mc']
        num_iter = self.p['num_iter']
        bits_congelados = np.array([j for j in range(N) if j != bit_idx])
        codificador_temp = sionna.phy.fec.polar.encoding.PolarEncoder(bits_congelados, N)
        decodificador_temp = sionna.phy.fec.polar.decoding.PolarSCLDecoder(bits_congelados, N, list_size=1)

        # Evita recompilación XLA anidada por cada canal de bit para prevenir fugas de RAM.
        @tf.function
        def simulacion_batch_size():
            bits = cb['datos']([lote_mc, 1])
            x = cb['mapeador'](cb['interleaver'](codificador_temp(bits)))
            llr = apply_channel_and_equalizer(x, snr_sim, coderate_sim, self.p, cb)
            bits_decodif = decodificador_temp(llr)
            err = tf.math.count_nonzero(tf.not_equal(bits, bits_decodif), dtype=tf.int32)
            return tf.cast(err, tf.int64)

        errores_totales = 0
        stopped = False
        try:
            iter_restantes = num_iter
            while iter_restantes > 0:
                paso = min(MC_CHUNK_SYNC, iter_restantes)
                err_bloque = tf.constant(0, dtype=tf.int64)
                for _ in range(paso):
                    if self._stop_requested:
                        stopped = True
                        break
                    err_bloque = err_bloque + simulacion_batch_size()
                if stopped:
                    break
                errores_totales += int(err_bloque.numpy())
                iter_restantes -= paso
        finally:
            del simulacion_batch_size, codificador_temp, decodificador_temp
            gc.collect()

        if stopped:
            return None
        return errores_totales / (lote_mc * num_iter)

    def run(self):
        try:
            cb = build_common_blocks(self.p)
            N = self.p['N']
            snr_sim = tf.constant(self.p['snr_simulacion'], dtype=tf.float32)
            coderate_sim = float(self.p['K'] / N)
            backup_file = f"BER_bits_backup_{N}.npz"

            BER_bits, start_index = self._init_ber_array(backup_file, N)
            run_start_time = time.monotonic()

            for i in range(start_index, N):
                if self._stop_requested:
                    save_mc_backup(backup_file, BER_bits, i, self.p)
                    self.log_signal.emit(
                        f"⚠️ Simulación detenida por el usuario antes del bit-canal {i + 1}/{N}. Respaldo guardado."
                    )
                    self.stopped_signal.emit(i, N, backup_file)
                    return

                ber_val = self._simulate_bit_channel(i, N, cb, snr_sim, coderate_sim)
                if ber_val is None:
                    save_mc_backup(backup_file, BER_bits, i, self.p)
                    self.log_signal.emit(
                        f"⚠️ Simulación detenida por el usuario durante el bit-canal {i + 1}/{N}. Respaldo guardado."
                    )
                    self.stopped_signal.emit(i, N, backup_file)
                    return

                BER_bits[i] = ber_val
                completed_now = (i - start_index) + 1
                elapsed = time.monotonic() - run_start_time
                eta = (elapsed / completed_now) * (N - (i + 1))

                self.progress_signal.emit(i + 1, N, elapsed, eta)
                self.log_signal.emit(f"Bit-canal {i + 1}/{N} | BER: {BER_bits[i]:.2e}")

                if (i + 1) % MC_BACKUP_INTERVAL == 0:
                    save_mc_backup(backup_file, BER_bits, i + 1, self.p)

                mem = psutil.virtual_memory()
                if mem.percent > MC_RAM_LIMIT_PERCENT:
                    save_mc_backup(backup_file, BER_bits, i + 1, self.p)
                    self.error_signal.emit(
                        f"El uso de RAM superó el {MC_RAM_LIMIT_PERCENT:.0f}% ({mem.percent:.1f}%). "
                        f"Respaldo de emergencia guardado en bit-canal {i + 1}/{N}.",
                        ""
                    )
                    return

            if not np.all(np.isfinite(BER_bits)):
                raise RuntimeError("Monte Carlo finalizó con valores BER incompletos o no finitos.")

            filename = default_ber_bits_csv_path(self.p, N)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            pd.DataFrame({"BER bits": BER_bits}).to_csv(filename, index=True, encoding="utf-8-sig")

            if os.path.exists(backup_file):
                os.remove(backup_file)

            self.finished_signal.emit(BER_bits, filename)

        except Exception as e:
            self.error_signal.emit(f"{type(e).__name__}: {e}", traceback.format_exc())


# Ejecuta las barridas BER frente a Eb/N0 para los distintos sistemas.
class BERSimWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, str)
    curve_ready_signal = pyqtSignal(str, np.ndarray, np.ndarray)
    finished_signal = pyqtSignal()
    stopped_signal = pyqtSignal()
    error_signal = pyqtSignal(str, str)  # mensaje_corto, detalle_traceback

    def __init__(self, params, ber_bits=None):
        super().__init__()
        self.p = dict(params)
        self.ber_bits = None if ber_bits is None else np.array(ber_bits, copy=True)
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def _run_curve(self, ber_plots, model, ebno_dbs, label, step, total_steps,
                   target_errors, max_iter, soft_estimates=False, log_name=None):
        if self._stop_requested:
            self.stopped_signal.emit()
            return None
        display_name = log_name or label
        self.progress_signal.emit(step, total_steps, f"Simulando: {display_name}...")
        self.log_signal.emit(f"Ejecutando simulación BER: {display_name}...")
        ber_vals, _ = ber_plots.simulate(
            model,
            ebno_dbs=ebno_dbs,
            batch_size=self.p['lote_berplot'],
            num_target_block_errors=target_errors,
            legend=label,
            soft_estimates=soft_estimates,
            max_mc_iter=max_iter,
            show_fig=False,
            forward_keyboard_interrupt=False
        )
        next_step = step + 1
        self.curve_ready_signal.emit(label, ebno_dbs, np.array(ber_vals))
        self.progress_signal.emit(next_step, total_steps, f"Completado: {label}")
        return next_step

    def run(self):
        try:
            cb = build_common_blocks(self.p)
            ebno_dbs = np.linspace(self.p['ebno_min'], self.p['ebno_max'], self.p['ebno_rango'])
            ber_plots = sionna.phy.utils.PlotBER(f"Comparación canal {self.p['canal']}")

            has_valid_frozen_bits = (
                self.ber_bits is not None
                and len(self.ber_bits) == self.p['N']
                and np.all(np.isfinite(self.ber_bits))
            )
            run_polar_crc = has_valid_frozen_bits and (self.p['K'] > cb['crc_long'])
            total_steps = 2 + (1 if has_valid_frozen_bits else 0) + (1 if run_polar_crc else 0)
            step = 0

            # 1. Sistema sin código de canal
            step = self._run_curve(
                ber_plots, SinCodigo(self.p, cb), ebno_dbs, CURVE_UNCODED,
                step, total_steps, target_errors=100, max_iter=100, soft_estimates=True
            )
            if step is None:
                return

            # 2 y 3. Sistemas Polar y Polar + CRC (si existen bits congelados válidos)
            if has_valid_frozen_bits:
                indices_ordenados = np.argsort(self.ber_bits)
                bits_congelados = indices_ordenados[-(self.p['N'] - self.p['K']):]

                step = self._run_curve(
                    ber_plots, CodigoPolar(self.p, cb, bits_congelados), ebno_dbs, CURVE_POLAR,
                    step, total_steps, target_errors=250, max_iter=25, log_name=f"{CURVE_POLAR} (SCL)"
                )
                if step is None:
                    return

                if run_polar_crc:
                    crc_label = f"{CURVE_POLAR} + {self.p['crc_degree']}"
                    step = self._run_curve(
                        ber_plots, CodigoPolarCRC(self.p, cb, bits_congelados), ebno_dbs, crc_label,
                        step, total_steps, target_errors=250, max_iter=25
                    )
                    if step is None:
                        return
                else:
                    self.log_signal.emit(
                        f"⚠️ Omitiendo Polar + {self.p['crc_degree']}: "
                        f"K ({self.p['K']}) debe ser mayor que la longitud del CRC ({cb['crc_long']})."
                    )
            else:
                self.log_signal.emit(
                    "ℹ️ No hay bits congelados compatibles cargados/generados. Se omiten las curvas Polar."
                )

            # 4. Código LDPC 5G NR (aislado para no perder curvas previas si (K, N) está fuera de 5G NR)
            if self._stop_requested:
                self.stopped_signal.emit()
                return
            try:
                codigo_ldpc = CodigoLDPC(self.p, cb)
            except (ValueError, AssertionError) as ldpc_err:
                self.log_signal.emit(
                    f"⚠️ Omitiendo LDPC 5G NR para (K={self.p['K']}, N={self.p['N']}): {ldpc_err}"
                )
                self.progress_signal.emit(step + 1, total_steps, f"Omitido: {CURVE_LDPC}")
            else:
                step = self._run_curve(
                    ber_plots, codigo_ldpc, ebno_dbs, CURVE_LDPC,
                    step, total_steps, target_errors=250, max_iter=25
                )
                if step is None:
                    return

            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(f"{type(e).__name__}: {e}", traceback.format_exc())


# Estima automáticamente el SNR de diseño [dB] mediante bisección de información mutua I(W).
class DesignSNRWorker(QThread):
    progress_signal = pyqtSignal(int, int, float, float)  # paso, total, snr_db, mi
    finished_signal = pyqtSignal(float, float, float, float)  # snr_db, mi_lograda, mi_objetivo, ber_est
    error_signal = pyqtSignal(str, str)  # mensaje_corto, detalle_traceback

    def __init__(self, params, target_alpha=AUTO_SNR_TARGET_ALPHA, num_steps=AUTO_SNR_BISECTION_STEPS):
        super().__init__()
        self.p = dict(params)
        self.target_alpha = float(target_alpha)
        self.num_steps = int(num_steps)
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            cb = build_common_blocks(self.p)
            N = self.p['N']
            coderate = float(self.p['K'] / N)
            target_mi = float(np.clip(self.target_alpha * coderate, 0.02, 0.98))

            # Usa bloques de longitud N para respetar bloq_coher e interleaver sin decodificador Polar.
            batch_size = max(64, min(2048, 65536 // N))
            num_batches = 4

            @tf.function
            def eval_batch(ebno_db_tf):
                b = cb['datos']([batch_size, N])
                x = cb['mapeador'](cb['interleaver'](b))
                llr = apply_channel_and_equalizer(x, ebno_db_tf, coderate, self.p, cb)
                # En Sionna, LLR = ln(P(b=1|y)/P(b=0|y)), por lo que (2b - 1)*LLR > 0 en aciertos.
                aligned_llr = (2.0 * tf.cast(b, tf.float32) - 1.0) * tf.cast(llr, tf.float32)
                mi = 1.0 - tf.reduce_mean(tf.math.softplus(-aligned_llr) / tf.math.log(2.0))
                b_hat = tf.cast(llr > 0.0, b.dtype)
                ber = tf.reduce_mean(tf.cast(tf.not_equal(b, b_hat), tf.float32))
                return mi, ber

            lo, hi = -20.0, 40.0
            best_snr, best_mi, best_ber = 0.0, 0.0, 0.5

            for step in range(1, self.num_steps + 1):
                if self._stop_requested:
                    return
                mid = 0.5 * (lo + hi)
                ebno_tf = tf.constant(mid, dtype=tf.float32)
                mi_sum = 0.0
                ber_sum = 0.0
                for _ in range(num_batches):
                    mi_t, ber_t = eval_batch(ebno_tf)
                    mi_sum += float(mi_t.numpy())
                    ber_sum += float(ber_t.numpy())

                mi_val = float(np.clip(mi_sum / num_batches, 0.0, 1.0))
                ber_val = ber_sum / num_batches
                best_snr, best_mi, best_ber = mid, mi_val, ber_val

                self.progress_signal.emit(step, self.num_steps, mid, mi_val)
                if mi_val < target_mi:
                    lo = mid
                else:
                    hi = mid

            self.finished_signal.emit(round(best_snr, 2), best_mi, target_mi, best_ber)

        except Exception as e:
            self.error_signal.emit(f"{type(e).__name__}: {e}", traceback.format_exc())


# =====================================================================
# 4. VENTANA PRINCIPAL PYQT6
# =====================================================================

# Ventana principal con controles, progreso, registros y gráfica BER.
class PolarSimGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador Sionna de Códigos Polar/LDPC y Diseñador Monte Carlo")
        self.resize(1380, 880)

        self.ber_bits = None
        self.ber_bits_source = ""
        self.mc_worker = None
        self.ber_worker = None
        self.snr_worker = None
        self.active_mc_params = None
        self.active_ber_params = None
        self.ber_curves_ebno = None
        self.ber_curves_data = {}
        self.plot_min_ber = BER_MIN_YLIM

        self.init_ui()
        self.init_status_bar()
        self.sync_initial_ui_state()

    # =================================================================
    # CONSTRUCCIÓN DE LA INTERFAZ
    # =================================================================

    # Agrega una fila de formulario asignando el mismo tooltip a la etiqueta y al control.
    @staticmethod
    def _add_row_with_tip(form_layout, label_text, widget, tooltip=None):
        label = QLabel(label_text)
        if tooltip:
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
        form_layout.addRow(label, widget)

    # Crea un botón destacado con estilo consistente para estados activo y deshabilitado.
    @staticmethod
    def _styled_button(text, bg_color, callback):
        btn = QPushButton(text)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {bg_color}; color: white; font-weight: bold; padding: 8px; }}"
            "QPushButton:disabled { background-color: #9aa5b1; color: #eceff1; }"
        )
        btn.clicked.connect(callback)
        return btn

    def init_ui(self):
        # Divide la interfaz entre parámetros de entrada y visualización de resultados.
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.grp_basic = self._build_basic_group()
        self.grp_chan = self._build_channel_group()
        self.grp_rx = self._build_receiver_group()
        self.grp_mc = self._build_mc_group()
        self.grp_ber = self._build_ber_group()
        self.param_groups = [
            self.grp_basic, self.grp_chan, self.grp_rx, self.grp_mc, self.grp_ber
        ]

        for grp in self.param_groups:
            left_layout.addWidget(grp)
        left_layout.addLayout(self._build_action_buttons())
        left_layout.addStretch()

        scroll_area.setWidget(left_widget)
        splitter.addWidget(scroll_area)
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([450, 930])

    # Grupo 1: parámetros básicos y configuración del código.
    def _build_basic_group(self):
        grp = QGroupBox("1. Parámetros Básicos y del Código")
        form = QFormLayout(grp)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(0)

        self.combo_N = QComboBox()
        for exp in range(5, 14):  # 32 a 8192
            self.combo_N.addItem(str(2**exp), 2**exp)
        self.combo_N.setCurrentText("1024")
        self.combo_N.currentIndexChanged.connect(self.on_N_changed)

        self.combo_rate = QComboBox()
        for label, rate_val in PREDEFINED_CODE_RATES:
            self.combo_rate.addItem(label, rate_val)
        self.combo_rate.setCurrentText("1/2 (0.500)")
        self.combo_rate.currentIndexChanged.connect(self.update_code_rate_label)

        self.combo_bits = QComboBox()
        self.combo_bits.addItem("1 (BPSK - PAM)", 1)
        self.combo_bits.addItem("2 (QPSK - QAM)", 2)
        self.combo_bits.addItem("4 (16-QAM)", 4)
        self.combo_bits.addItem("6 (64-QAM)", 6)
        self.combo_bits.currentIndexChanged.connect(self.update_code_rate_label)

        self.combo_crc = QComboBox()
        for crc in CRC_LENGTHS:
            self.combo_crc.addItem(crc)
        self.combo_crc.currentTextChanged.connect(self.update_code_rate_label)

        self.lbl_code_rate = QLabel()
        self.lbl_code_rate.setStyleSheet("color: #495057; font-size: 11px;")

        self._add_row_with_tip(
            form, "Semilla aleatoria:", self.spin_seed,
            "Semilla aleatoria para el RNG de Sionna y el interleaver (reproducibilidad).\n"
            "Recomendado: 0 (semilla fija para comparaciones repetibles)."
        )
        self._add_row_with_tip(
            form, "Longitud de código (N):", self.combo_N,
            "Longitud de palabra código Polar N (debe ser potencia de 2).\n"
            "Valores típicos: 128, 256, 512 o 1024 (por defecto: 1024)."
        )
        self._add_row_with_tip(
            form, "Tasa de código (R = K/N):", self.combo_rate,
            "Tasas de código estándar predefinidas (K se calcula automáticamente como N × R).\n"
            "Referencia recomendada: 1/2 (0.500).\n"
            "Nota: LDPC 5G NR soporta tasas aprox. entre 1/5 y 11/12 (K >= 12)."
        )
        self._add_row_with_tip(
            form, "Bits por símbolo:", self.combo_bits,
            "Orden de modulación en bits por símbolo (1=BPSK, 2=QPSK, 4=16-QAM).\n"
            "N (y K para Sin código) deben ser divisibles por este valor.\n"
            "Nota: 64-QAM (6) no es divisible en longitudes Polar potencias de 2."
        )
        self._add_row_with_tip(
            form, "Polinomio CRC:", self.combo_crc,
            "Polinomio CRC 3GPP 5G NR usado por Polar + CRC (CA-SCL).\n"
            "Recomendado: CRC24A (estándar 5G NR de 24 bits; requiere K > longitud CRC)."
        )
        form.addRow("Bits de información (K):", self.lbl_code_rate)
        return grp

    # Grupo 2: parámetros del modelo de canal.
    def _build_channel_group(self):
        grp = QGroupBox("2. Parámetros del Canal")
        form = QFormLayout(grp)

        self.combo_canal = QComboBox()
        for c in (
            "suzuki", "awgn", "rayleigh", "rice", "lognormal",
            "nakagami", "shadowed_rician", "impulsive"
        ):
            self.combo_canal.addItem(c)
        self.combo_canal.currentTextChanged.connect(self.on_channel_changed)

        self.spin_coher = QSpinBox()
        self.spin_coher.setRange(1, 8192)
        self.spin_coher.setValue(64)

        self.chk_flat_fading = QCheckBox("Desvanecimiento plano (bloque = N)")
        self.chk_flat_fading.setChecked(False)
        self.chk_flat_fading.setToolTip(
            "Activa desvanecimiento plano en toda la palabra código (bloque de coherencia = N).\n"
            "• Desmarcado: usa sub-bloques de coherencia (por defecto N // 16, editable).\n"
            "• Marcado: mantiene una única ganancia h constante durante los N símbolos."
        )
        self.chk_flat_fading.toggled.connect(self.on_flat_fading_toggled)

        self.spin_k_db = QDoubleSpinBox()
        self.spin_k_db.setRange(-20.0, 50.0)
        self.spin_k_db.setValue(6.0)

        self.spin_sigma_db = QDoubleSpinBox()
        self.spin_sigma_db.setRange(0.1, 30.0)
        self.spin_sigma_db.setValue(6.0)

        self.spin_m_nakagami = QDoubleSpinBox()
        self.spin_m_nakagami.setRange(0.5, 20.0)
        self.spin_m_nakagami.setSingleStep(0.5)
        self.spin_m_nakagami.setValue(1.5)

        self.spin_imp_prob = QDoubleSpinBox()
        self.spin_imp_prob.setDecimals(3)
        self.spin_imp_prob.setRange(0.001, 0.500)
        self.spin_imp_prob.setSingleStep(0.01)
        self.spin_imp_prob.setValue(0.050)

        self.spin_imp_gamma = QDoubleSpinBox()
        self.spin_imp_gamma.setRange(1.0, 1000.0)
        self.spin_imp_gamma.setSingleStep(5.0)
        self.spin_imp_gamma.setValue(50.0)

        self._add_row_with_tip(
            form, "Modelo de canal:", self.combo_canal,
            "Modelo de canal inalámbrico:\n"
            "• awgn: solo ruido gaussiano aditivo (sin desvanecimiento)\n"
            "• rayleigh: desvanecimiento multitrayecto NLOS\n"
            "• rice: multitrayecto con LOS (usa Factor K)\n"
            "• lognormal: sombreado lento (usa σ)\n"
            "• suzuki: Rayleigh × Lognormal (usa σ)\n"
            "• nakagami: desvanecimiento Nakagami-m (usa parámetro m)\n"
            "• shadowed_rician: satelital LMS/NTN con LOS sombreada (usa K y m)\n"
            "• impulsive: AWGN + ruido impulsivo Bernoulli-Gaussian (usa p y Γ)"
        )
        self._add_row_with_tip(
            form, "Bloque de coherencia:", self.spin_coher,
            "Cantidad de símbolos que comparten la misma ganancia de canal h "
            "(solo canales con desvanecimiento).\n"
            "Recomendado: N // 16 (ej. 64 para N=1024, 32 para N=512) o N para desvanecimiento plano.\n"
            "Valores menores modelan desvanecimiento más rápido; mayores producen ráfagas más largas."
        )
        form.addRow("", self.chk_flat_fading)
        self._add_row_with_tip(
            form, "Factor K de Rice [dB]:", self.spin_k_db,
            "Relación de potencia LOS/NLOS en dB (canales Rice y Shadowed-Rician).\n"
            "Valores razonables típicos:\n"
            "• Entornos suburbanos / satelital AS: 4.0 a 15.0 dB (por defecto: 6.0 dB)\n"
            "• LOS débil / sombreado fuerte: -1 a 3 dB (tiende a Rayleigh si K → -∞)\n"
            "• LOS fuerte: > 10 dB (tiende a AWGN)"
        )
        self._add_row_with_tip(
            form, "σ Lognormal/Suzuki [dB]:", self.spin_sigma_db,
            "Desviación estándar del sombreado σ en dB (solo canales Lognormal y Suzuki).\n"
            "Valores razonables típicos:\n"
            "• Sombreado leve: ~2.0 dB\n"
            "• Sombreado moderado: 4.0 dB\n"
            "• Sombreado severo: 6.0 dB (recomendado por defecto: 6.0 dB)"
        )
        self._add_row_with_tip(
            form, "Parámetro Nakagami m:", self.spin_m_nakagami,
            "Parámetro de forma m >= 0.5 (canales Nakagami-m y Shadowed-Rician).\n"
            "Valores razonables típicos:\n"
            "• m = 0.5: desvanecimiento muy severo (peor que Rayleigh)\n"
            "• m = 1.0: equivale a Rayleigh (en Shadowed-Rician: sombreado fuerte FHS)\n"
            "• m = 1.5–5.0: desvanecimiento moderado / sombreado satelital medio (AS)\n"
            "• m >= 10: sombreado leve (tiende a Rice / AWGN)"
        )
        self._add_row_with_tip(
            form, "Prob. impulso p:", self.spin_imp_prob,
            "Probabilidad por símbolo de sufrir un pulso de ruido (solo canal Impulsive).\n"
            "Valores típicos: 0.01 (1% raro) a 0.10 (10% frecuente); recomendado: 0.05."
        )
        self._add_row_with_tip(
            form, "Razón impulsiva Γ:", self.spin_imp_gamma,
            "Relación de potencia entre ruido impulsivo y ruido de fondo (solo canal Impulsive).\n"
            "La potencia media total se normaliza a N0.\n"
            "Valores típicos: 10 (moderado) a 100 (severo); recomendado: 50."
        )
        return grp

    # Grupo 3: parámetros del receptor.
    def _build_receiver_group(self):
        grp = QGroupBox("3. Parámetros del Receptor")
        form = QFormLayout(grp)

        self.combo_eq = QComboBox()
        for eq in ("mmse", "zf", "no_eq"):
            self.combo_eq.addItem(eq)

        self.combo_scl = QComboBox()
        for list_size in (1, 2, 4, 8, 16, 32):
            self.combo_scl.addItem(str(list_size), list_size)
        self.combo_scl.setCurrentText("8")

        self.spin_llr = QDoubleSpinBox()
        self.spin_llr.setRange(10.0, 5000.0)
        self.spin_llr.setValue(500.0)

        self._add_row_with_tip(
            form, "Ecualizador:", self.combo_eq,
            "Estrategia de ecualización en el receptor:\n"
            "• mmse: equilibra compensación del canal y amplificación de ruido "
            "(recomendado con desvanecimiento)\n"
            "• zf: invierte directamente la ganancia del canal "
            "(amplifica ruido en desvanecimientos profundos)\n"
            "• no_eq: sin ecualización (solo disponible para canales AWGN e Impulsive)"
        )
        self._add_row_with_tip(
            form, "Tamaño de lista SCL:", self.combo_scl,
            "Número de trayectorias candidatas en paralelo para decodificación Polar SCL.\n"
            "Recomendado: 8 (buen equilibrio BER/velocidad; 1 = SC simple, 16–32 = más lento)."
        )
        self._add_row_with_tip(
            form, "Recorte de LLR:", self.spin_llr,
            "Límite simétrico [-clip, +clip] aplicado a los LLR del demapeador\n"
            "para evitar desbordamiento o inestabilidad numérica.\n"
            "Recomendado: 500.0."
        )
        return grp

    # Grupo 4: generación Monte Carlo de bits congelados.
    def _build_mc_group(self):
        grp = QGroupBox("4. Diseño Polar por Monte Carlo")
        form = QFormLayout(grp)

        self.spin_lote_mc = QSpinBox()
        self.spin_lote_mc.setRange(10, 50000)
        self.spin_lote_mc.setSingleStep(500)
        self.spin_lote_mc.setValue(2000)
        self.spin_lote_mc.valueChanged.connect(self.update_mc_summary_label)

        self.spin_num_iter = QSpinBox()
        self.spin_num_iter.setRange(1, 50000)
        self.spin_num_iter.setSingleStep(500)
        self.spin_num_iter.setValue(2000)
        self.spin_num_iter.valueChanged.connect(self.update_mc_summary_label)

        self.spin_snr_sim = QDoubleSpinBox()
        self.spin_snr_sim.setRange(-20.0, 40.0)
        self.spin_snr_sim.setSingleStep(0.5)
        self.spin_snr_sim.setValue(5.0)

        self.btn_auto_snr = QPushButton("⚡ Auto")
        self.btn_auto_snr.setFixedWidth(72)
        self.btn_auto_snr.setToolTip(
            "Calcula automáticamente el SNR de diseño Monte Carlo [dB] mediante bisección "
            "de información mutua (I(W) ≈ 0.88 × R).\n"
            "Se adapta a tasa de código, modulación, canal, desvanecimiento, ecualizador y recorte LLR."
        )
        self.btn_auto_snr.clicked.connect(self.start_auto_design_snr)

        snr_row_widget = QWidget()
        snr_row_layout = QHBoxLayout(snr_row_widget)
        snr_row_layout.setContentsMargins(0, 0, 0, 0)
        snr_row_layout.setSpacing(6)
        snr_row_layout.addWidget(self.spin_snr_sim, stretch=1)
        snr_row_layout.addWidget(self.btn_auto_snr)

        self.lbl_mc_total_bits = QLabel()
        self.lbl_mc_total_bits.setStyleSheet("color: #495057; font-size: 11px;")

        self.lbl_mc_status = QLabel("Sin cargar")
        self.lbl_mc_status.setStyleSheet("color: #d9534f; font-weight: bold;")

        self._add_row_with_tip(
            form, "Lote en paralelo:", self.spin_lote_mc,
            "Muestras procesadas en paralelo por cada paso Monte Carlo (consume RAM/VRAM).\n"
            "Recomendado: 2000 (reducir si el uso de memoria es elevado)."
        )
        self._add_row_with_tip(
            form, "Iteraciones secuenciales:", self.spin_num_iter,
            "Número de lotes secuenciales acumulados por bit-canal (afecta el tiempo de ejecución).\n"
            "Recomendado: 2000 (2000 × 2000 = 4,000,000 bits/bit-canal)."
        )
        snr_tooltip = (
            "Eb/N0 de diseño Monte Carlo [dB] para separar bit-canales ruidosos (congelados) "
            "de los confiables.\n"
            "Pulse '⚡ Auto' para estimarlo según su configuración, o use los valores guía (R = 0.5):\n"
            "• AWGN: -1.2 dB | Impulsive (p=0.05, Γ=50): -1.9 dB\n"
            "• Rayleigh: 1.0 dB | Nakagami (m=1.5): 0.3 dB\n"
            "• Rice (K=6 dB): -0.3 dB | Shadowed-Rician (K=6, m=1.5): 0.5 dB\n"
            "• Lognormal (σ=6 dB): 2.4 dB | Suzuki (σ=6 dB): 5.0 dB"
        )
        lbl_snr_sim = QLabel("SNR de diseño [dB]:")
        lbl_snr_sim.setToolTip(snr_tooltip)
        self.spin_snr_sim.setToolTip(snr_tooltip)
        form.addRow(lbl_snr_sim, snr_row_widget)
        form.addRow("Muestras totales:", self.lbl_mc_total_bits)
        form.addRow("Bits congelados:", self.lbl_mc_status)
        return grp

    # Grupo 5: parámetros de la gráfica BER vs Eb/N0.
    def _build_ber_group(self):
        grp = QGroupBox("5. Parámetros de Gráfica BER vs Eb/N0")
        form = QFormLayout(grp)

        self.spin_ebno_min = QDoubleSpinBox()
        self.spin_ebno_min.setRange(-20.0, 50.0)
        self.spin_ebno_min.setValue(0.0)

        self.spin_ebno_max = QDoubleSpinBox()
        self.spin_ebno_max.setRange(-20.0, 50.0)
        self.spin_ebno_max.setValue(20.0)

        self.spin_ebno_rango = QSpinBox()
        self.spin_ebno_rango.setRange(2, 100)
        self.spin_ebno_rango.setValue(21)

        self.spin_lote_ber = QSpinBox()
        self.spin_lote_ber.setRange(10, 10000)
        self.spin_lote_ber.setValue(200)

        self._add_row_with_tip(
            form, "Eb/N0 mín [dB]:", self.spin_ebno_min,
            "Inicio del barrido de Eb/N0 en dB.\n"
            "Valores típicos: -2.0 a 0.0 dB (por defecto: 0.0 dB)."
        )
        self._add_row_with_tip(
            form, "Eb/N0 máx [dB]:", self.spin_ebno_max,
            "Fin del barrido de Eb/N0 en dB.\n"
            "Valores típicos: 6–10 dB para AWGN, 15–25 dB para canales con desvanecimiento "
            "(por defecto: 20.0 dB)."
        )
        self._add_row_with_tip(
            form, "Número de puntos:", self.spin_ebno_rango,
            "Cantidad de puntos de Eb/N0 equidistantes evaluados en el barrido.\n"
            "Recomendado: 21 (pasos de 1 dB entre 0 y 20 dB)."
        )
        self._add_row_with_tip(
            form, "Tamaño de lote BER:", self.spin_lote_ber,
            "Palabras código simuladas por lote al evaluar las curvas BER.\n"
            "Recomendado: 200 (aumentar para acelerar en GPU si la memoria lo permite)."
        )
        return grp

    # Botones de ejecución, detención y exportación/carga CSV.
    def _build_action_buttons(self):
        btn_layout = QVBoxLayout()
        self.btn_run_mc = self._styled_button(
            "▶ 1. Ejecutar Monte Carlo (Generar Bits Congelados)", "#0275d8", self.start_monte_carlo
        )
        self.btn_stop_mc = QPushButton("⏹ Detener Monte Carlo")
        self.btn_stop_mc.setEnabled(False)
        self.btn_stop_mc.clicked.connect(self.stop_monte_carlo)

        mc_csv_row = QHBoxLayout()
        self.btn_load_csv = QPushButton("📂 Cargar CSV BER_bits...")
        self.btn_load_csv.clicked.connect(self.load_ber_bits_csv)

        self.btn_save_mc_csv = QPushButton("💾 Guardar CSV BER_bits...")
        self.btn_save_mc_csv.setEnabled(False)
        self.btn_save_mc_csv.setToolTip("Guardar los resultados actuales de BER_bits en un nuevo archivo CSV.")
        self.btn_save_mc_csv.clicked.connect(self.save_ber_bits_csv)
        mc_csv_row.addWidget(self.btn_load_csv)
        mc_csv_row.addWidget(self.btn_save_mc_csv)

        self.btn_run_ber = self._styled_button(
            "📈 2. Simular y Graficar Curvas BER", "#5cb85c", self.start_ber_simulation
        )
        self.btn_stop_ber = QPushButton("⏹ Detener Simulación BER")
        self.btn_stop_ber.setEnabled(False)
        self.btn_stop_ber.clicked.connect(self.stop_ber_simulation)

        self.btn_save_ber_csv = QPushButton("💾 Guardar CSV Curvas BER...")
        self.btn_save_ber_csv.setEnabled(False)
        self.btn_save_ber_csv.setToolTip("Exportar las curvas BER vs Eb/N0 simuladas a un nuevo archivo CSV.")
        self.btn_save_ber_csv.clicked.connect(self.save_ber_curves_csv)

        btn_layout.addWidget(self.btn_run_mc)
        btn_layout.addWidget(self.btn_stop_mc)
        btn_layout.addLayout(mc_csv_row)
        btn_layout.addWidget(self.btn_run_ber)
        btn_layout.addWidget(self.btn_stop_ber)
        btn_layout.addWidget(self.btn_save_ber_csv)
        return btn_layout

    # Panel derecho: gráfica BER, barra de progreso y consola de registros.
    def _build_right_panel(self):
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.setup_plot_axes()

        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas, stretch=3)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        right_layout.addWidget(self.progress_bar)

        log_header = QHBoxLayout()
        lbl_log_title = QLabel("Registro de Simulación")
        lbl_log_title.setStyleSheet("font-weight: bold;")
        self.btn_clear_log = QPushButton("Limpiar registro")
        self.btn_clear_log.setFixedWidth(120)
        self.btn_clear_log.clicked.connect(self.clear_log)
        log_header.addWidget(lbl_log_title)
        log_header.addStretch()
        log_header.addWidget(self.btn_clear_log)
        right_layout.addLayout(log_header)

        self.log_console = QPlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(5000)
        self.log_console.setStyleSheet(
            'background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas, "Courier New", monospace;'
        )
        right_layout.addWidget(self.log_console, stretch=1)
        return right_widget

    # Configura la barra de estado con dispositivo de cómputo y monitor de RAM.
    def init_status_bar(self):
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

        self.lbl_status_activity = QLabel("Listo")
        status_bar.addWidget(self.lbl_status_activity, stretch=1)

        self.lbl_status_device = QLabel(f"Dispositivo: {COMPUTE_DEVICE_INFO}")
        self.lbl_status_device.setStyleSheet("padding: 0 8px; font-weight: bold;")
        status_bar.addPermanentWidget(self.lbl_status_device)

        self.lbl_status_ram = QLabel()
        self.lbl_status_ram.setStyleSheet("padding: 0 8px;")
        status_bar.addPermanentWidget(self.lbl_status_ram)

        self.update_ram_indicator()
        self.ram_timer = QTimer(self)
        self.ram_timer.setInterval(2000)
        self.ram_timer.timeout.connect(self.update_ram_indicator)
        self.ram_timer.start()

    # Sincroniza límites iniciales y visibilidad de campos al arrancar la aplicación.
    def sync_initial_ui_state(self):
        self.on_N_changed()
        self.on_channel_changed(self.combo_canal.currentText())
        self.update_code_rate_label()
        self.update_mc_summary_label()
        self.update_mc_status_label()

    # =================================================================
    # EVENTOS Y FUNCIONES AUXILIARES DE LA INTERFAZ
    # =================================================================

    # Actualiza el indicador de uso de memoria RAM en la barra de estado.
    def update_ram_indicator(self):
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        self.lbl_status_ram.setText(f"RAM: {mem.percent:.0f}% ({used_gb:.1f}/{total_gb:.1f} GB)")
        if mem.percent >= 80.0:
            self.lbl_status_ram.setStyleSheet("padding: 0 8px; color: #d9534f; font-weight: bold;")
        elif mem.percent >= 65.0:
            self.lbl_status_ram.setStyleSheet("padding: 0 8px; color: #f0ad4e; font-weight: bold;")
        else:
            self.lbl_status_ram.setStyleSheet("padding: 0 8px;")

    # Devuelve el número actual de bits de información K según N, tasa y modulación.
    def get_current_K(self):
        N = self.combo_N.currentData()
        rate_spec = self.combo_rate.currentData()
        bits = self.combo_bits.currentData()
        if N is None or rate_spec is None or bits is None:
            return 1
        return compute_info_bits(N, rate_spec, bits)

    # Selecciona una tasa predefinida equivalente a K/N o crea una entrada personalizada para respaldos.
    def select_rate_for_K(self, N, target_K, bitsxsimb):
        for idx in range(self.combo_rate.count()):
            data = self.combo_rate.itemData(idx)
            if not isinstance(data, tuple) and compute_info_bits(N, data, bitsxsimb) == target_K:
                self.combo_rate.setCurrentIndex(idx)
                return

        custom_label = f"Personalizada ({target_K}/{N} = {target_K / N:.3f})"
        custom_data = ("exact_k", int(target_K))
        for idx in range(self.combo_rate.count()):
            data = self.combo_rate.itemData(idx)
            if isinstance(data, tuple) and data[0] == "exact_k":
                self.combo_rate.setItemText(idx, custom_label)
                self.combo_rate.setItemData(idx, custom_data)
                self.combo_rate.setCurrentIndex(idx)
                return
        self.combo_rate.addItem(custom_label, custom_data)
        self.combo_rate.setCurrentIndex(self.combo_rate.count() - 1)

    # Actualiza la etiqueta con los bits de información K y la tasa efectiva con CRC.
    def update_code_rate_label(self):
        N = self.combo_N.currentData()
        if N is None:
            return
        K = self.get_current_K()
        crc_len = CRC_LENGTHS.get(self.combo_crc.currentText(), 0)
        rate = K / N
        if K > crc_len:
            eff_rate = (K - crc_len) / N
            self.lbl_code_rate.setText(f"K = {K}  (R = {rate:.3f} | Polar+CRC R_efec = {eff_rate:.3f})")
        else:
            self.lbl_code_rate.setText(f"K = {K}  (R = {rate:.3f} | K <= CRC {crc_len})")

    # Actualiza la cantidad total de bits simulados por canal en Monte Carlo.
    def update_mc_summary_label(self):
        total_bits = self.spin_lote_mc.value() * self.spin_num_iter.value()
        min_ber = 1.0 / total_bits
        self.lbl_mc_total_bits.setText(f"{total_bits:,} bits/canal (BER mín ≈ {min_ber:.1e})")

    # Muestra el estado actual de los bits congelados y alerta si no coincide con N.
    def update_mc_status_label(self):
        N = self.combo_N.currentData()
        if self.ber_bits is None:
            self.lbl_mc_status.setText("Sin cargar")
            self.lbl_mc_status.setStyleSheet("color: #d9534f; font-weight: bold;")
            self.lbl_mc_status.setToolTip("")
            return

        loaded_len = len(self.ber_bits)
        if loaded_len == N:
            self.lbl_mc_status.setText(f"Listo ({loaded_len} bits)")
            self.lbl_mc_status.setStyleSheet("color: #5cb85c; font-weight: bold;")
            self.lbl_mc_status.setToolTip(self.ber_bits_source)
        else:
            self.lbl_mc_status.setText(f"Desajuste ({loaded_len} ≠ N={N})")
            self.lbl_mc_status.setStyleSheet("color: #e67e22; font-weight: bold;")
            self.lbl_mc_status.setToolTip(
                f"El archivo BER_bits cargado tiene {loaded_len} bits y no coincide con N={N}. "
                "Las curvas Polar se omitirán hasta que N coincida o cargue/genere nuevos bits congelados."
            )

    # Configura los ejes de la gráfica logarítmica de BER.
    def setup_plot_axes(self, title="BER vs Eb/N0"):
        self.plot_min_ber = BER_MIN_YLIM
        self.ax.clear()
        self.ax.set_yscale('log')
        self.ax.set_xlabel(r"$E_b/N_0$ [dB]", fontsize=11)
        self.ax.set_ylabel("BER", fontsize=11)
        self.ax.set_title(title, fontsize=12, fontweight='bold')
        self.ax.set_ylim([self.plot_min_ber, 1.0])
        self.ax.grid(True, which="both", linestyle="--", alpha=0.6)
        self.fig.tight_layout()

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_console.appendPlainText(f"[{timestamp}] {message}")
        scrollbar = self.log_console.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        self.log_console.clear()

    # Bloquea o desbloquea los controles según el estado de ejecución.
    def set_simulation_running(self, running, mode=None):
        for grp in self.param_groups:
            grp.setEnabled(not running)
        self.btn_run_mc.setEnabled(not running)
        self.btn_run_ber.setEnabled(not running)
        self.btn_load_csv.setEnabled(not running)
        self.btn_save_mc_csv.setEnabled((not running) and (self.ber_bits is not None))
        self.btn_save_ber_csv.setEnabled((not running) and bool(self.ber_curves_data))

        self.btn_stop_mc.setEnabled(running and mode == "mc")
        self.btn_stop_mc.setText("⏹ Detener Monte Carlo")
        self.btn_stop_ber.setEnabled(running and mode == "ber")
        self.btn_stop_ber.setText("⏹ Detener Simulación BER")

    def on_N_changed(self):
        N = self.combo_N.currentData()
        if N is None:
            return
        self.spin_coher.setMaximum(N)
        if self.chk_flat_fading.isChecked():
            self.spin_coher.setValue(N)
        else:
            self.spin_coher.setValue(max(1, N // 16))
        self.update_code_rate_label()
        self.update_mc_status_label()

    # Alterna entre bloque de coherencia N // 16 y desvanecimiento plano (bloque = N).
    def on_flat_fading_toggled(self, checked):
        N = self.combo_N.currentData()
        if N is not None:
            self.spin_coher.setValue(N if checked else max(1, N // 16))
        is_fading_channel = self.combo_canal.currentText() not in NON_FADING_CHANNELS
        self.spin_coher.setEnabled(is_fading_channel and not checked)

    # Actualiza los valores SNR, parámetros aplicables y opciones válidas al cambiar de canal.
    def on_channel_changed(self, channel_name):
        if channel_name in CHANNEL_SNR_DEFAULTS:
            self.spin_snr_sim.setValue(CHANNEL_SNR_DEFAULTS[channel_name])

        # Habilita solo los parámetros físicos pertinentes al canal seleccionado.
        is_fading = channel_name not in NON_FADING_CHANNELS
        self.chk_flat_fading.setEnabled(is_fading)
        self.spin_coher.setEnabled(is_fading and not self.chk_flat_fading.isChecked())
        self.spin_k_db.setEnabled(channel_name in ("rice", "shadowed_rician"))
        self.spin_sigma_db.setEnabled(channel_name in ("lognormal", "suzuki"))
        self.spin_m_nakagami.setEnabled(channel_name in ("nakagami", "shadowed_rician"))
        self.spin_imp_prob.setEnabled(channel_name == "impulsive")
        self.spin_imp_gamma.setEnabled(channel_name == "impulsive")
        self.update_equalizer_options(channel_name)

    # Permite sin_ecualizar solo en canales sin desvanecimiento multiplicativo (AWGN e Impulsive).
    def update_equalizer_options(self, channel_name):
        no_eq_index = self.combo_eq.findText("no_eq")
        no_eq_enabled = channel_name in NON_FADING_CHANNELS
        model = self.combo_eq.model()
        if isinstance(model, QStandardItemModel) and no_eq_index >= 0:
            item = model.item(no_eq_index)
            if item is not None:
                item.setEnabled(no_eq_enabled)

        if not no_eq_enabled and self.combo_eq.currentText() == "no_eq":
            self.combo_eq.setCurrentText("mmse")

    # Reúne una configuración consistente antes de iniciar una simulación.
    def gather_parameters(self):
        if self.combo_canal.currentText() not in NON_FADING_CHANNELS and self.combo_eq.currentText() == "no_eq":
            self.combo_eq.setCurrentText("mmse")
        return {
            'seed': self.spin_seed.value(),
            'N': self.combo_N.currentData(),
            'K': self.get_current_K(),
            'bitsxsimb': self.combo_bits.currentData(),
            'crc_degree': self.combo_crc.currentText(),
            'canal': self.combo_canal.currentText(),
            'bloq_coher': self.spin_coher.value(),
            'k_factor_db': self.spin_k_db.value(),
            'sigma_db': self.spin_sigma_db.value(),
            'm_nakagami': self.spin_m_nakagami.value(),
            'imp_prob': self.spin_imp_prob.value(),
            'imp_gamma': self.spin_imp_gamma.value(),
            'ecualizador': self.combo_eq.currentText(),
            'list_scl': self.combo_scl.currentData(),
            'LLR_clip': self.spin_llr.value(),
            'lote_mc': self.spin_lote_mc.value(),
            'num_iter': self.spin_num_iter.value(),
            'snr_simulacion': self.spin_snr_sim.value(),
            'ebno_min': self.spin_ebno_min.value(),
            'ebno_max': self.spin_ebno_max.value(),
            'ebno_rango': self.spin_ebno_rango.value(),
            'lote_berplot': self.spin_lote_ber.value(),
        }

    # Valida restricciones matemáticas y de Sionna antes de ejecutar cualquier hilo.
    def validate_parameters(self, params, mode):
        errors = []
        N = params['N']
        K = params['K']
        bits = params['bitsxsimb']

        if not (1 <= K < N):
            errors.append(f"Los bits de información K ({K}) deben cumplir 1 <= K < N ({N}).")

        if N % bits != 0:
            errors.append(
                f"La longitud de código N ({N}) no es divisible por Bits por símbolo ({bits}). "
                "Nota: 64-QAM (6 bits/símbolo) es incompatible con longitudes Polar potencias de 2."
            )

        if params['canal'] not in NON_FADING_CHANNELS and not (1 <= params['bloq_coher'] <= N):
            errors.append(f"El bloque de coherencia ({params['bloq_coher']}) debe estar entre 1 y N ({N}).")

        if mode == "ber":
            if K % bits != 0:
                errors.append(
                    f"Para la curva de referencia Sin código, K ({K}) debe ser divisible por Bits por símbolo ({bits})."
                )
            if params['ebno_min'] >= params['ebno_max']:
                errors.append(
                    f"Eb/N0 mín ({params['ebno_min']} dB) debe ser estrictamente menor que "
                    f"Eb/N0 máx ({params['ebno_max']} dB)."
                )
            if params['ebno_rango'] < 2:
                errors.append("El número de puntos de Eb/N0 debe ser al menos 2.")

        if errors:
            QMessageBox.warning(
                self,
                "Parámetros de Simulación Inválidos",
                "Corrija los siguientes problemas antes de ejecutar:\n\n• " + "\n• ".join(errors)
            )
            return False
        return True

    # =================================================================
    # GESTIÓN DE RESPALDOS Y ARCHIVOS CSV
    # =================================================================

    # Verifica si los metadatos de un respaldo .npz son aplicables a los controles de la interfaz.
    def is_backup_meta_applicable(self, loaded_bits, saved_index, saved_meta):
        if not isinstance(saved_meta, dict) or not all(k in saved_meta for k in MC_BACKUP_CORE_KEYS):
            return False
        try:
            backup_N = int(saved_meta["N"])
            backup_K = int(saved_meta["K"])
            backup_bits = int(saved_meta["bitsxsimb"])
            backup_canal = str(saved_meta["canal"])
            backup_eq = str(saved_meta["ecualizador"])
        except (TypeError, ValueError):
            return False

        return (
            len(loaded_bits) == backup_N
            and 0 <= saved_index <= backup_N
            and 1 <= backup_K < backup_N
            and self.combo_N.findData(backup_N) != -1
            and self.combo_bits.findData(backup_bits) != -1
            and self.combo_canal.findText(backup_canal) != -1
            and self.combo_eq.findText(backup_eq) != -1
        )

    # Aplica en orden seguro los parámetros guardados en un respaldo .npz sobre la interfaz.
    def apply_backup_meta_to_ui(self, saved_meta):
        backup_N = int(saved_meta["N"])
        backup_K = int(saved_meta["K"])
        backup_bits = int(saved_meta["bitsxsimb"])

        n_idx = self.combo_N.findData(backup_N)
        if n_idx >= 0:
            self.combo_N.setCurrentIndex(n_idx)

        bits_idx = self.combo_bits.findData(backup_bits)
        if bits_idx >= 0:
            self.combo_bits.setCurrentIndex(bits_idx)

        self.select_rate_for_K(backup_N, backup_K, backup_bits)

        canal_idx = self.combo_canal.findText(str(saved_meta["canal"]))
        if canal_idx >= 0:
            self.combo_canal.setCurrentIndex(canal_idx)

        # Restaura estos valores después de cambiar N y canal para no perderlos por valores por defecto.
        backup_coher = int(saved_meta["bloq_coher"])
        self.chk_flat_fading.setChecked(backup_coher == backup_N)
        self.spin_coher.setValue(backup_coher)
        self.spin_k_db.setValue(float(saved_meta["k_factor_db"]))
        self.spin_sigma_db.setValue(float(saved_meta["sigma_db"]))
        self.spin_m_nakagami.setValue(float(saved_meta.get("m_nakagami", 1.5)))
        self.spin_imp_prob.setValue(float(saved_meta.get("imp_prob", 0.05)))
        self.spin_imp_gamma.setValue(float(saved_meta.get("imp_gamma", 50.0)))

        eq_idx = self.combo_eq.findText(str(saved_meta["ecualizador"]))
        if eq_idx >= 0:
            self.combo_eq.setCurrentIndex(eq_idx)

        self.spin_llr.setValue(float(saved_meta["LLR_clip"]))
        self.spin_snr_sim.setValue(float(saved_meta["snr_simulacion"]))

        self.update_code_rate_label()
        self.update_mc_summary_label()
        self.update_mc_status_label()

    # Busca primero el respaldo del N actual y, si no existe, el .npz más reciente de otro N.
    @staticmethod
    def _find_mc_backup_file(N):
        exact_path = f"BER_bits_backup_{N}.npz"
        if os.path.exists(exact_path):
            return exact_path, True
        other_backups = sorted(glob.glob("BER_bits_backup_*.npz"), key=os.path.getmtime, reverse=True)
        if other_backups:
            return other_backups[0], False
        return None, True

    # Diálogo cuando el respaldo coincide exactamente con la configuración actual.
    def _prompt_matching_backup(self, backup_file, saved_index, N):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("¿Reanudar Respaldo Monte Carlo?")
        msg.setText(
            f"Se encontró un respaldo compatible '{backup_file}' en el bit-canal {saved_index}/{N}.\n"
            "¿Desea reanudar desde el respaldo o empezar de cero?"
        )
        btn_resume = msg.addButton("Reanudar", QMessageBox.ButtonRole.AcceptRole)
        btn_fresh = msg.addButton("Descartar y empezar de cero", QMessageBox.ButtonRole.DestructiveRole)
        msg.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_resume:
            return ("resume", backup_file, None)
        if clicked == btn_fresh:
            return ("fresh", backup_file, None)
        return ("cancel", None, None)

    # Compara metadatos de respaldo de forma retrocompatible con respaldos de 10 claves.
    @staticmethod
    def _backup_meta_matches(saved_meta, current_meta):
        if not isinstance(saved_meta, dict):
            return False
        keys_to_check = [k for k in MC_BACKUP_KEYS if k in saved_meta]
        if not all(k in saved_meta for k in MC_BACKUP_CORE_KEYS):
            return False
        return all(saved_meta.get(k) == current_meta.get(k) for k in keys_to_check)

    # Diálogo cuando el respaldo tiene parámetros distintos (permite actualizar y reanudar).
    def _prompt_mismatched_backup(self, backup_file, same_N_backup, loaded_bits, saved_index, saved_meta, current_meta):
        diffs = [
            f"{k}: respaldo={saved_meta.get(k)} vs actual={current_meta.get(k)}"
            for k in MC_BACKUP_KEYS
            if k in saved_meta and saved_meta.get(k) != current_meta.get(k)
        ]
        diff_text = (
            "\n".join(diffs)
            if diffs
            else "La longitud o metadatos del respaldo difieren de la configuración actual."
        )
        can_apply = self.is_backup_meta_applicable(loaded_bits, saved_index, saved_meta)
        fresh_label = "Descartar y empezar de cero" if same_N_backup else "Ignorar respaldo y empezar de cero"

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Diferencia en Parámetros del Respaldo")
        msg.setText(
            f"El respaldo existente '{backup_file}' (bit-canal {saved_index}/{len(loaded_bits)}) "
            f"utiliza parámetros distintos:\n\n{diff_text}\n\nElija cómo desea continuar:"
        )
        btn_update = (
            msg.addButton("Actualizar parámetros y reanudar", QMessageBox.ButtonRole.AcceptRole)
            if can_apply else None
        )
        btn_fresh = msg.addButton(fresh_label, QMessageBox.ButtonRole.DestructiveRole)
        msg.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if btn_update is not None and clicked == btn_update:
            return ("update_and_resume", backup_file, saved_meta)
        if clicked == btn_fresh:
            return ("fresh", backup_file if same_N_backup else None, None)
        return ("cancel", None, None)

    # Verifica si existe un respaldo Monte Carlo previo y pregunta cómo proceder.
    def resolve_mc_backup_action(self, params):
        N = params['N']
        backup_file, same_N_backup = self._find_mc_backup_file(N)
        if backup_file is not None:
            try:
                loaded_bits, saved_index, saved_meta = load_mc_backup(backup_file)
            except Exception as e:
                if not same_N_backup:
                    return ("fresh", None, None)
                reply = QMessageBox.warning(
                    self,
                    "Respaldo Dañado Detectado",
                    f"No se pudo leer el archivo de respaldo '{backup_file}' ({e}).\n\n"
                    "¿Descartarlo e iniciar desde cero?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
                )
                if reply == QMessageBox.StandardButton.Yes:
                    return ("fresh", backup_file, None)
                return ("cancel", None, None)

            current_meta = extract_mc_backup_meta(params)
            if (
                same_N_backup
                and len(loaded_bits) == N
                and self._backup_meta_matches(saved_meta, current_meta)
                and 0 <= saved_index <= N
            ):
                return self._prompt_matching_backup(backup_file, saved_index, N)
            return self._prompt_mismatched_backup(
                backup_file, same_N_backup, loaded_bits, saved_index, saved_meta, current_meta
            )

        legacy_file = f"BER_bits_backup_{N}.npy"
        if os.path.exists(legacy_file):
            reply = QMessageBox.information(
                self,
                "Respaldo Anterior Detectado",
                f"Se encontró un archivo de respaldo antiguo '{legacy_file}' (sin metadatos de parámetros).\n"
                "¿Iniciar una ejecución nueva usando el formato seguro de respaldo?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
            )
            return ("fresh", None, None) if reply == QMessageBox.StandardButton.Yes else ("cancel", None, None)

        return ("fresh", None, None)

    # Abre el diálogo para elegir una ruta de guardado CSV y crea su carpeta padre si es necesario.
    def _prompt_csv_save_path(self, dialog_title, default_path):
        file_path, _ = QFileDialog.getSaveFileName(
            self, dialog_title, default_path, "Archivos CSV (*.csv);;Todos los archivos (*)"
        )
        if not file_path:
            return None
        if not file_path.lower().endswith(".csv"):
            file_path += ".csv"
        parent_dir = os.path.dirname(file_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        return file_path

    # Permite guardar los valores BER_bits actuales en un nuevo archivo CSV elegido por el usuario.
    def save_ber_bits_csv(self):
        if self.ber_bits is None:
            QMessageBox.information(
                self, "Sin Datos para Guardar", "Ejecute Monte Carlo o cargue un CSV de BER_bits primero."
            )
            return

        p = self.active_mc_params if self.active_mc_params else self.gather_parameters()
        default_path = default_ber_bits_csv_path(p, len(self.ber_bits))
        try:
            file_path = self._prompt_csv_save_path("Guardar archivo CSV de BER_bits", default_path)
            if not file_path:
                return
            pd.DataFrame({"BER bits": self.ber_bits}).to_csv(file_path, index=True, encoding="utf-8-sig")
            self.ber_bits_source = file_path
            self.update_mc_status_label()
            self.log(f"💾 CSV de BER_bits guardado en: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al Guardar CSV", str(e))

    # Carga desde CSV un conjunto previamente generado de valores BER por bit.
    def load_ber_bits_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar archivo CSV de BER_bits", "", "Archivos CSV (*.csv);;Todos los archivos (*)"
        )
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path, encoding="utf-8-sig")
            if "BER bits" not in df.columns:
                raise ValueError("El archivo CSV no contiene la columna 'BER bits'.")

            numeric_col = pd.to_numeric(df["BER bits"], errors="coerce")
            if numeric_col.isna().any():
                raise ValueError("La columna 'BER bits' del CSV contiene valores vacíos o no numéricos.")

            loaded_bits = numeric_col.to_numpy(dtype=np.float64)
            if not np.all(np.isfinite(loaded_bits)):
                raise ValueError("La columna 'BER bits' del CSV contiene valores infinitos.")
            if np.any((loaded_bits < 0.0) | (loaded_bits > 1.0)):
                raise ValueError("Los valores de 'BER bits' deben estar en el rango [0.0, 1.0].")

            loaded_len = len(loaded_bits)
            n_index = self.combo_N.findData(loaded_len)
            if n_index == -1:
                raise ValueError(
                    f"El CSV cargado tiene {loaded_len} bit-canales, lo cual no es una longitud N soportada "
                    "(potencias de 2 entre 32 y 8192)."
                )

            current_N = self.combo_N.currentData()
            if loaded_len != current_N:
                QMessageBox.information(
                    self,
                    "Longitud de Código Ajustada",
                    f"El CSV cargado tiene {loaded_len} bits, mientras que N estaba en {current_N}.\n"
                    f"Se ajustó N a {loaded_len}."
                )
                self.combo_N.setCurrentIndex(n_index)

            self.ber_bits = loaded_bits
            self.ber_bits_source = file_path
            self.active_mc_params = None
            self.btn_save_mc_csv.setEnabled(True)
            self.update_mc_status_label()
            self.log(f"📂 BER_bits cargado ({loaded_len} bits) desde: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al Cargar CSV", str(e))

    # Exporta a un nuevo archivo CSV las curvas BER frente a Eb/N0 simuladas.
    def save_ber_curves_csv(self):
        if self.ber_curves_ebno is None or not self.ber_curves_data:
            QMessageBox.information(self, "Sin Datos para Guardar", "Ejecute primero una simulación BER vs Eb/N0.")
            return

        p = self.active_ber_params if self.active_ber_params else self.gather_parameters()
        default_path = default_ber_curves_csv_path(p)
        try:
            file_path = self._prompt_csv_save_path("Guardar archivo CSV de curvas BER", default_path)
            if not file_path:
                return
            data = {"Eb/N0 [dB]": self.ber_curves_ebno}
            data.update(self.ber_curves_data)
            pd.DataFrame(data).to_csv(file_path, index=False, encoding="utf-8-sig")
            self.log(f"💾 CSV de curvas BER guardado en: {file_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error al Guardar CSV", str(e))

    # =================================================================
    # CONTROLADORES DE SIMULACIÓN
    # =================================================================

    # Comprueba si existe algún hilo de simulación o cálculo activo.
    def has_active_worker(self):
        return any(
            w is not None and w.isRunning()
            for w in (self.mc_worker, self.ber_worker, self.snr_worker)
        )

    # Calcula automáticamente el SNR de diseño [dB] óptimo por información mutua.
    def start_auto_design_snr(self):
        if self.has_active_worker():
            return

        params = self.gather_parameters()
        if not self.validate_parameters(params, mode="mc"):
            return

        self.set_simulation_running(True, mode="auto_snr")
        self.progress_bar.setRange(0, AUTO_SNR_BISECTION_STEPS)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Auto SNR: %v / %m")
        self.lbl_status_activity.setText("Estimando SNR de diseño por información mutua...")
        self.log(
            f"⚡ Estimando SNR de diseño Monte Carlo "
            f"(N={params['N']}, K={params['K']}, R={params['K']/params['N']:.3f}, "
            f"Canal={params['canal']}, Eq={params['ecualizador']}, bits/símb={params['bitsxsimb']})..."
        )

        self.snr_worker = DesignSNRWorker(params)
        self.snr_worker.progress_signal.connect(self.update_auto_snr_progress)
        self.snr_worker.finished_signal.connect(self.on_auto_snr_finished)
        self.snr_worker.error_signal.connect(self.on_worker_error)
        self.snr_worker.finished.connect(self.cleanup_snr_worker)
        self.snr_worker.start()

    def update_auto_snr_progress(self, step, total, snr_db, mi_val):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(step)
        self.lbl_status_activity.setText(
            f"Estimando SNR de diseño ({step}/{total}): {snr_db:.2f} dB → I(W) = {mi_val:.3f}"
        )

    def on_auto_snr_finished(self, snr_db, achieved_mi, target_mi, est_ber):
        self.spin_snr_sim.setValue(snr_db)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.lbl_status_activity.setText(f"SNR de diseño calculado: {snr_db:.2f} dB")
        self.set_simulation_running(False)
        self.log(
            f"✅ SNR de diseño calculado = {snr_db:.2f} dB "
            f"(I(W) objetivo = {target_mi:.3f}, I(W) logrado = {achieved_mi:.3f}, "
            f"BER sin código est. = {est_ber:.3f})"
        )

    def cleanup_snr_worker(self):
        if self.snr_worker is not None:
            self.snr_worker.deleteLater()
            self.snr_worker = None

    # Aplica en la interfaz los ajustes de reducción sugeridos por la verificación de recursos.
    def _apply_resource_recommendations(self, recommendations):
        if 'N' in recommendations:
            idx = self.combo_N.findData(int(recommendations['N']))
            if idx >= 0:
                self.combo_N.setCurrentIndex(idx)
        if 'lote_mc' in recommendations:
            self.spin_lote_mc.setValue(int(recommendations['lote_mc']))
        if 'num_iter' in recommendations:
            self.spin_num_iter.setValue(int(recommendations['num_iter']))
        if 'lote_berplot' in recommendations:
            self.spin_lote_ber.setValue(int(recommendations['lote_berplot']))
        if 'ebno_rango' in recommendations:
            self.spin_ebno_rango.setValue(int(recommendations['ebno_rango']))
        if 'list_scl' in recommendations:
            scl_idx = self.combo_scl.findData(int(recommendations['list_scl']))
            if scl_idx >= 0:
                self.combo_scl.setCurrentIndex(scl_idx)

    # Pre-verifica CPU/GPU y RAM antes de lanzar Monte Carlo o BER, sugiriendo reducir carga si aplica.
    def precheck_system_resources(self, params, mode, lock_n=False):
        warnings, recommendations, summary = assess_system_resources(
            params, mode, has_gpu=HAS_GPU, lock_n=lock_n
        )
        if not warnings:
            return True, params

        labels = {
            'N': "Longitud N",
            'lote_mc': "Lote paralelo MC",
            'num_iter': "Iteraciones MC",
            'lote_berplot': "Tamaño de lote BER",
            'ebno_rango': "Puntos Eb/N0",
            'list_scl': "Tamaño lista SCL",
        }
        rec_lines = [
            f"• {labels.get(k, k)}: {params[k]} → {v}"
            for k, v in recommendations.items()
        ]
        dev_str = COMPUTE_DEVICE_INFO if summary["has_gpu"] else f"CPU ({summary['cores']} hilos)"
        body = (
            f"Recursos detectados: {dev_str} | "
            f"RAM disponible: {summary['avail_gb']:.1f} GB ({summary['mem_pct']:.0f}% en uso)\n\n"
            "Advertencias:\n• " + "\n• ".join(warnings)
        )
        if rec_lines:
            body += "\n\nReducción de parámetros recomendada:\n" + "\n".join(rec_lines)

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Verificación de Recursos del Sistema")
        msg.setText(body)
        btn_apply = (
            msg.addButton("Aplicar recomendaciones y continuar", QMessageBox.ButtonRole.AcceptRole)
            if recommendations else None
        )
        btn_continue = msg.addButton("Continuar sin cambios", QMessageBox.ButtonRole.ActionRole)
        msg.addButton("Cancelar", QMessageBox.ButtonRole.RejectRole)
        msg.exec()

        clicked = msg.clickedButton()
        if btn_apply is not None and clicked == btn_apply:
            self._apply_resource_recommendations(recommendations)
            updated = self.gather_parameters()
            self.log(
                "⚙️ Parámetros ajustados según recursos disponibles: "
                + ", ".join(f"{k}={v}" for k, v in recommendations.items())
            )
            return True, updated
        if clicked == btn_continue:
            return True, params
        return False, params

    # Inicia y supervisa la generación Monte Carlo de bits congelados.
    def start_monte_carlo(self):
        if self.has_active_worker():
            return

        params = self.gather_parameters()
        action, backup_file, saved_meta = self.resolve_mc_backup_action(params)
        if action == "cancel":
            return

        if action == "update_and_resume":
            self.apply_backup_meta_to_ui(saved_meta)
            params = self.gather_parameters()
            self.log(f"🔄 Parámetros de la interfaz actualizados desde el respaldo '{backup_file}'.")

        if not self.validate_parameters(params, mode="mc"):
            return

        resume_flag = action in ("resume", "update_and_resume")
        proceed, params = self.precheck_system_resources(params, mode="mc", lock_n=resume_flag)
        if not proceed or not self.validate_parameters(params, mode="mc"):
            return
        self.active_mc_params = dict(params)
        self.set_simulation_running(True, mode="mc")
        self.progress_bar.setRange(0, params['N'])
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m (%p%)")
        self.lbl_status_activity.setText(f"Ejecutando Monte Carlo (0/{params['N']})...")

        self.log(
            f"--- Iniciando generación Monte Carlo de bits congelados "
            f"(N={params['N']}, K={params['K']}, Canal={params['canal']}, "
            f"Eq={params['ecualizador']}, SNR={params['snr_simulacion']} dB) ---"
        )
        self.mc_worker = MonteCarloWorker(params, resume_backup=resume_flag)
        self.mc_worker.log_signal.connect(self.log)
        self.mc_worker.progress_signal.connect(self.update_mc_progress)
        self.mc_worker.finished_signal.connect(self.on_mc_finished)
        self.mc_worker.stopped_signal.connect(self.on_mc_stopped)
        self.mc_worker.error_signal.connect(self.on_worker_error)
        self.mc_worker.finished.connect(self.cleanup_mc_worker)
        self.mc_worker.start()

    def stop_monte_carlo(self):
        if self.mc_worker and self.mc_worker.isRunning():
            self.mc_worker.request_stop()
            self.btn_stop_mc.setEnabled(False)
            self.btn_stop_mc.setText("⏳ Deteniendo...")
            self.lbl_status_activity.setText("Deteniendo Monte Carlo al finalizar el lote actual...")
            self.log("⏹ Detención solicitada. Esperando a que finalice el lote actual de forma segura...")

    def cleanup_mc_worker(self):
        if self.mc_worker is not None:
            self.mc_worker.deleteLater()
            self.mc_worker = None

    # Actualiza el progreso, tiempo transcurrido y ETA de Monte Carlo.
    def update_mc_progress(self, current, total, elapsed, eta):
        self.progress_bar.setValue(current)
        if elapsed > 0:
            self.lbl_status_activity.setText(
                f"Monte Carlo: {current}/{total} | "
                f"Transcurrido: {format_duration(elapsed)} | ETA: {format_duration(eta)}"
            )
        else:
            self.lbl_status_activity.setText(f"Monte Carlo: {current}/{total}")

    # Guarda en la interfaz los resultados completados de Monte Carlo.
    def on_mc_finished(self, ber_bits, filename):
        self.ber_bits = ber_bits
        self.ber_bits_source = filename
        self.update_mc_status_label()
        self.log(f"✅ ¡Monte Carlo completado! Guardado automáticamente en: {filename}")
        self.lbl_status_activity.setText("Monte Carlo completado")
        self.set_simulation_running(False)

    # Restaura los controles solo cuando el hilo Monte Carlo se detiene por completo.
    def on_mc_stopped(self, completed, total, _backup_path):
        self.lbl_status_activity.setText(f"Monte Carlo detenido en {completed}/{total} (respaldo guardado)")
        self.set_simulation_running(False)

    # Inicia la barrida BER y prepara la gráfica para las nuevas curvas.
    def start_ber_simulation(self):
        if self.has_active_worker():
            return

        params = self.gather_parameters()
        if not self.validate_parameters(params, mode="ber"):
            return

        has_matching_bits = self.ber_bits is not None and len(self.ber_bits) == params['N']
        proceed, params = self.precheck_system_resources(params, mode="ber", lock_n=has_matching_bits)
        if not proceed or not self.validate_parameters(params, mode="ber"):
            return

        self.active_ber_params = dict(params)
        self.ber_curves_ebno = None
        self.ber_curves_data = {}
        self.setup_plot_axes(
            f"Comparación BER vs Eb/N0 — Canal {params['canal'].capitalize()} "
            f"(N={params['N']}, K={params['K']}, Eq={params['ecualizador'].upper()})"
        )
        self.ax.set_xlim([params['ebno_min'], params['ebno_max']])
        self.canvas.draw()

        self.set_simulation_running(True, mode="ber")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m curvas (%p%)")
        self.lbl_status_activity.setText("Iniciando simulación BER vs Eb/N0...")

        self.log("--- Iniciando simulación BER vs Eb/N0 ---")
        self.ber_worker = BERSimWorker(params, self.ber_bits)
        self.ber_worker.log_signal.connect(self.log)
        self.ber_worker.progress_signal.connect(self.update_ber_progress)
        self.ber_worker.curve_ready_signal.connect(self.plot_ber_curve)
        self.ber_worker.finished_signal.connect(self.on_ber_finished)
        self.ber_worker.stopped_signal.connect(self.on_ber_stopped)
        self.ber_worker.error_signal.connect(self.on_worker_error)
        self.ber_worker.finished.connect(self.cleanup_ber_worker)
        self.ber_worker.start()

    def stop_ber_simulation(self):
        if self.ber_worker and self.ber_worker.isRunning():
            self.ber_worker.request_stop()
            self.btn_stop_ber.setEnabled(False)
            self.btn_stop_ber.setText("⏳ Deteniendo...")
            self.lbl_status_activity.setText("Deteniendo simulación BER al finalizar la curva actual...")
            self.log("⏹ Detención solicitada. Esperando a que finalice la curva actual...")

    def cleanup_ber_worker(self):
        if self.ber_worker is not None:
            self.ber_worker.deleteLater()
            self.ber_worker = None

    def update_ber_progress(self, current, total, status_text):
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current)
        self.lbl_status_activity.setText(status_text)

    # Dibuja cada curva BER recibida, almacena sus datos y actualiza su leyenda.
    def plot_ber_curve(self, legend_label, ebno_dbs, ber_values):
        self.ber_curves_ebno = np.array(ebno_dbs, copy=True)
        self.ber_curves_data[legend_label] = np.array(ber_values, copy=True)

        crc_name = (
            self.active_ber_params['crc_degree']
            if self.active_ber_params
            else self.combo_crc.currentText()
        )
        markers = dict(CURVE_MARKERS)
        markers[f"{CURVE_POLAR} + {crc_name}"] = "^-"
        style = markers.get(legend_label, "d-")

        valid_mask = np.isfinite(ber_values) & (ber_values > 0.0)
        zero_count = int(np.sum(ber_values == 0.0))

        if np.any(valid_mask):
            min_pos_ber = float(np.min(ber_values[valid_mask]))
            if min_pos_ber < self.plot_min_ber:
                decade = 10.0 ** np.floor(np.log10(min_pos_ber))
                self.plot_min_ber = max(BER_FLOOR_YLIM, decade)
                self.ax.set_ylim([self.plot_min_ber, 1.0])

            self.ax.plot(
                ebno_dbs[valid_mask],
                ber_values[valid_mask],
                style,
                linewidth=2,
                markersize=6,
                label=legend_label
            )
            self.ax.legend(loc="best", fontsize=10)
            self.fig.tight_layout()
            self.canvas.draw()
            if zero_count > 0:
                self.log(
                    f"✔ Curva graficada: {legend_label} "
                    f"({zero_count} punto(s) con BER=0 omitidos en escala logarítmica)"
                )
            else:
                self.log(f"✔ Curva graficada: {legend_label}")
        else:
            self.log(
                f"ℹ️ La curva '{legend_label}' tuvo 0 errores en todos los puntos Eb/N0 "
                "(sin puntos para escala logarítmica)."
            )

    # Restaura los controles al finalizar todas las simulaciones BER.
    def on_ber_finished(self):
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.lbl_status_activity.setText("Simulación BER completada")
        self.set_simulation_running(False)
        self.log("✅ Todas las curvas BER fueron simuladas y graficadas.")

    def on_ber_stopped(self):
        self.progress_bar.setFormat("%p%")
        self.lbl_status_activity.setText("Simulación BER detenida por el usuario")
        self.set_simulation_running(False)
        self.log("⚠️ Simulación BER detenida por el usuario.")

    # Informa errores de los hilos y restablece el estado de la interfaz.
    def on_worker_error(self, err_msg, detail_tb=""):
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_status_activity.setText("Error de simulación")
        self.set_simulation_running(False)
        self.log(f"❌ ERROR: {err_msg}")
        if detail_tb:
            self.log(detail_tb.rstrip())
        QMessageBox.critical(self, "Error de Simulación", err_msg)

    # Gestiona el cierre seguro de la ventana cuando existen hilos activos.
    def closeEvent(self, event):
        mc_running = self.mc_worker is not None and self.mc_worker.isRunning()
        ber_running = self.ber_worker is not None and self.ber_worker.isRunning()
        snr_running = self.snr_worker is not None and self.snr_worker.isRunning()

        if mc_running or ber_running or snr_running:
            reply = QMessageBox.question(
                self,
                "Simulación en Progreso",
                "Hay una simulación o cálculo en ejecución.\n"
                "¿Desea detenerlo de forma segura (guardando respaldo Monte Carlo si aplica) y salir?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            self.lbl_status_activity.setText("Deteniendo simulación activa antes de salir...")
            QApplication.processEvents()

            if mc_running:
                self.mc_worker.request_stop()
                self.mc_worker.wait(10000)
            if ber_running:
                self.ber_worker.request_stop()
                self.ber_worker.wait(10000)
            if snr_running:
                self.snr_worker.request_stop()
                self.snr_worker.wait(5000)

        event.accept()


# Punto de entrada: crea la aplicación Qt y muestra la ventana principal.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PolarSimGUI()
    window.show()
    sys.exit(app.exec())
