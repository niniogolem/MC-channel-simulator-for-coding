"""Aplicación de escritorio PyQt6 para simulación de canales inalámbricos,
diseño Monte Carlo de códigos Polar (Genie-Aided SC) y comparación BER/BLER
frente a códigos LDPC 5G NR sobre TensorFlow / NVIDIA Sionna.

Estructura del módulo:
    0. Configuración de dispositivo (GPU/CPU) e importaciones
    1. Constantes, tooltips de interfaz y utilidades de persistencia/recursos
    2. Modelos de canal inalámbrico, CSI imperfecto, ecualizadores y demapeo (XLA)
    3. Primitivas Polar XLA (Polarization Weight, mariposa y Genie-Aided SC)
    4. Bloques de sistema Sionna (Sin código, Polar, Polar+CRC, LDPC 5G NR)
    5. Hilos de trabajo en segundo plano (MonteCarloWorker, BERSimWorker, DesignSNRWorker)
    6. Ventana principal PyQt6 (PolarSimGUI)
"""

import gc
import glob
import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np
import pandas as pd
import psutil

# Configura la GPU (o fuerza CPU con --cpu) antes de importar TensorFlow/Sionna.
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
elif os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf


def configure_compute_device():
    """Detecta GPUs disponibles y habilita el crecimiento dinámico de VRAM."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return "CPU", False
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception as exc:
            print(f"Advertencia al configurar memoria GPU: {exc}")
    try:
        details = tf.config.experimental.get_device_details(gpus[0])
        device_name = details.get("device_name", gpus[0].name)
    except Exception:
        device_name = gpus[0].name
    return f"GPU: {device_name}", True


COMPUTE_DEVICE_INFO, HAS_GPU = configure_compute_device()
tf.get_logger().setLevel("ERROR")

import sionna

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QStandardItemModel
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

import matplotlib
matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# =====================================================================
# 1. CONSTANTES, TOOLTIPS Y UTILIDADES DE PERSISTENCIA / RECURSOS
# =====================================================================

# Longitudes en bits de cada polinomio CRC 3GPP 5G NR soportado por Sionna.
CRC_LENGTHS = {
    "CRC24A": 24,
    "CRC24B": 24,
    "CRC24C": 24,
    "CRC16": 16,
    "CRC11": 11,
    "CRC6": 6,
}

# Tasas de código estándar predefinidas: (etiqueta, tasa nominal R = K/N).
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

# Valores guía de SNR de diseño [dB] por canal (verificados para R = 0.5).
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

AVAILABLE_CHANNELS = (
    "suzuki",
    "awgn",
    "rayleigh",
    "rice",
    "lognormal",
    "nakagami",
    "shadowed_rician",
    "impulsive",
)

# Canales sin desvanecimiento multiplicativo (h = 1, admiten ecualizador 'no_eq').
NON_FADING_CHANNELS = ("awgn", "impulsive")

# Etiquetas y estilos de marcador compartidos para las curvas BER/BLER.
CURVE_UNCODED = "Sin código"
CURVE_POLAR = "Código Polar"
CURVE_LDPC = "Código LDPC 5G NR"
CURVE_MARKERS = {
    CURVE_UNCODED: "o-",
    CURVE_POLAR: "s-",
    CURVE_LDPC: "D-",
}

# Parámetros operativos de Monte Carlo, Auto SNR y gráfica logarítmica.
MC_RAM_LIMIT_PERCENT = 85.0
MC_BACKUP_INTERVAL = 10
MC_CHUNK_SYNC = 50
MC_RESUME_SEED_STRIDE = 1_000_003  # Desplazamiento primo para evitar repetir lotes al reanudar
AUTO_SNR_TARGET_ALPHA = 0.88
AUTO_SNR_BISECTION_STEPS = 12
AUTO_SNR_MIN_FADING_BLOCKS = 4096
BER_MIN_YLIM = 1e-6
BER_FLOOR_YLIM = 1e-9

# Claves obligatorias (retrocompatibles) y completas de metadatos en respaldos .npz.
MC_BACKUP_CORE_KEYS = (
    "N", "K", "bitsxsimb", "canal", "bloq_coher",
    "k_factor_db", "sigma_db", "ecualizador", "LLR_clip", "snr_simulacion",
)
MC_BACKUP_KEYS = MC_BACKUP_CORE_KEYS + (
    "m_nakagami", "imp_prob", "imp_gamma", "imp_demapper", "csi_err_var",
    "seed", "lote_mc", "num_iter",
)

# Textos de ayuda centralizados para los controles de la interfaz gráfica.
TOOLTIPS = {
    "seed": (
        "Semilla aleatoria para el RNG de Sionna y el interleaver (reproducibilidad).\n"
        "Recomendado: 0 (semilla fija para comparaciones repetibles)."
    ),
    "N": (
        "Longitud de palabra código Polar N (debe ser potencia de 2).\n"
        "Valores típicos: 128, 256, 512 o 1024 (por defecto: 1024)."
    ),
    "rate": (
        "Tasas de código estándar predefinidas (K se calcula automáticamente como N × R).\n"
        "Referencia recomendada: 1/2 (0.500).\n"
        "Nota: LDPC 5G NR soporta tasas aprox. entre 1/5 y 11/12 (K >= 12)."
    ),
    "bits": (
        "Orden de modulación en bits por símbolo (1=BPSK, 2=QPSK, 4=16-QAM).\n"
        "N (y K para Sin código) deben ser divisibles por este valor.\n"
        "Nota: 64-QAM (6) no es divisible en longitudes Polar potencias de 2."
    ),
    "crc": (
        "Polinomio CRC 3GPP 5G NR usado por Polar + CRC (CA-SCL).\n"
        "Recomendado: CRC24A (estándar 5G NR de 24 bits; requiere K > longitud CRC)."
    ),
    "canal": (
        "Modelo de canal inalámbrico:\n"
        "• awgn: solo ruido gaussiano aditivo (sin desvanecimiento)\n"
        "• rayleigh: desvanecimiento multitrayecto NLOS\n"
        "• rice: multitrayecto con LOS (usa Factor K)\n"
        "• lognormal: sombreado lento (usa σ)\n"
        "• suzuki: Rayleigh × Lognormal (usa σ)\n"
        "• nakagami: desvanecimiento Nakagami-m (usa parámetro m)\n"
        "• shadowed_rician: satelital LMS/NTN con LOS sombreada (usa K y m)\n"
        "• impulsive: AWGN + ruido impulsivo Bernoulli-Gaussian (usa p y Γ)"
    ),
    "coher": (
        "Cantidad de símbolos que comparten la misma ganancia de canal h "
        "(solo canales con desvanecimiento).\n"
        "Recomendado: N // 16 (ej. 64 para N=1024, 32 para N=512) o N para desvanecimiento plano.\n"
        "Valores menores modelan desvanecimiento más rápido; mayores producen ráfagas más largas."
    ),
    "flat_fading": (
        "Activa desvanecimiento plano en toda la palabra código (bloque de coherencia = N).\n"
        "• Desmarcado: usa sub-bloques de coherencia (por defecto N // 16, editable).\n"
        "• Marcado: mantiene una única ganancia h constante durante todos los símbolos del bloque."
    ),
    "k_db": (
        "Relación de potencia LOS/NLOS en dB (canales Rice y Shadowed-Rician).\n"
        "Valores razonables típicos:\n"
        "• Entornos suburbanos / satelital AS: 4.0 a 15.0 dB (por defecto: 6.0 dB)\n"
        "• LOS débil / sombreado fuerte: -1 a 3 dB (tiende a Rayleigh si K → -∞)\n"
        "• LOS fuerte: > 10 dB (tiende a AWGN)"
    ),
    "sigma_db": (
        "Desviación estándar del sombreado σ en dB (solo canales Lognormal y Suzuki).\n"
        "Valores razonables típicos:\n"
        "• Sombreado leve: ~2.0 dB\n"
        "• Sombreado moderado: 4.0 dB\n"
        "• Sombreado severo: 6.0 dB (recomendado por defecto: 6.0 dB)"
    ),
    "m_nakagami": (
        "Parámetro de forma m >= 0.5 (canales Nakagami-m y Shadowed-Rician).\n"
        "Valores razonables típicos:\n"
        "• m = 0.5: desvanecimiento muy severo (peor que Rayleigh)\n"
        "• m = 1.0: equivale a Rayleigh (en Shadowed-Rician: sombreado fuerte FHS)\n"
        "• m = 1.5–5.0: desvanecimiento moderado / sombreado satelital medio (AS)\n"
        "• m >= 10: sombreado leve (tiende a Rice / AWGN)"
    ),
    "imp_prob": (
        "Probabilidad por símbolo de sufrir un pulso de ruido (solo canal Impulsive).\n"
        "Valores típicos: 0.01 (1% raro) a 0.10 (10% frecuente); recomendado: 0.05."
    ),
    "imp_gamma": (
        "Relación de potencia entre ruido impulsivo y ruido de fondo (solo canal Impulsive).\n"
        "La potencia media total se normaliza a N0.\n"
        "Valores típicos: 10 (moderado) a 100 (severo); recomendado: 50."
    ),
    "eq": (
        "Estrategia de ecualización en el receptor:\n"
        "• mmse: ecualizador MMSE insesgado; incorpora N0 y el error de estimación CSI σ_e² "
        "(recomendado bajo CSI imperfecto).\n"
        "• zf: invierte ingenuamente la ganancia estimada h_hat sin modelar σ_e².\n"
        "• no_eq: sin ecualización (solo disponible para canales AWGN e Impulsive).\n"
        "Nota: En SISO con desvanecimiento plano escalar y CSI perfecto (σ_e² = 0), "
        "MMSE insesgado y ZF producen LLRs blandos idénticos."
    ),
    "csi_err": (
        "Varianza del error de estimación de canal en escala lineal (solo canales con "
        "desvanecimiento): h_hat = h + e_h, con e_h ~ CN(0, σ_e²).\n"
        "• 0.0000: CSI perfecto (h_hat = h; MMSE insesgado y ZF coinciden en SISO)\n"
        "• 0.0100: error leve (-20 dB) | 0.0500: moderado (-13 dB) | 0.1000: severo (-10 dB)\n"
        "Con σ_e² > 0, MMSE incluye σ_e² en el filtro y en N0_eff, mientras ZF invierte h_hat."
    ),
    "imp_demapper": (
        "Estrategia de demapeo APP para el canal Impulsive (Bernoulli-Gaussian):\n"
        "• Mezcla óptima (APP): calcula LLRs exactos de la mezcla de dos componentes "
        "(fondo N0_bg e impulso (1 + Γ)·N0_bg).\n"
        "• Gaussiano desajustado: usa el demapeador gaussiano estándar con potencia total N0 "
        "(útil para comparar la degradación por desajuste)."
    ),
    "scl": (
        "Número de trayectorias candidatas en paralelo para decodificación Polar SCL.\n"
        "Recomendado: 8 (buen equilibrio BER/velocidad; 1 = SC simple, 16–32 = más lento)."
    ),
    "llr": (
        "Límite simétrico [-clip, +clip] aplicado a los LLR del demapeador\n"
        "para evitar desbordamiento o inestabilidad numérica.\n"
        "Recomendado: 500.0."
    ),
    "lote_mc": (
        "Muestras procesadas en paralelo por cada paso Monte Carlo (consume RAM/VRAM).\n"
        "Recomendado: 2000 (reducir si el uso de memoria es elevado)."
    ),
    "num_iter": (
        "Número de lotes secuenciales acumulados por bit-canal (afecta el tiempo de ejecución).\n"
        "Recomendado: 2000 (2000 × 2000 = 4,000,000 bits/bit-canal)."
    ),
    "snr_sim": (
        "Eb/N0 de diseño Monte Carlo [dB] para separar bit-canales ruidosos (congelados) "
        "de los confiables.\n"
        "Pulse '⚡ Auto' para estimarlo según su configuración, o use los valores guía (R = 0.5):\n"
        "• AWGN: -1.2 dB | Impulsive (p=0.05, Γ=50): -1.9 dB\n"
        "• Rayleigh: 1.0 dB | Nakagami (m=1.5): 0.3 dB\n"
        "• Rice (K=6 dB): -0.3 dB | Shadowed-Rician (K=6, m=1.5): 0.5 dB\n"
        "• Lognormal (σ=6 dB): 2.4 dB | Suzuki (σ=6 dB): 5.0 dB"
    ),
    "auto_snr": (
        "Calcula automáticamente el SNR de diseño Monte Carlo [dB] mediante bisección "
        "de información mutua (I(W) ≈ 0.88 × R).\n"
        "Se adapta a tasa de código, modulación, canal, desvanecimiento, ecualizador y recorte LLR."
    ),
    "ebno_min": (
        "Inicio del barrido de Eb/N0 en dB.\n"
        "Valores típicos: -2.0 a 0.0 dB (por defecto: 0.0 dB)."
    ),
    "ebno_max": (
        "Fin del barrido de Eb/N0 en dB.\n"
        "Valores típicos: 6–10 dB para AWGN, 15–25 dB para canales con desvanecimiento "
        "(por defecto: 20.0 dB)."
    ),
    "ebno_rango": (
        "Cantidad de puntos de Eb/N0 equidistantes evaluados en el barrido.\n"
        "Recomendado: 21 (pasos de 1 dB entre 0 y 20 dB)."
    ),
    "lote_ber": (
        "Palabras código simuladas por lote al evaluar las curvas BER/BLER.\n"
        "Recomendado: 200 (aumentar para acelerar en GPU si la memoria lo permite)."
    ),
    "target_errors": (
        "Criterio de parada temprana por punto de Eb/N0: detiene el punto al acumular "
        "al menos esta cantidad de bloques con error.\n"
        "Recomendado: 250 (mayor precisión estadística en colas con valores más altos)."
    ),
    "max_iter_ber": (
        "Límite máximo de lotes evaluados en cada punto de Eb/N0 si no se alcanzan "
        "los errores de bloque objetivo.\n"
        "Si ocurren 0 errores tras el máximo de iteraciones, los puntos superiores se omiten.\n"
        "Por defecto: 100."
    ),
}


def compute_info_bits(N, rate_spec, bitsxsimb=1):
    """Calcula K a partir de N, la tasa seleccionada y bitsxsimb (ajustando múltiplos)."""
    if isinstance(rate_spec, tuple) and len(rate_spec) == 2 and rate_spec[0] == "exact_k":
        return max(1, min(N - 1, int(rate_spec[1])))
    k = int(round(N * float(rate_spec)))
    if bitsxsimb > 1 and N % bitsxsimb == 0 and k % bitsxsimb != 0:
        snapped = int(round(k / bitsxsimb)) * bitsxsimb
        k = max(bitsxsimb, min(N - bitsxsimb, snapped))
    return max(1, min(N - 1, k))


def compute_polarization_weights(N):
    """Calcula el vector de pesos de polarización W_PW(i) con β = 2^(1/4) (3GPP)."""
    n_bits = int(np.log2(N))
    indices = np.arange(N, dtype=np.int64)
    k = np.arange(n_bits, dtype=np.int64)[:, None]
    bits = (indices[None, :] >> k) & 1
    beta_powers = np.power(2.0, np.arange(n_bits, dtype=np.float64) / 4.0)[:, None]
    return np.sum(bits * beta_powers, axis=0)


def rank_polar_bit_channels(ber_bits):
    """Ordena bit-canales por BER ascendente desempatando determinísticamente por mayor W_PW(i)."""
    ber_arr = np.asarray(ber_bits, dtype=np.float64)
    pw_weights = compute_polarization_weights(len(ber_arr))
    return np.lexsort((-pw_weights, ber_arr))


def format_duration(seconds):
    """Formatea una duración en segundos como MM:SS o HH:MM:SS."""
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def extract_mc_backup_meta(params):
    """Extrae los metadatos relevantes de Monte Carlo para validar respaldos .npz."""
    return {k: params[k] for k in MC_BACKUP_KEYS if k in params}


def save_mc_backup(backup_path, ber_bits, next_index, params, error_counts=None, completed_iters=None):
    """Guarda de forma atómica el progreso Monte Carlo junto con conteos y metadatos."""
    tmp_path = f"{backup_path}.tmp.npz"
    meta_json = json.dumps(extract_mc_backup_meta(params), sort_keys=True)
    payload = {
        "ber_bits": np.asarray(ber_bits, dtype=np.float64),
        "next_index": np.int64(next_index),
        "meta_json": np.array(meta_json),
    }
    if error_counts is not None and completed_iters is not None:
        payload["error_counts"] = np.asarray(error_counts, dtype=np.int64)
        payload["completed_iters"] = np.int64(completed_iters)

    np.savez(tmp_path, **payload)
    # Reintentos breves ante bloqueos temporales de archivo en Windows (antivirus/OneDrive).
    for attempt in range(3):
        try:
            os.replace(tmp_path, backup_path)
            return
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(0.05)


def load_mc_backup_state(backup_path):
    """Carga un respaldo Monte Carlo .npz (incluyendo estado vectorizado si existe)."""
    with np.load(backup_path, allow_pickle=False) as data:
        ber_bits = np.array(data["ber_bits"], dtype=np.float64)
        next_index = int(data["next_index"])
        meta = json.loads(str(data["meta_json"]))
        error_counts = (
            np.array(data["error_counts"], dtype=np.int64)
            if "error_counts" in data.files else None
        )
        completed_iters = (
            int(data["completed_iters"])
            if "completed_iters" in data.files else None
        )
    return ber_bits, next_index, meta, error_counts, completed_iters


def load_mc_backup(backup_path):
    """Carga un respaldo Monte Carlo .npz devolviendo la tupla básica (ber_bits, next_index, meta)."""
    ber_bits, next_index, meta, _, _ = load_mc_backup_state(backup_path)
    return ber_bits, next_index, meta


def assess_system_resources(params, mode, has_gpu=HAS_GPU, mem_info=None, cpu_count=None,
                            lock_n=False, lock_mc_params=False, use_legacy=False):
    """Evalúa recursos de CPU/GPU y RAM antes de simular y sugiere ajustes seguros.

    Si ``lock_mc_params`` es True (al reanudar un respaldo), bloquea ``N``, ``lote_mc`` y
    ``num_iter`` para impedir que una recomendación corrompa la escala de conteos acumulados.
    """
    mem = mem_info if mem_info is not None else psutil.virtual_memory()
    avail_gb = mem.available / (1024 ** 3)
    mem_pct = float(mem.percent)
    cores = cpu_count if cpu_count is not None else (os.cpu_count() or 1)

    N = int(params["N"])
    warnings = []
    recommendations = {}

    if mode == "mc":
        lote_mc = int(params["lote_mc"])
        num_iter = int(params["num_iter"])
        total_bit_ops = N * lote_mc * num_iter
        est_batch_gb = (lote_mc * N * 64) / (1024 ** 3)
        allow_resize_n = not (lock_n or lock_mc_params)

        if mem_pct >= 75.0 or avail_gb < 2.0:
            warnings.append(
                f"Memoria RAM disponible limitada ({avail_gb:.1f} GB libres, {mem_pct:.0f}% en uso)."
            )
            if not lock_mc_params:
                recommendations["lote_mc"] = min(lote_mc, 500)
            if allow_resize_n and N > 512:
                recommendations["N"] = 512

        if est_batch_gb > max(0.5, avail_gb * 0.4):
            warnings.append(
                f"El lote paralelo ({lote_mc} × N={N}, ~{est_batch_gb:.2f} GB estimados) "
                "es alto para la RAM libre actual."
            )
            if not lock_mc_params:
                recommendations["lote_mc"] = min(
                    recommendations.get("lote_mc", lote_mc), max(100, min(lote_mc // 2, 1000))
                )
            if allow_resize_n and N > 512:
                recommendations["N"] = min(recommendations.get("N", N), N // 2)

        if not has_gpu and total_bit_ops > 200_000_000:
            if use_legacy:
                warnings.append(
                    f"Ejecutando modo legado en CPU ({cores} hilos, sin GPU NVIDIA): "
                    f"{N} bit-canales × {lote_mc * num_iter:,} muestras puede tardar varias horas."
                )
            else:
                warnings.append(
                    f"Ejecutando en CPU ({cores} hilos, sin GPU NVIDIA): "
                    f"{num_iter:,} lotes × {lote_mc:,} tramas (N={N}, {lote_mc * num_iter:,} muestras/bit-canal) "
                    "puede tomar bastante tiempo."
                )
            if not lock_mc_params:
                recommendations["lote_mc"] = min(recommendations.get("lote_mc", lote_mc), 500)
                recommendations["num_iter"] = min(num_iter, 400)
            if allow_resize_n and N > 256:
                recommendations["N"] = min(recommendations.get("N", N), 256)

    elif mode == "ber":
        lote_ber = int(params["lote_berplot"])
        ebno_pts = int(params["ebno_rango"])
        list_scl = int(params["list_scl"])
        est_ber_gb = (lote_ber * N * list_scl * 32) / (1024 ** 3)

        if mem_pct >= 75.0 or avail_gb < 2.0 or est_ber_gb > max(0.5, avail_gb * 0.4):
            warnings.append(
                f"Memoria RAM disponible limitada ({avail_gb:.1f} GB libres, {mem_pct:.0f}% en uso) "
                f"para lote BER={lote_ber} con SCL={list_scl}."
            )
            recommendations["lote_berplot"] = max(20, min(lote_ber // 2, 100))
            if list_scl > 4:
                recommendations["list_scl"] = 4

        if not has_gpu and (N >= 1024 or ebno_pts > 15 or list_scl >= 8 or lote_ber > 100):
            warnings.append(
                f"Ejecutando en CPU ({cores} hilos, sin GPU NVIDIA): el barrido BER "
                f"({ebno_pts} puntos, N={N}, SCL={list_scl}) puede ser muy lento."
            )
            recommendations["lote_berplot"] = min(
                recommendations.get("lote_berplot", lote_ber), 100
            )
            recommendations["ebno_rango"] = min(ebno_pts, 11)
            recommendations["list_scl"] = min(recommendations.get("list_scl", list_scl), 4)
            if not lock_n and N >= 1024:
                recommendations["N"] = 512

    recommendations = {k: v for k, v in recommendations.items() if v < params.get(k, v)}
    summary = {
        "has_gpu": has_gpu,
        "avail_gb": avail_gb,
        "mem_pct": mem_pct,
        "cores": cores,
    }
    return warnings, recommendations, summary


def default_ber_bits_csv_path(params, n_len=None):
    """Construye la ruta estándar para guardar los resultados CSV de BER_bits."""
    N = int(n_len) if n_len is not None else int(params["N"])
    fecha = datetime.now().strftime("%y%m%d")
    filename = (
        f"{fecha}_{params['canal']}_{params['ecualizador']}_{params['snr_simulacion']}SNR_"
        f"BER_bits_{N}_{params['K']}_{params['num_iter']}iter.csv"
    )
    return os.path.join(f"results_{N}", filename)


def default_ber_curves_csv_path(params):
    """Construye la ruta estándar para exportar las curvas BER/BLER vs Eb/N0 en CSV."""
    N = int(params["N"])
    fecha = datetime.now().strftime("%y%m%d")
    filename = f"{fecha}_{params['canal']}_{params['ecualizador']}_BER_curves_{N}_{params['K']}.csv"
    return os.path.join(f"results_{N}", filename)


# =====================================================================
# 2. MODELOS DE CANAL, CSI IMPERFECTO, ECUALIZADORES Y DEMAPEO (XLA)
# =====================================================================

def _sample_complex_normal(shape, variance):
    """Genera muestras complejas circularmente simétricas CN(0, variance)."""
    stddev = tf.sqrt(tf.cast(variance, tf.float32) / 2.0)
    re = sionna.phy.config.tf_rng.normal(shape, mean=0.0, stddev=stddev, dtype=tf.float32)
    im = sionna.phy.config.tf_rng.normal(shape, mean=0.0, stddev=stddev, dtype=tf.float32)
    return tf.complex(re, im)


def _sample_unit_phasor(shape):
    """Genera fasores complejos de amplitud unitaria y fase uniforme en [0, 2*pi)."""
    phi = sionna.phy.config.tf_rng.uniform(
        shape, minval=0.0, maxval=tf.constant(2.0 * np.pi, tf.float32), dtype=tf.float32
    )
    return tf.complex(tf.cos(phi), tf.sin(phi))


def _sample_lognormal_amplitude(shape, sigma_db):
    """Genera amplitudes lognormales e^z normalizadas a potencia unitaria E[e^(2z)] = 1."""
    sigma_lin = tf.cast(tf.math.log(10.0) / 20.0 * sigma_db, tf.float32)
    mu_ln = -tf.square(sigma_lin)
    z = sionna.phy.config.tf_rng.normal(shape, mean=mu_ln, stddev=sigma_lin, dtype=tf.float32)
    return tf.exp(z)


def _fading_repetitions(n_sym, bloq_coher):
    """Calcula cuántos bloques de coherencia cubren una secuencia de n_sym símbolos."""
    return tf.cast(tf.math.floordiv(n_sym + bloq_coher - 1, bloq_coher), tf.int32)


def _expand_coherence_blocks(h_block, bloq_coher, n_sym):
    """Expande cada bloque de coherencia sobre bloq_coher símbolos y recorta a n_sym."""
    return tf.repeat(h_block, repeats=bloq_coher, axis=1)[:, :n_sym]


def _prepare_fading_context(x, bloq_coher, ebno_db, bitsxsimb, coderate):
    """Prepara dimensiones, bloques de coherencia, varianza N0 y ruido AWGN."""
    batch_size = tf.shape(x)[0]
    n_sym = tf.shape(x)[-1]
    bloq_c = tf.cast(bloq_coher, tf.int32)
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    rep = _fading_repetitions(n_sym, bloq_c)
    awgn = _sample_complex_normal(tf.shape(x), no)
    return batch_size, n_sym, bloq_c, rep, no, awgn


def _sample_gamma_unit_mean(shape, m_param):
    """Genera muestras Gamma(m, m) de media unitaria compatibles con XLA (Marsaglia-Tsang)."""
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
            tf.math.log(u) < (0.5 * z * z + d * (1.0 - v_safe + tf.math.log(v_safe))),
        )
        newly_acc = tf.logical_and(tf.logical_not(acc), accept_cond)
        return tf.logical_or(acc, newly_acc), tf.where(newly_acc, d * v_safe, samp)

    _, samples = tf.while_loop(cond, body, [accepted, samples])

    # Reducción exacta Gamma(m) = Gamma(m + 1) * U^(1/m) cuando 0.5 <= m < 1.
    u_boost = sionna.phy.config.tf_rng.uniform(shape, minval=1e-7, maxval=1.0, dtype=tf.float32)
    samples = tf.where(m_tf < 1.0, samples * tf.pow(u_boost, 1.0 / m_tf), samples)
    return samples / m_tf


@tf.function(jit_compile=True)
def canalAWGN(ebno_db, bitsxsimb, coderate, x):
    """Canal AWGN: ganancia unitaria h = 1 y ruido gaussiano complejo CN(0, N0)."""
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    awgn = _sample_complex_normal(tf.shape(x), no)
    h = tf.complex(1.0, 0.0)
    return x + awgn, h, no


@tf.function(jit_compile=True)
def canalRayleigh(bloq_coher, ebno_db, bitsxsimb, coderate, x):
    """Canal Rayleigh NLOS: h ~ CN(0, 1) constante por bloque de coherencia."""
    batch_size, n_sym, bloq_c, rep, no, awgn = _prepare_fading_context(
        x, bloq_coher, ebno_db, bitsxsimb, coderate
    )
    h_blk = _sample_complex_normal([batch_size, rep], 1.0)
    h = _expand_coherence_blocks(h_blk, bloq_c, n_sym)
    return h * x + awgn, h, no


@tf.function(jit_compile=True)
def canalRice(bloq_coher, k_factor_db, ebno_db, bitsxsimb, coderate, x):
    """Canal Rice: combina componente LOS de fase aleatoria y dispersión NLOS Rayleigh."""
    batch_size, n_sym, bloq_c, rep, no, awgn = _prepare_fading_context(
        x, bloq_coher, ebno_db, bitsxsimb, coderate
    )
    k_lin = tf.pow(10.0, tf.cast(k_factor_db, tf.float32) / 10.0)
    a_los = tf.cast(tf.sqrt(k_lin / (k_lin + 1.0)), tf.complex64)
    b_nlos = tf.cast(tf.sqrt(1.0 / (k_lin + 1.0)), tf.complex64)

    h_los = _sample_unit_phasor([batch_size, rep])
    h_nlos = _sample_complex_normal([batch_size, rep], 1.0)
    h = _expand_coherence_blocks(a_los * h_los + b_nlos * h_nlos, bloq_c, n_sym)
    return h * x + awgn, h, no


@tf.function(jit_compile=True)
def canalLognormal(bloq_coher, sigma_db, ebno_db, bitsxsimb, coderate, x):
    """Canal Log-Normal: sombreado multiplicativo real normalizado a potencia unitaria."""
    batch_size, n_sym, bloq_c, rep, no, awgn = _prepare_fading_context(
        x, bloq_coher, ebno_db, bitsxsimb, coderate
    )
    amp = _sample_lognormal_amplitude([batch_size, rep], sigma_db)
    h_blk = tf.complex(amp, tf.zeros_like(amp))
    h = _expand_coherence_blocks(h_blk, bloq_c, n_sym)
    return h * x + awgn, h, no


@tf.function(jit_compile=True)
def canalRayleighLognormal(bloq_coher, sigma_db, ebno_db, bitsxsimb, coderate, x):
    """Canal Suzuki: desvanecimiento rápido Rayleigh multiplicado por sombreado Log-Normal."""
    batch_size, n_sym, bloq_c, rep, no, awgn = _prepare_fading_context(
        x, bloq_coher, ebno_db, bitsxsimb, coderate
    )
    g = _sample_complex_normal([batch_size, rep], 1.0)
    amp = _sample_lognormal_amplitude([batch_size, rep], sigma_db)
    h = _expand_coherence_blocks(g * tf.complex(amp, tf.zeros_like(amp)), bloq_c, n_sym)
    return h * x + awgn, h, no


@tf.function(jit_compile=True)
def canalNakagami(bloq_coher, m_nakagami, ebno_db, bitsxsimb, coderate, x):
    """Canal Nakagami-m: |h|^2 ~ Gamma(m, m) normalizado a potencia unitaria (m >= 0.5)."""
    batch_size, n_sym, bloq_c, rep, no, awgn = _prepare_fading_context(
        x, bloq_coher, ebno_db, bitsxsimb, coderate
    )
    potencia = _sample_gamma_unit_mean([batch_size, rep], m_nakagami)
    amp = tf.cast(tf.sqrt(tf.maximum(potencia, 1e-12)), tf.complex64)
    h_blk = amp * _sample_unit_phasor([batch_size, rep])
    h = _expand_coherence_blocks(h_blk, bloq_c, n_sym)
    return h * x + awgn, h, no


@tf.function(jit_compile=True)
def canalShadowedRician(bloq_coher, k_factor_db, m_nakagami, ebno_db, bitsxsimb, coderate, x):
    """Canal Shadowed-Rician (Abdi / LMS): LOS con sombreado Nakagami-m + NLOS Rayleigh."""
    batch_size, n_sym, bloq_c, rep, no, awgn = _prepare_fading_context(
        x, bloq_coher, ebno_db, bitsxsimb, coderate
    )
    k_lin = tf.pow(10.0, tf.cast(k_factor_db, tf.float32) / 10.0)
    p_los = k_lin / (k_lin + 1.0)
    p_nlos = 1.0 / (k_lin + 1.0)

    sombreado_los = _sample_gamma_unit_mean([batch_size, rep], m_nakagami)
    amp_los = tf.cast(tf.sqrt(tf.maximum(p_los * sombreado_los, 1e-12)), tf.complex64)
    h_los = amp_los * _sample_unit_phasor([batch_size, rep])
    h_nlos = _sample_complex_normal([batch_size, rep], p_nlos)

    h = _expand_coherence_blocks(h_los + h_nlos, bloq_c, n_sym)
    return h * x + awgn, h, no


@tf.function(jit_compile=True)
def canalImpulsivo(imp_prob, imp_gamma, ebno_db, bitsxsimb, coderate, x):
    """Canal AWGN con ruido impulsivo Bernoulli-Gaussian normalizado a potencia total N0."""
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    p_tf = tf.cast(imp_prob, tf.float32)
    gamma_tf = tf.cast(imp_gamma, tf.float32)

    no_bg = no / (1.0 + p_tf * gamma_tf)
    ruido_bg = _sample_complex_normal(tf.shape(x), no_bg)
    ruido_imp = _sample_complex_normal(tf.shape(x), gamma_tf * no_bg)

    u = sionna.phy.config.tf_rng.uniform(tf.shape(x), minval=0.0, maxval=1.0, dtype=tf.float32)
    mascara = tf.cast(u < p_tf, tf.complex64)
    h = tf.complex(1.0, 0.0)
    return x + ruido_bg + mascara * ruido_imp, h, no


@tf.function(jit_compile=True)
def aplicar_error_csi(h, bloq_coher, csi_err_var):
    """Añade error de estimación e_h ~ CN(0, sigma_e^2) por bloque de coherencia: h_hat = h + e_h."""
    sigma_e_sq = tf.maximum(tf.cast(csi_err_var, tf.float32), 0.0)
    batch_size = tf.shape(h)[0]
    n_sym = tf.shape(h)[-1]
    bloq_c = tf.cast(bloq_coher, tf.int32)
    rep = _fading_repetitions(n_sym, bloq_c)
    e_h = _sample_complex_normal([batch_size, rep], sigma_e_sq)
    return h + _expand_coherence_blocks(e_h, bloq_c, n_sym)


@tf.function(jit_compile=True)
def sin_ecualizar(y, _h, no):
    """Sin ecualización: conserva la señal recibida para canales sin desvanecimiento."""
    return y, no


@tf.function(jit_compile=True)
def zf(y, h, no, eps=1e-12):
    """Ecualizador Zero-Forcing: invierte directamente h_hat (ignora sigma_e^2)."""
    h2 = tf.maximum(tf.math.square(tf.abs(h)), eps)
    x_hat = y * tf.math.conj(h) / tf.cast(h2, tf.complex64)
    no_eff = no / h2
    return x_hat, no_eff


@tf.function(jit_compile=True)
def mmse(y, h, no, csi_err_var=0.0, eps=1e-12):
    """Ecualizador MMSE insesgado: incorpora ruido térmico N0 y varianza de error CSI sigma_e^2."""
    sigma_e_sq = tf.maximum(tf.cast(csi_err_var, tf.float32), 0.0)
    h2 = tf.math.square(tf.abs(h))
    g = tf.math.conj(h) / tf.cast(h2 + sigma_e_sq + no, tf.complex64)
    d_abs = tf.maximum(tf.abs(g * h), eps)
    x_hat = g * y / tf.cast(d_abs, tf.complex64)
    no_eff = (no + sigma_e_sq) / tf.maximum(h2, eps)
    return x_hat, no_eff


def build_constellation_bit_masks(num_bits_per_symbol):
    """Construye máscaras booleanas [bitsxsimb, M] para bits 1 y 0 (orden MSB-first Sionna)."""
    num_points = 1 << num_bits_per_symbol
    indices = np.arange(num_points, dtype=np.int32)
    shifts = np.arange(num_bits_per_symbol - 1, -1, -1, dtype=np.int32)[:, None]
    bits_table = ((indices[None, :] >> shifts) & 1).astype(bool)
    return tf.constant(bits_table, dtype=tf.bool), tf.constant(~bits_table, dtype=tf.bool)


@tf.function(jit_compile=True)
def demap_impulsive_mixture(y, constellation_points, bit_masks_1, bit_masks_0, no_bg, imp_prob, imp_gamma):
    """Demapeador APP exacto para mezcla Bernoulli-Gaussian (convención Sionna: LLR > 0 => b = 1)."""
    eps = tf.constant(1e-12, dtype=tf.float32)
    p_tf = tf.clip_by_value(tf.cast(imp_prob, tf.float32), 0.0, 1.0)
    gamma_tf = tf.maximum(tf.cast(imp_gamma, tf.float32), 0.0)
    sigma0_sq = tf.maximum(tf.cast(no_bg, tf.float32), eps)
    sigma1_sq = sigma0_sq * (1.0 + gamma_tf)

    if sigma0_sq.shape.rank is not None and sigma0_sq.shape.rank > 0:
        sigma0_sq = tf.expand_dims(sigma0_sq, axis=-1)
        sigma1_sq = tf.expand_dims(sigma1_sq, axis=-1)

    pts = tf.cast(constellation_points, tf.complex64)
    diff = tf.expand_dims(tf.cast(y, tf.complex64), axis=-1) - pts
    dist_sq = tf.math.square(tf.math.real(diff)) + tf.math.square(tf.math.imag(diff))

    # Log-verosimilitudes de la mezcla conservando los términos de normalización -ln(sigma_k^2).
    log_c0 = tf.math.log(1.0 - p_tf) - tf.math.log(sigma0_sq) - dist_sq / sigma0_sq
    log_c1 = tf.math.log(p_tf) - tf.math.log(sigma1_sq) - dist_sq / sigma1_sq
    log_p_y_s = tf.math.reduce_logsumexp(tf.stack([log_c0, log_c1], axis=-1), axis=-1)

    log_p_exp = tf.expand_dims(log_p_y_s, axis=-2)
    neg_inf = tf.constant(-np.inf, dtype=tf.float32)
    log_p_1 = tf.where(bit_masks_1, log_p_exp, neg_inf)
    log_p_0 = tf.where(bit_masks_0, log_p_exp, neg_inf)
    llr_sym = tf.math.reduce_logsumexp(log_p_1, axis=-1) - tf.math.reduce_logsumexp(log_p_0, axis=-1)
    # Protección ante casos extremos de subdesbordamiento (-inf - (-inf) -> NaN).
    llr_sym = tf.where(tf.math.is_finite(llr_sym), llr_sym, tf.zeros_like(llr_sym))

    num_bits_per_sym = tf.shape(bit_masks_1)[0]
    out_shape = tf.concat([tf.shape(y)[:-1], [tf.shape(y)[-1] * num_bits_per_sym]], axis=0)
    return tf.reshape(llr_sym, out_shape)


def _dispatch_channel(x, ebno_db, coderate, params):
    """Despacha la señal x por el modelo de canal seleccionado en params."""
    c = params["canal"]
    bits = params["bitsxsimb"]
    if c == "awgn":
        return canalAWGN(ebno_db, bits, coderate, x)
    if c == "rayleigh":
        return canalRayleigh(params["bloq_coher"], ebno_db, bits, coderate, x)
    if c == "rice":
        return canalRice(params["bloq_coher"], params["k_factor_db"], ebno_db, bits, coderate, x)
    if c == "lognormal":
        return canalLognormal(params["bloq_coher"], params["sigma_db"], ebno_db, bits, coderate, x)
    if c == "suzuki":
        return canalRayleighLognormal(params["bloq_coher"], params["sigma_db"], ebno_db, bits, coderate, x)
    if c == "nakagami":
        return canalNakagami(params["bloq_coher"], params["m_nakagami"], ebno_db, bits, coderate, x)
    if c == "shadowed_rician":
        return canalShadowedRician(
            params["bloq_coher"], params["k_factor_db"], params["m_nakagami"],
            ebno_db, bits, coderate, x,
        )
    if c == "impulsive":
        return canalImpulsivo(params["imp_prob"], params["imp_gamma"], ebno_db, bits, coderate, x)
    raise ValueError(f"Canal inválido: {c}")


def apply_channel_and_equalizer(x, ebno_db, coderate, params, common_blocks):
    """Aplica el modelo de canal, error CSI, ecualización, demapeo con recorte y desintercalado."""
    c = params["canal"]
    y, h, no = _dispatch_channel(x, ebno_db, coderate, params)

    csi_err = float(params.get("csi_err_var", 0.0)) if c not in NON_FADING_CHANNELS else 0.0
    h_obs = (
        aplicar_error_csi(h, params["bloq_coher"], tf.constant(csi_err, tf.float32))
        if csi_err > 0.0 else h
    )

    eq = params["ecualizador"]
    if eq == "no_eq":
        x_hat, no_eff = sin_ecualizar(y, h_obs, no)
    elif eq == "zf":
        x_hat, no_eff = zf(y, h_obs, no)
    elif eq == "mmse":
        x_hat, no_eff = mmse(y, h_obs, no, csi_err_var=tf.constant(csi_err, tf.float32))
    else:
        raise ValueError(f"Ecualizador inválido: {eq}")

    if c == "impulsive" and params.get("imp_demapper", "mixture") == "mixture":
        p_tf = tf.cast(params["imp_prob"], tf.float32)
        gamma_tf = tf.cast(params["imp_gamma"], tf.float32)
        no_bg = tf.cast(no_eff, tf.float32) / (1.0 + p_tf * gamma_tf)
        raw_llr = demap_impulsive_mixture(
            x_hat,
            common_blocks["constellation_points"],
            common_blocks["bit_masks_1"],
            common_blocks["bit_masks_0"],
            no_bg,
            p_tf,
            gamma_tf,
        )
    else:
        raw_llr = common_blocks["demapeador"](x_hat, no_eff)

    llr_clip = float(params["LLR_clip"])
    llr_prev = tf.clip_by_value(raw_llr, -llr_clip, llr_clip)
    return common_blocks["deinterleaver"](llr_prev)


# =====================================================================
# 3. PRIMITIVAS POLAR XLA (MARIPOSA Y GENIE-AIDED SC)
# =====================================================================

def _boxplus_exact(a, b):
    """Operador box-plus exacto en dominio logarítmico (corrección de Jacobi sin desbordamiento)."""
    min_sum = tf.sign(a) * tf.sign(b) * tf.minimum(tf.abs(a), tf.abs(b))
    jacobi = tf.math.softplus(-tf.abs(a + b)) - tf.math.softplus(-tf.abs(a - b))
    return min_sum + jacobi


@tf.function(jit_compile=True)
def polar_encode_butterfly(u_bits):
    """Codifica u_bits [batch, N] -> x_bits = u * G_N mod 2 mediante mariposa XOR O(N log N)."""
    x = tf.cast(u_bits, tf.uint8)
    batch_size = tf.shape(x)[0]
    n_len = u_bits.shape[-1]
    stride = 1
    while stride < n_len:
        x_blk = tf.reshape(x, [batch_size, n_len // (2 * stride), 2, stride])
        left = tf.bitwise.bitwise_xor(x_blk[:, :, 0, :], x_blk[:, :, 1, :])
        right = x_blk[:, :, 1, :]
        x = tf.reshape(tf.stack([left, right], axis=2), [batch_size, n_len])
        stride *= 2
    return tf.cast(x, tf.float32)


@tf.function(jit_compile=True)
def polar_genie_sc_errors(llr_sionna, u_bits):
    """Motor SC vectorizado asistido por genio en una sola pasada (retorna errores [N] por lote)."""
    u_true = tf.cast(u_bits, tf.uint8)
    batch_size = tf.shape(u_true)[0]
    n_len = u_bits.shape[-1]

    # Precalcula las sumas parciales verdaderas (el genio) para cada etapa de la mariposa.
    stages = [u_true]
    cur = u_true
    stride = 1
    while stride < n_len:
        cur_blk = tf.reshape(cur, [batch_size, n_len // (2 * stride), 2, stride])
        left = tf.bitwise.bitwise_xor(cur_blk[:, :, 0, :], cur_blk[:, :, 1, :])
        right = cur_blk[:, :, 1, :]
        cur = tf.reshape(tf.stack([left, right], axis=2), [batch_size, n_len])
        stages.append(cur)
        stride *= 2

    # Convierte LLR de Sionna ln(P(1)/P(0)) a convención de Arıkan ln(P(0)/P(1)).
    llr = -tf.cast(llr_sionna, tf.float32)
    num_stages = len(stages) - 1
    for stage_idx in range(num_stages - 1, -1, -1):
        stage_stride = 1 << stage_idx
        num_blocks = n_len // (2 * stage_stride)
        llr_blk = tf.reshape(llr, [batch_size, num_blocks, 2, stage_stride])
        a = llr_blk[:, :, 0, :]
        b = llr_blk[:, :, 1, :]
        u_left = tf.cast(
            tf.reshape(stages[stage_idx], [batch_size, num_blocks, 2, stage_stride])[:, :, 0, :],
            tf.float32,
        )
        llr_upper = _boxplus_exact(a, b)
        llr_lower = (1.0 - 2.0 * u_left) * a + b
        llr = tf.reshape(tf.stack([llr_upper, llr_lower], axis=2), [batch_size, n_len])

    u_hat = tf.cast(llr < 0.0, tf.uint8)
    return tf.math.count_nonzero(tf.not_equal(u_hat, u_true), axis=0, dtype=tf.int64)


# =====================================================================
# 4. BLOQUES DEL SISTEMA SIONNA
# =====================================================================

class BaseSystemBlock(sionna.phy.Block):
    """Base común para canal, error CSI, ecualización y desintercalado de todos los sistemas."""

    def __init__(self, params, common_blocks):
        super().__init__()
        self.p = params
        self.cb = common_blocks

    def apply_channel_and_eq(self, x, ebno_db, coderate):
        return apply_channel_and_equalizer(x, ebno_db, coderate, self.p, self.cb)


class SinCodigo(BaseSystemBlock):
    """Sistema de referencia sin código de canal (R = 1.0)."""

    def call(self, batch_size, ebno_db):
        b = self.cb["datos"]([batch_size, self.p["K"]])
        x = self.cb["mapeador"](self.cb["interleaver"](b))
        llr = self.apply_channel_and_eq(x, ebno_db, coderate=1.0)
        return b, llr


class CodigoPolar(BaseSystemBlock):
    """Sistema con código Polar y decodificación SCL."""

    def __init__(self, params, common_blocks, bits_congelados):
        super().__init__(params, common_blocks)
        self.polar_codif = sionna.phy.fec.polar.encoding.PolarEncoder(bits_congelados, self.p["N"])
        self.polar_decodif = sionna.phy.fec.polar.decoding.PolarSCLDecoder(
            bits_congelados, self.p["N"], list_size=self.p["list_scl"]
        )

    def call(self, batch_size, ebno_db):
        b = self.cb["datos"]([batch_size, self.p["K"]])
        x = self.cb["mapeador"](self.cb["interleaver"](self.polar_codif(b)))
        llr = self.apply_channel_and_eq(x, ebno_db, coderate=self.p["K"] / self.p["N"])
        return b, self.polar_decodif(llr)


class CodigoPolarCRC(BaseSystemBlock):
    """Sistema con código Polar y verificación CRC 3GPP 5G NR (CA-SCL)."""

    def __init__(self, params, common_blocks, bits_congelados):
        super().__init__(params, common_blocks)
        self.polar_codif = sionna.phy.fec.polar.encoding.PolarEncoder(bits_congelados, self.p["N"])
        self.polar_decodif = sionna.phy.fec.polar.decoding.PolarSCLDecoder(
            bits_congelados,
            self.p["N"],
            list_size=self.p["list_scl"],
            crc_degree=self.p["crc_degree"],
        )

    def call(self, batch_size, ebno_db):
        k_efec = self.p["K"] - self.cb["crc_long"]
        b = self.cb["datos"]([batch_size, k_efec])
        bits_codif = self.polar_codif(self.cb["crc_codif"](b))
        x = self.cb["mapeador"](self.cb["interleaver"](bits_codif))
        llr = self.apply_channel_and_eq(x, ebno_db, coderate=k_efec / self.p["N"])
        b_hat = self.polar_decodif(llr)[..., :k_efec]
        return b, b_hat


class CodigoLDPC(BaseSystemBlock):
    """Sistema con código LDPC 5G NR y decodificación Belief Propagation."""

    def __init__(self, params, common_blocks):
        super().__init__(params, common_blocks)
        self.ldpc_codif = sionna.phy.fec.ldpc.encoding.LDPC5GEncoder(
            self.p["K"], self.p["N"], self.p["bitsxsimb"]
        )
        self.ldpc_decodif = sionna.phy.fec.ldpc.decoding.LDPC5GDecoder(self.ldpc_codif)

    def call(self, batch_size, ebno_db):
        b = self.cb["datos"]([batch_size, self.p["K"]])
        x = self.cb["mapeador"](self.cb["interleaver"](self.ldpc_codif(b)))
        llr = self.apply_channel_and_eq(x, ebno_db, coderate=self.p["K"] / self.p["N"])
        return b, self.ldpc_decodif(llr)


def build_common_blocks(p):
    """Crea los bloques compartidos de generación binaria, constelación, intercalado y CRC."""
    sionna.phy.config.seed = p["seed"]
    modulacion = "pam" if p["bitsxsimb"] == 1 else "qam"
    datos = sionna.phy.mapping.BinarySource()
    constelacion = sionna.phy.mapping.Constellation(modulacion, p["bitsxsimb"])
    constellation_points = tf.cast(constelacion.points, tf.complex64)
    bit_masks_1, bit_masks_0 = build_constellation_bit_masks(p["bitsxsimb"])
    mapeador = sionna.phy.mapping.Mapper(constellation=constelacion)
    demapeador = sionna.phy.mapping.Demapper("app", constellation=constelacion)
    interleaver = sionna.phy.fec.interleaving.RandomInterleaver(seed=p["seed"], axis=-1)
    deinterleaver = sionna.phy.fec.interleaving.Deinterleaver(interleaver)
    crc_codif = sionna.phy.fec.crc.CRCEncoder(crc_degree=p["crc_degree"])
    return {
        "datos": datos,
        "constelacion": constelacion,
        "constellation_points": constellation_points,
        "bit_masks_1": bit_masks_1,
        "bit_masks_0": bit_masks_0,
        "mapeador": mapeador,
        "demapeador": demapeador,
        "interleaver": interleaver,
        "deinterleaver": deinterleaver,
        "crc_codif": crc_codif,
        "crc_long": crc_codif.crc_length,
    }


# =====================================================================
# 5. HILOS DE TRABAJO EN SEGUNDO PLANO (QTHREAD)
# =====================================================================

class MonteCarloWorker(QThread):
    """Estima la confiabilidad BER de los N bit-canales Polar mediante Monte Carlo."""

    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, float, float)  # actual, total, transcurrido, eta
    finished_signal = pyqtSignal(np.ndarray, str)
    stopped_signal = pyqtSignal(int, int, str)
    error_signal = pyqtSignal(str, str)  # mensaje_corto, detalle_traceback

    def __init__(self, params, resume_backup=False, use_legacy_loop=False):
        super().__init__()
        self.p = dict(params)
        self.resume_backup = resume_backup
        self.use_legacy_loop = use_legacy_loop
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def _init_ber_array(self, backup_file, N):
        """Inicializa o recupera el vector BER_bits para el modo legado por bit-canal."""
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

    def _init_vectorized_state(self, backup_file, N, num_iter):
        """Inicializa o recupera conteos de error acumulados para el motor Genie-Aided SC."""
        error_counts = np.zeros(N, dtype=np.int64)
        start_iter = 0
        if self.resume_backup and os.path.exists(backup_file):
            _, _, saved_meta, loaded_counts, saved_iters = load_mc_backup_state(backup_file)
            saved_lote = saved_meta.get("lote_mc") if isinstance(saved_meta, dict) else None
            lote_compatible = (saved_lote is None) or (int(saved_lote) == int(self.p["lote_mc"]))
            if (
                loaded_counts is not None
                and saved_iters is not None
                and lote_compatible
                and len(loaded_counts) == N
                and 0 <= saved_iters <= num_iter
            ):
                error_counts = loaded_counts
                start_iter = saved_iters
                if start_iter < num_iter:
                    self.log_signal.emit(
                        f"Respaldo vectorizado recuperado. Reanudando desde iteración {start_iter + 1}/{num_iter}."
                    )
                else:
                    self.log_signal.emit(
                        f"Respaldo completo recuperado ({num_iter}/{num_iter} iteraciones). Finalizando..."
                    )
                self.progress_signal.emit(start_iter, num_iter, 0.0, 0.0)
        elif not self.resume_backup and os.path.exists(backup_file):
            os.remove(backup_file)
        return error_counts, start_iter

    def _offset_rng_for_resume(self, step_offset):
        """Desplaza determinísticamente la semilla del RNG al reanudar para no repetir lotes previos."""
        if step_offset > 0:
            # El interleaver ya fue construido con self.p["seed"]; solo se avanza el RNG de canal/datos.
            sionna.phy.config.seed = int(self.p["seed"]) + int(step_offset) * MC_RESUME_SEED_STRIDE

    def _simulate_bit_channel(self, bit_idx, N, cb, snr_sim, coderate_sim):
        """Simula un único bit-canal Polar (modo legado para respaldos antiguos)."""
        lote_mc = self.p["lote_mc"]
        num_iter = self.p["num_iter"]
        bits_congelados = np.array([j for j in range(N) if j != bit_idx])
        codificador_temp = sionna.phy.fec.polar.encoding.PolarEncoder(bits_congelados, N)
        decodificador_temp = sionna.phy.fec.polar.decoding.PolarSCLDecoder(bits_congelados, N, list_size=1)

        # Grafo TF estándar sin XLA anidado por bit-canal para prevenir fugas de memoria.
        @tf.function
        def simulacion_batch_size():
            bits = cb["datos"]([lote_mc, 1])
            x = cb["mapeador"](cb["interleaver"](codificador_temp(bits)))
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

    def _run_legacy_loop(self, cb, N, snr_sim, coderate_sim, backup_file):
        """Ejecuta el bucle legado bit-canal por bit-canal (para reanudar respaldos previos)."""
        BER_bits, start_index = self._init_ber_array(backup_file, N)
        self._offset_rng_for_resume(start_index)
        run_start_time = time.monotonic()

        for i in range(start_index, N):
            if self._stop_requested:
                if i > 0:
                    save_mc_backup(backup_file, BER_bits, i, self.p)
                self.log_signal.emit(
                    f"⚠️ Simulación detenida por el usuario antes del bit-canal {i + 1}/{N}."
                    + (" Respaldo guardado." if i > 0 else "")
                )
                self.stopped_signal.emit(i, N, backup_file if i > 0 else "")
                return None

            ber_val = self._simulate_bit_channel(i, N, cb, snr_sim, coderate_sim)
            if ber_val is None:
                if i > 0:
                    save_mc_backup(backup_file, BER_bits, i, self.p)
                self.log_signal.emit(
                    f"⚠️ Simulación detenida por el usuario durante el bit-canal {i + 1}/{N}."
                    + (" Respaldo guardado." if i > 0 else "")
                )
                self.stopped_signal.emit(i, N, backup_file if i > 0 else "")
                return None

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
                    "",
                )
                return None

        return BER_bits

    def _run_vectorized_genie_sc(self, cb, N, snr_sim, coderate_sim, backup_file):
        """Motor principal: SC vectorizado asistido por genio en una sola pasada sobre los N bit-canales."""
        lote_mc = int(self.p["lote_mc"])
        num_iter = int(self.p["num_iter"])
        error_counts, start_iter = self._init_vectorized_state(backup_file, N, num_iter)
        self._offset_rng_for_resume(start_iter)

        run_start_time = time.monotonic()
        completed_iters = start_iter
        log_interval = max(MC_BACKUP_INTERVAL, num_iter // 20)

        @tf.function
        def simulacion_genie_batch():
            u_bits = cb["datos"]([lote_mc, N])
            x_bits = polar_encode_butterfly(u_bits)
            x = cb["mapeador"](cb["interleaver"](x_bits))
            llr = apply_channel_and_equalizer(x, snr_sim, coderate_sim, self.p, cb)
            return polar_genie_sc_errors(llr, u_bits)

        self.log_signal.emit(
            f"⚡ Motor SC vectorizado asistido por genio activo "
            f"({num_iter} lotes × {lote_mc} tramas = {num_iter * lote_mc:,} muestras/bit-canal)."
        )

        stopped = False
        try:
            while completed_iters < num_iter:
                paso = min(MC_BACKUP_INTERVAL, num_iter - completed_iters)
                err_chunk = tf.zeros([N], dtype=tf.int64)
                iters_in_chunk = 0
                for _ in range(paso):
                    if self._stop_requested:
                        stopped = True
                        break
                    err_chunk = err_chunk + simulacion_genie_batch()
                    iters_in_chunk += 1

                if iters_in_chunk > 0:
                    error_counts += err_chunk.numpy()
                    completed_iters += iters_in_chunk

                ber_est = (
                    error_counts / float(completed_iters * lote_mc)
                    if completed_iters > 0 else np.full(N, np.nan, dtype=np.float64)
                )

                if stopped:
                    if completed_iters > 0:
                        save_mc_backup(
                            backup_file, ber_est, completed_iters, self.p,
                            error_counts=error_counts, completed_iters=completed_iters,
                        )
                    self.log_signal.emit(
                        f"⚠️ Simulación detenida por el usuario en iteración {completed_iters}/{num_iter}."
                        + (" Respaldo guardado." if completed_iters > 0 else "")
                    )
                    self.stopped_signal.emit(
                        completed_iters, num_iter, backup_file if completed_iters > 0 else ""
                    )
                    return None

                session_done = completed_iters - start_iter
                elapsed = time.monotonic() - run_start_time
                eta = (elapsed / session_done) * (num_iter - completed_iters) if session_done > 0 else 0.0
                self.progress_signal.emit(completed_iters, num_iter, elapsed, eta)

                if (
                    completed_iters == num_iter
                    or completed_iters == paso
                    or completed_iters % log_interval < MC_BACKUP_INTERVAL
                ):
                    self.log_signal.emit(
                        f"Iteración {completed_iters}/{num_iter} | "
                        f"BER medio: {float(np.mean(ber_est)):.3e} | "
                        f"BER mín/máx: {float(np.min(ber_est)):.2e}/{float(np.max(ber_est)):.2e}"
                    )

                save_mc_backup(
                    backup_file, ber_est, completed_iters, self.p,
                    error_counts=error_counts, completed_iters=completed_iters,
                )

                mem = psutil.virtual_memory()
                if mem.percent > MC_RAM_LIMIT_PERCENT:
                    self.error_signal.emit(
                        f"El uso de RAM superó el {MC_RAM_LIMIT_PERCENT:.0f}% ({mem.percent:.1f}%). "
                        f"Respaldo de emergencia guardado en iteración {completed_iters}/{num_iter}.",
                        "",
                    )
                    return None
        finally:
            del simulacion_genie_batch
            gc.collect()

        return error_counts / float(num_iter * lote_mc)

    def run(self):
        try:
            cb = build_common_blocks(self.p)
            N = self.p["N"]
            snr_sim = tf.constant(self.p["snr_simulacion"], dtype=tf.float32)
            coderate_sim = float(self.p["K"] / N)
            backup_file = f"BER_bits_backup_{N}.npz"

            if self.use_legacy_loop:
                BER_bits = self._run_legacy_loop(cb, N, snr_sim, coderate_sim, backup_file)
            else:
                BER_bits = self._run_vectorized_genie_sc(cb, N, snr_sim, coderate_sim, backup_file)

            if BER_bits is None:
                return
            if not np.all(np.isfinite(BER_bits)):
                raise RuntimeError("Monte Carlo finalizó con valores BER incompletos o no finitos.")

            filename = default_ber_bits_csv_path(self.p, N)
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            pd.DataFrame({"BER bits": BER_bits}).to_csv(filename, index=True, encoding="utf-8-sig")

            if os.path.exists(backup_file):
                os.remove(backup_file)

            self.finished_signal.emit(BER_bits, filename)

        except Exception as exc:
            self.error_signal.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())


class BERSimWorker(QThread):
    """Ejecuta el barrido BER/BLER frente a Eb/N0 punto por punto con parada temprana y cancelación."""

    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, str)
    curve_ready_signal = pyqtSignal(str, np.ndarray, np.ndarray, np.ndarray)
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

    def _run_curve(self, model, ebno_dbs, label, curve_idx, total_curves,
                   target_errors, max_iter, soft_estimates=False, log_name=None):
        """Evalúa una curva completa de BER/BLER sobre ebno_dbs y emite el resultado al terminar."""
        if self._stop_requested:
            self.stopped_signal.emit()
            return None

        display_name = log_name or label
        num_points = len(ebno_dbs)
        total_work = total_curves * num_points
        batch_size_tf = tf.constant(int(self.p["lote_berplot"]), dtype=tf.int32)

        self.log_signal.emit(
            f"Ejecutando simulación BER/BLER: {display_name} "
            f"(objetivo={target_errors} bloques con error, máx={max_iter} lotes/punto)..."
        )

        @tf.function
        def eval_batch(ebno_tf):
            b, b_out = model(batch_size_tf, ebno_tf)
            b_hat = tf.cast(b_out > 0.0, b.dtype) if soft_estimates else tf.cast(b_out, b.dtype)
            neq = tf.not_equal(b, b_hat)
            bit_err = tf.math.count_nonzero(neq, dtype=tf.int64)
            total_bits = tf.size(b, out_type=tf.int64)
            block_err = tf.math.count_nonzero(tf.reduce_any(neq, axis=-1), dtype=tf.int64)
            total_blocks = tf.cast(tf.shape(b)[0], tf.int64)
            return bit_err, total_bits, block_err, total_blocks

        ber_vals = np.zeros(num_points, dtype=np.float64)
        bler_vals = np.zeros(num_points, dtype=np.float64)
        stopped = False

        try:
            for pt_idx, ebno_val in enumerate(ebno_dbs):
                if self._stop_requested:
                    stopped = True
                    break

                ebno_tf = tf.constant(float(ebno_val), dtype=tf.float32)
                bit_errors = total_bits = block_errors = total_blocks = 0

                for _ in range(max_iter):
                    if self._stop_requested:
                        stopped = True
                        break
                    be, tb, ble, tbl = eval_batch(ebno_tf)
                    bit_errors += int(be.numpy())
                    total_bits += int(tb.numpy())
                    block_errors += int(ble.numpy())
                    total_blocks += int(tbl.numpy())
                    if self._stop_requested:
                        stopped = True
                        break
                    if block_errors >= target_errors:
                        break

                if stopped:
                    break

                ber_vals[pt_idx] = bit_errors / total_bits if total_bits > 0 else 0.0
                bler_vals[pt_idx] = block_errors / total_blocks if total_blocks > 0 else 0.0

                done_points = curve_idx * num_points + (pt_idx + 1)
                self.progress_signal.emit(
                    done_points,
                    total_work,
                    f"{display_name}: punto {pt_idx + 1}/{num_points} (Eb/N0={ebno_val:.1f} dB)",
                )

                if block_errors == 0 and bit_errors == 0:
                    remaining = num_points - (pt_idx + 1)
                    if remaining > 0:
                        ber_vals[pt_idx + 1:] = 0.0
                        bler_vals[pt_idx + 1:] = 0.0
                        self.log_signal.emit(
                            f"ℹ️ {display_name}: 0 errores tras {max_iter} lotes en Eb/N0 = {ebno_val:.2f} dB; "
                            f"finalizando barrido anticipadamente ({remaining} punto(s) superiores omitidos)."
                        )
                        self.progress_signal.emit(
                            (curve_idx + 1) * num_points,
                            total_work,
                            f"Completado anticipadamente: {display_name}",
                        )
                    break
        finally:
            del eval_batch
            gc.collect()

        if stopped:
            self.stopped_signal.emit()
            return None

        self.curve_ready_signal.emit(label, ebno_dbs, ber_vals, bler_vals)
        self.progress_signal.emit((curve_idx + 1) * num_points, total_work, f"Completado: {label}")
        return curve_idx + 1

    def run(self):
        try:
            cb = build_common_blocks(self.p)
            ebno_dbs = np.linspace(self.p["ebno_min"], self.p["ebno_max"], self.p["ebno_rango"])
            target_errors = int(self.p.get("target_block_errors", 250))
            max_iter = int(self.p.get("max_iter_ber", 100))

            has_valid_frozen_bits = (
                self.ber_bits is not None
                and len(self.ber_bits) == self.p["N"]
                and np.all(np.isfinite(self.ber_bits))
            )
            run_polar_crc = has_valid_frozen_bits and (self.p["K"] > cb["crc_long"])
            total_curves = 2 + (1 if has_valid_frozen_bits else 0) + (1 if run_polar_crc else 0)
            curve_idx = 0

            # 1. Línea base sin código de canal
            curve_idx = self._run_curve(
                SinCodigo(self.p, cb), ebno_dbs, CURVE_UNCODED,
                curve_idx, total_curves, target_errors=target_errors,
                max_iter=max_iter, soft_estimates=True,
            )
            if curve_idx is None:
                return

            # 2 y 3. Sistemas Polar y Polar + CRC
            if has_valid_frozen_bits:
                zero_ber_count = int(np.count_nonzero(self.ber_bits == 0.0))
                if zero_ber_count > self.p["K"]:
                    self.log_signal.emit(
                        f"⚠️ Advertencia: {zero_ber_count}/{self.p['N']} bit-canales registraron BER = 0.0 "
                        f"(supera K = {self.p['K']}), lo que indica un SNR de diseño demasiado alto. "
                        "Los empates se resuelven determinísticamente con Polarization Weight (β = 2^(1/4))."
                    )
                indices_ordenados = rank_polar_bit_channels(self.ber_bits)
                bits_congelados = indices_ordenados[-(self.p["N"] - self.p["K"]):]

                curve_idx = self._run_curve(
                    CodigoPolar(self.p, cb, bits_congelados), ebno_dbs, CURVE_POLAR,
                    curve_idx, total_curves, target_errors=target_errors,
                    max_iter=max_iter, log_name=f"{CURVE_POLAR} (SCL)",
                )
                if curve_idx is None:
                    return

                if run_polar_crc:
                    crc_label = f"{CURVE_POLAR} + {self.p['crc_degree']}"
                    curve_idx = self._run_curve(
                        CodigoPolarCRC(self.p, cb, bits_congelados), ebno_dbs, crc_label,
                        curve_idx, total_curves, target_errors=target_errors, max_iter=max_iter,
                    )
                    if curve_idx is None:
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

            # 4. Código LDPC 5G NR (aislado por si (K, N) cae fuera del estándar 5G NR)
            if self._stop_requested:
                self.stopped_signal.emit()
                return
            try:
                codigo_ldpc = CodigoLDPC(self.p, cb)
            except (ValueError, AssertionError) as ldpc_err:
                self.log_signal.emit(
                    f"⚠️ Omitiendo LDPC 5G NR para (K={self.p['K']}, N={self.p['N']}): {ldpc_err}"
                )
                self.progress_signal.emit(
                    (curve_idx + 1) * len(ebno_dbs), total_curves * len(ebno_dbs), f"Omitido: {CURVE_LDPC}"
                )
            else:
                curve_idx = self._run_curve(
                    codigo_ldpc, ebno_dbs, CURVE_LDPC,
                    curve_idx, total_curves, target_errors=target_errors, max_iter=max_iter,
                )
                if curve_idx is None:
                    return

            self.finished_signal.emit()

        except Exception as exc:
            self.error_signal.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())


class DesignSNRWorker(QThread):
    """Estima automáticamente el SNR de diseño [dB] mediante bisección de información mutua I(W)."""

    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int, float, float)  # paso, total, snr_db, mi
    finished_signal = pyqtSignal(float, float, float, float)  # snr_db, mi_lograda, mi_objetivo, ber_est
    stopped_signal = pyqtSignal()
    error_signal = pyqtSignal(str, str)  # mensaje_corto, detalle_traceback

    def __init__(self, params, target_alpha=AUTO_SNR_TARGET_ALPHA, num_steps=AUTO_SNR_BISECTION_STEPS):
        super().__init__()
        self.p = dict(params)
        self.target_alpha = float(target_alpha)
        self.num_steps = int(num_steps)
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def _compute_batch_plan(self, N):
        """Calcula (batch_size, num_batches) asegurando suficiente diversidad de desvanecimiento."""
        batch_size = max(64, min(2048, 65536 // N))
        if self.p["canal"] in NON_FADING_CHANNELS:
            return batch_size, 4

        n_sym = max(1, N // self.p["bitsxsimb"])
        bloq_coher = max(1, int(self.p["bloq_coher"]))
        blocks_per_frame = max(1, (n_sym + bloq_coher - 1) // bloq_coher)
        fading_per_batch = batch_size * blocks_per_frame
        num_batches = max(
            4,
            min(64, (AUTO_SNR_MIN_FADING_BLOCKS + fading_per_batch - 1) // fading_per_batch),
        )
        return batch_size, num_batches

    def run(self):
        eval_batch = None
        try:
            cb = build_common_blocks(self.p)
            N = self.p["N"]
            coderate = float(self.p["K"] / N)
            target_mi = float(np.clip(self.target_alpha * coderate, 0.02, 0.98))
            batch_size, num_batches = self._compute_batch_plan(N)
            crn_seed = int(self.p["seed"])

            @tf.function
            def eval_batch(ebno_db_tf):
                b = cb["datos"]([batch_size, N])
                x = cb["mapeador"](cb["interleaver"](b))
                llr = apply_channel_and_equalizer(x, ebno_db_tf, coderate, self.p, cb)
                # En Sionna, LLR = ln(P(b=1|y)/P(b=0|y)), por lo que (2b - 1)*LLR > 0 en aciertos.
                aligned_llr = (2.0 * tf.cast(b, tf.float32) - 1.0) * tf.cast(llr, tf.float32)
                mi = 1.0 - tf.reduce_mean(tf.math.softplus(-aligned_llr) / tf.math.log(2.0))
                b_hat = tf.cast(llr > 0.0, b.dtype)
                ber = tf.reduce_mean(tf.cast(tf.not_equal(b, b_hat), tf.float32))
                return mi, ber

            def eval_snr(snr_db):
                # Common Random Numbers (CRN): garantiza monotonicidad en la misma trayectoria muestral.
                sionna.phy.config.seed = crn_seed
                ebno_tf = tf.constant(float(snr_db), dtype=tf.float32)
                mi_sum = ber_sum = 0.0
                for _ in range(num_batches):
                    if self._stop_requested:
                        return None, None
                    mi_t, ber_t = eval_batch(ebno_tf)
                    mi_sum += float(mi_t.numpy())
                    ber_sum += float(ber_t.numpy())
                return float(np.clip(mi_sum / num_batches, 0.0, 1.0)), ber_sum / num_batches

            lo, hi = -20.0, 40.0
            mi_lo, ber_lo = eval_snr(lo)
            if mi_lo is None:
                self.stopped_signal.emit()
                return
            mi_hi, ber_hi = eval_snr(hi)
            if mi_hi is None:
                self.stopped_signal.emit()
                return

            if mi_lo >= target_mi:
                self.log_signal.emit(
                    f"⚠️ Advertencia: La raíz de SNR de diseño está por debajo de {lo:.1f} dB "
                    f"(I(W) en {lo:.1f} dB = {mi_lo:.4f} >= objetivo {target_mi:.4f})."
                )
                self.finished_signal.emit(round(lo, 2), mi_lo, target_mi, ber_lo)
                return

            if mi_hi <= target_mi:
                self.log_signal.emit(
                    f"⚠️ Advertencia: La raíz de SNR de diseño está por encima de {hi:.1f} dB "
                    f"o es inalcanzable (I(W) en {hi:.1f} dB = {mi_hi:.4f} <= objetivo {target_mi:.4f})."
                )
                self.finished_signal.emit(round(hi, 2), mi_hi, target_mi, ber_hi)
                return

            best_snr, best_mi, best_ber = 0.0, 0.0, 0.5
            for step in range(1, self.num_steps + 1):
                if self._stop_requested:
                    self.stopped_signal.emit()
                    return
                mid = 0.5 * (lo + hi)
                mi_val, ber_val = eval_snr(mid)
                if mi_val is None:
                    self.stopped_signal.emit()
                    return
                best_snr, best_mi, best_ber = mid, mi_val, ber_val

                self.progress_signal.emit(step, self.num_steps, mid, mi_val)
                if mi_val < target_mi:
                    lo = mid
                else:
                    hi = mid

            if abs(best_mi - target_mi) > 0.03:
                self.log_signal.emit(
                    f"⚠️ Advertencia: I(W) convergido ({best_mi:.4f}) difiere en más de 0.03 "
                    f"del objetivo ({target_mi:.4f}); verifique el rango [-20, 40] dB."
                )
            self.finished_signal.emit(round(best_snr, 2), best_mi, target_mi, best_ber)

        except Exception as exc:
            self.error_signal.emit(f"{type(exc).__name__}: {exc}", traceback.format_exc())
        finally:
            if eval_batch is not None:
                del eval_batch
            gc.collect()


# =====================================================================
# 6. VENTANA PRINCIPAL PYQT6
# =====================================================================

class PolarSimGUI(QMainWindow):
    """Ventana principal con controles de simulación, progreso, registros y gráfica BER/BLER."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simulador Sionna de Códigos Polar/LDPC y Diseñador Monte Carlo")
        self.resize(1380, 880)

        self.ber_bits = None
        self.ber_bits_source = ""
        self.mc_worker = None
        self.ber_worker = None
        self.snr_worker = None
        self._pending_close = False
        self.active_mc_params = None
        self.active_ber_params = None
        self.ber_curves_ebno = None
        self.ber_curves_data = {}
        self.bler_curves_data = {}
        self.plot_metric = "BER"
        self.plot_min_ber = BER_MIN_YLIM

        self.init_ui()
        self.init_status_bar()
        self.sync_initial_ui_state()

    # -----------------------------------------------------------------
    # CONSTRUCCIÓN DE LA INTERFAZ
    # -----------------------------------------------------------------

    @staticmethod
    def _add_row_with_tip(form_layout, label_text, widget, tooltip=None):
        """Agrega una fila de formulario asignando el mismo tooltip a la etiqueta y al control."""
        label = QLabel(label_text)
        if tooltip:
            label.setToolTip(tooltip)
            widget.setToolTip(tooltip)
        form_layout.addRow(label, widget)

    @staticmethod
    def _make_spinbox(min_val, max_val, value, step=1):
        """Crea un QSpinBox configurado en una sola llamada."""
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    @staticmethod
    def _make_double_spinbox(min_val, max_val, value, step=1.0, decimals=2):
        """Crea un QDoubleSpinBox configurado en una sola llamada."""
        spin = QDoubleSpinBox()
        spin.setDecimals(decimals)
        spin.setRange(min_val, max_val)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    @staticmethod
    def _styled_button(text, bg_color, callback):
        """Crea un botón destacado con estilo consistente para estados activo y deshabilitado."""
        btn = QPushButton(text)
        btn.setStyleSheet(
            f"QPushButton {{ background-color: {bg_color}; color: white; font-weight: bold; padding: 8px; }}"
            "QPushButton:disabled { background-color: #9aa5b1; color: #eceff1; }"
        )
        btn.clicked.connect(callback)
        return btn

    def init_ui(self):
        """Construye la estructura principal dividida en parámetros (izquierda) y resultados (derecha)."""
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

    def _build_basic_group(self):
        """Grupo 1: parámetros básicos, longitud N, tasa de código, modulación y CRC."""
        grp = QGroupBox("1. Parámetros Básicos y del Código")
        form = QFormLayout(grp)

        self.spin_seed = self._make_spinbox(0, 999999, 0)

        self.combo_N = QComboBox()
        for exp in range(5, 14):  # 32 a 8192
            self.combo_N.addItem(str(2 ** exp), 2 ** exp)
        self.combo_N.setCurrentText("1024")
        self.combo_N.currentIndexChanged.connect(self.on_N_changed)

        self.combo_rate = QComboBox()
        for label, rate_val in PREDEFINED_CODE_RATES:
            self.combo_rate.addItem(label, rate_val)
        self.combo_rate.setCurrentText("1/2 (0.500)")
        self.combo_rate.currentIndexChanged.connect(self.update_code_rate_label)

        self.combo_bits = QComboBox()
        for label, val in (
            ("1 (BPSK - PAM)", 1),
            ("2 (QPSK - QAM)", 2),
            ("4 (16-QAM)", 4),
            ("6 (64-QAM)", 6),
        ):
            self.combo_bits.addItem(label, val)
        self.combo_bits.currentIndexChanged.connect(self.update_code_rate_label)

        self.combo_crc = QComboBox()
        for crc in CRC_LENGTHS:
            self.combo_crc.addItem(crc)
        self.combo_crc.currentTextChanged.connect(self.update_code_rate_label)

        self.lbl_code_rate = QLabel()
        self.lbl_code_rate.setStyleSheet("color: #495057; font-size: 11px;")

        self._add_row_with_tip(form, "Semilla aleatoria:", self.spin_seed, TOOLTIPS["seed"])
        self._add_row_with_tip(form, "Longitud de código (N):", self.combo_N, TOOLTIPS["N"])
        self._add_row_with_tip(form, "Tasa de código (R = K/N):", self.combo_rate, TOOLTIPS["rate"])
        self._add_row_with_tip(form, "Bits por símbolo:", self.combo_bits, TOOLTIPS["bits"])
        self._add_row_with_tip(form, "Polinomio CRC:", self.combo_crc, TOOLTIPS["crc"])
        form.addRow("Bits de información (K):", self.lbl_code_rate)
        return grp

    def _build_channel_group(self):
        """Grupo 2: selección de canal inalámbrico y parámetros estadísticos físicos."""
        grp = QGroupBox("2. Parámetros del Canal")
        form = QFormLayout(grp)

        self.combo_canal = QComboBox()
        for c in AVAILABLE_CHANNELS:
            self.combo_canal.addItem(c)
        self.combo_canal.currentTextChanged.connect(self.on_channel_changed)

        self.spin_coher = self._make_spinbox(1, 8192, 64)

        self.chk_flat_fading = QCheckBox("Desvanecimiento plano (bloque = N)")
        self.chk_flat_fading.setChecked(False)
        self.chk_flat_fading.setToolTip(TOOLTIPS["flat_fading"])
        self.chk_flat_fading.toggled.connect(self.on_flat_fading_toggled)

        self.spin_k_db = self._make_double_spinbox(-20.0, 50.0, 6.0)
        self.spin_sigma_db = self._make_double_spinbox(0.1, 30.0, 6.0)
        self.spin_m_nakagami = self._make_double_spinbox(0.5, 20.0, 1.5, step=0.5)
        self.spin_imp_prob = self._make_double_spinbox(0.001, 0.500, 0.050, step=0.01, decimals=3)
        self.spin_imp_gamma = self._make_double_spinbox(1.0, 1000.0, 50.0, step=5.0)

        self._add_row_with_tip(form, "Modelo de canal:", self.combo_canal, TOOLTIPS["canal"])
        self._add_row_with_tip(form, "Bloque de coherencia:", self.spin_coher, TOOLTIPS["coher"])
        form.addRow("", self.chk_flat_fading)
        self._add_row_with_tip(form, "Factor K de Rice [dB]:", self.spin_k_db, TOOLTIPS["k_db"])
        self._add_row_with_tip(form, "σ Lognormal/Suzuki [dB]:", self.spin_sigma_db, TOOLTIPS["sigma_db"])
        self._add_row_with_tip(form, "Parámetro Nakagami m:", self.spin_m_nakagami, TOOLTIPS["m_nakagami"])
        self._add_row_with_tip(form, "Prob. impulso p:", self.spin_imp_prob, TOOLTIPS["imp_prob"])
        self._add_row_with_tip(form, "Razón impulsiva Γ:", self.spin_imp_gamma, TOOLTIPS["imp_gamma"])
        return grp

    def _build_receiver_group(self):
        """Grupo 3: ecualizador, error CSI, estrategia de demapeo impulsivo, lista SCL y LLR clip."""
        grp = QGroupBox("3. Parámetros del Receptor")
        form = QFormLayout(grp)

        self.combo_eq = QComboBox()
        for eq in ("mmse", "zf", "no_eq"):
            self.combo_eq.addItem(eq)

        self.spin_csi_err = self._make_double_spinbox(0.0, 1.0, 0.0, step=0.01, decimals=4)

        self.combo_imp_demapper = QComboBox()
        self.combo_imp_demapper.addItem("Mezcla óptima (APP)", "mixture")
        self.combo_imp_demapper.addItem("Gaussiano desajustado", "gaussian")

        self.combo_scl = QComboBox()
        for list_size in (1, 2, 4, 8, 16, 32):
            self.combo_scl.addItem(str(list_size), list_size)
        self.combo_scl.setCurrentText("8")

        self.spin_llr = self._make_double_spinbox(10.0, 5000.0, 500.0)

        self._add_row_with_tip(form, "Ecualizador:", self.combo_eq, TOOLTIPS["eq"])
        self._add_row_with_tip(form, "Error CSI σ_e²:", self.spin_csi_err, TOOLTIPS["csi_err"])
        self._add_row_with_tip(form, "Demapeo impulsivo:", self.combo_imp_demapper, TOOLTIPS["imp_demapper"])
        self._add_row_with_tip(form, "Tamaño de lista SCL:", self.combo_scl, TOOLTIPS["scl"])
        self._add_row_with_tip(form, "Recorte de LLR:", self.spin_llr, TOOLTIPS["llr"])
        return grp

    def _build_mc_group(self):
        """Grupo 4: generación Monte Carlo de bits congelados y estimador Auto-SNR."""
        grp = QGroupBox("4. Diseño Polar por Monte Carlo")
        form = QFormLayout(grp)

        self.spin_lote_mc = self._make_spinbox(10, 50000, 2000, step=500)
        self.spin_lote_mc.valueChanged.connect(self.update_mc_summary_label)

        self.spin_num_iter = self._make_spinbox(1, 50000, 2000, step=500)
        self.spin_num_iter.valueChanged.connect(self.update_mc_summary_label)

        self.spin_snr_sim = self._make_double_spinbox(-20.0, 40.0, 5.0, step=0.5)
        self.spin_snr_sim.setToolTip(TOOLTIPS["snr_sim"])

        self.btn_auto_snr = QPushButton("⚡ Auto")
        self.btn_auto_snr.setFixedWidth(72)
        self.btn_auto_snr.setToolTip(TOOLTIPS["auto_snr"])
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

        self._add_row_with_tip(form, "Lote en paralelo:", self.spin_lote_mc, TOOLTIPS["lote_mc"])
        self._add_row_with_tip(form, "Iteraciones secuenciales:", self.spin_num_iter, TOOLTIPS["num_iter"])
        lbl_snr_sim = QLabel("SNR de diseño [dB]:")
        lbl_snr_sim.setToolTip(TOOLTIPS["snr_sim"])
        form.addRow(lbl_snr_sim, snr_row_widget)
        form.addRow("Muestras totales:", self.lbl_mc_total_bits)
        form.addRow("Bits congelados:", self.lbl_mc_status)
        return grp

    def _build_ber_group(self):
        """Grupo 5: parámetros del barrido BER/BLER vs Eb/N0."""
        grp = QGroupBox("5. Parámetros de Gráfica BER/BLER vs Eb/N0")
        form = QFormLayout(grp)

        self.spin_ebno_min = self._make_double_spinbox(-20.0, 50.0, 0.0)
        self.spin_ebno_max = self._make_double_spinbox(-20.0, 50.0, 20.0)
        self.spin_ebno_rango = self._make_spinbox(2, 100, 21)
        self.spin_lote_ber = self._make_spinbox(10, 10000, 200)
        self.spin_target_block_errors = self._make_spinbox(10, 10000, 250, step=50)
        self.spin_max_iter_ber = self._make_spinbox(1, 100000, 100, step=25)

        self._add_row_with_tip(form, "Eb/N0 mín [dB]:", self.spin_ebno_min, TOOLTIPS["ebno_min"])
        self._add_row_with_tip(form, "Eb/N0 máx [dB]:", self.spin_ebno_max, TOOLTIPS["ebno_max"])
        self._add_row_with_tip(form, "Número de puntos:", self.spin_ebno_rango, TOOLTIPS["ebno_rango"])
        self._add_row_with_tip(form, "Tamaño de lote BER:", self.spin_lote_ber, TOOLTIPS["lote_ber"])
        self._add_row_with_tip(form, "Errores de bloque objetivo:", self.spin_target_block_errors, TOOLTIPS["target_errors"])
        self._add_row_with_tip(form, "Máx. iteraciones por punto:", self.spin_max_iter_ber, TOOLTIPS["max_iter_ber"])
        return grp

    def _build_action_buttons(self):
        """Construye los botones de ejecución, detención y carga/exportación CSV."""
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

        self.btn_save_ber_csv = QPushButton("💾 Guardar CSV Curvas BER/BLER...")
        self.btn_save_ber_csv.setEnabled(False)
        self.btn_save_ber_csv.setToolTip(
            "Exportar las curvas BER y BLER vs Eb/N0 simuladas a un nuevo archivo CSV."
        )
        self.btn_save_ber_csv.clicked.connect(self.save_ber_curves_csv)

        btn_layout.addWidget(self.btn_run_mc)
        btn_layout.addWidget(self.btn_stop_mc)
        btn_layout.addLayout(mc_csv_row)
        btn_layout.addWidget(self.btn_run_ber)
        btn_layout.addWidget(self.btn_stop_ber)
        btn_layout.addWidget(self.btn_save_ber_csv)
        return btn_layout

    def _build_right_panel(self):
        """Panel derecho: gráfica BER/BLER, selector de métrica, progreso y consola de registros."""
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.setup_plot_axes()

        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        toolbar_row = QHBoxLayout()
        toolbar_row.addWidget(self.toolbar, stretch=1)
        self.combo_plot_metric = QComboBox()
        self.combo_plot_metric.addItem("BER", "BER")
        self.combo_plot_metric.addItem("BLER", "BLER")
        self.combo_plot_metric.setToolTip(
            "Alterna la gráfica entre BER (tasa de error de bit) y BLER (tasa de error de bloque)."
        )
        self.combo_plot_metric.currentIndexChanged.connect(self.on_plot_metric_changed)
        toolbar_row.addWidget(QLabel("Métrica:"))
        toolbar_row.addWidget(self.combo_plot_metric)

        right_layout.addLayout(toolbar_row)
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

    def init_status_bar(self):
        """Configura la barra de estado con dispositivo de cómputo y monitor periódico de RAM."""
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

    def sync_initial_ui_state(self):
        """Sincroniza límites iniciales y habilitación de campos al arrancar la aplicación."""
        self.on_N_changed()
        self.on_channel_changed(self.combo_canal.currentText())
        self.update_code_rate_label()
        self.update_mc_summary_label()
        self.update_mc_status_label()

    # -----------------------------------------------------------------
    # EVENTOS, ESTADO DE CONTROLES Y GRÁFICA BER/BLER
    # -----------------------------------------------------------------

    def update_ram_indicator(self):
        """Actualiza el indicador de uso de memoria RAM en la barra de estado."""
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

    def get_current_K(self):
        """Devuelve los bits de información K según N, tasa y modulación actuales."""
        N = self.combo_N.currentData()
        rate_spec = self.combo_rate.currentData()
        bits = self.combo_bits.currentData()
        if N is None or rate_spec is None or bits is None:
            return 1
        return compute_info_bits(N, rate_spec, bits)

    def select_rate_for_K(self, N, target_K, bitsxsimb):
        """Selecciona una tasa predefinida equivalente a K/N o crea una entrada personalizada."""
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

    def update_code_rate_label(self):
        """Actualiza la etiqueta con los bits de información K y la tasa efectiva con CRC."""
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

    def update_mc_summary_label(self):
        """Actualiza el resumen de muestras totales por bit-canal y resolución mínima BER."""
        total_bits = self.spin_lote_mc.value() * self.spin_num_iter.value()
        min_ber = 1.0 / total_bits
        self.lbl_mc_total_bits.setText(f"{total_bits:,} bits/canal (BER mín ≈ {min_ber:.1e})")

    def update_mc_status_label(self):
        """Muestra el estado de los bits congelados cargados y alerta si su longitud difiere de N."""
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

    def setup_plot_axes(self, title=None, ylabel=None):
        """Configura los ejes logarítmicos de la gráfica BER/BLER."""
        metric = ylabel or self.plot_metric
        plot_title = title or f"{metric} vs Eb/N0"
        self.plot_min_ber = BER_MIN_YLIM
        self.ax.clear()
        self.ax.set_yscale("log")
        self.ax.set_xlabel(r"$E_b/N_0$ [dB]", fontsize=11)
        self.ax.set_ylabel(metric, fontsize=11)
        self.ax.set_title(plot_title, fontsize=12, fontweight="bold")
        self.ax.set_ylim([self.plot_min_ber, 1.0])
        self.ax.grid(True, which="both", linestyle="--", alpha=0.6)
        self.fig.tight_layout()

    def _current_plot_title(self, metric=None):
        """Construye el título de la gráfica según la métrica activa y los parámetros simulados."""
        m = metric or self.plot_metric
        if self.active_ber_params:
            p = self.active_ber_params
            return (
                f"Comparación {m} vs Eb/N0 — Canal {p['canal'].capitalize()} "
                f"(N={p['N']}, K={p['K']}, Eq={p['ecualizador'].upper()})"
            )
        return f"{m} vs Eb/N0"

    def on_plot_metric_changed(self):
        """Alterna la visualización entre las curvas BER y BLER almacenadas."""
        self.plot_metric = self.combo_plot_metric.currentData() or "BER"
        self.redraw_curves()

    def redraw_curves(self):
        """Redibuja todas las curvas almacenadas para la métrica activa (BER o BLER)."""
        metric = self.plot_metric
        self.setup_plot_axes(title=self._current_plot_title(metric), ylabel=metric)
        if self.active_ber_params:
            self.ax.set_xlim([self.active_ber_params["ebno_min"], self.active_ber_params["ebno_max"]])

        curves_dict = self.ber_curves_data if metric == "BER" else self.bler_curves_data
        if self.ber_curves_ebno is not None and curves_dict:
            crc_name = (
                self.active_ber_params["crc_degree"]
                if self.active_ber_params
                else self.combo_crc.currentText()
            )
            markers = dict(CURVE_MARKERS)
            markers[f"{CURVE_POLAR} + {crc_name}"] = "^-"
            any_plotted = False

            for label, values in curves_dict.items():
                valid_mask = np.isfinite(values) & (values > 0.0)
                if not np.any(valid_mask):
                    continue
                min_pos = float(np.min(values[valid_mask]))
                if min_pos < self.plot_min_ber:
                    decade = 10.0 ** np.floor(np.log10(min_pos))
                    self.plot_min_ber = max(BER_FLOOR_YLIM, decade)
                    self.ax.set_ylim([self.plot_min_ber, 1.0])
                self.ax.plot(
                    self.ber_curves_ebno[valid_mask],
                    values[valid_mask],
                    markers.get(label, "d-"),
                    linewidth=2,
                    markersize=6,
                    label=label,
                )
                any_plotted = True

            if any_plotted:
                self.ax.legend(loc="best", fontsize=10)

        self.fig.tight_layout()
        self.canvas.draw()

    def log(self, message):
        """Añade una línea con marca de tiempo a la consola de simulación."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_console.appendPlainText(f"[{timestamp}] {message}")
        scrollbar = self.log_console.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def clear_log(self):
        self.log_console.clear()

    def set_simulation_running(self, running, mode=None):
        """Habilita o bloquea controles según el estado de ejecución activo."""
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
        """Actualiza bloque de coherencia, etiqueta de tasa y estado de bits congelados al cambiar N."""
        N = self.combo_N.currentData()
        if N is None:
            return
        self.spin_coher.setMaximum(N)
        self.spin_coher.setValue(N if self.chk_flat_fading.isChecked() else max(1, N // 16))
        self.update_code_rate_label()
        self.update_mc_status_label()

    def on_flat_fading_toggled(self, checked):
        """Alterna entre bloque de coherencia N // 16 y desvanecimiento plano (bloque = N)."""
        N = self.combo_N.currentData()
        if N is not None:
            self.spin_coher.setValue(N if checked else max(1, N // 16))
        is_fading_channel = self.combo_canal.currentText() not in NON_FADING_CHANNELS
        self.spin_coher.setEnabled(is_fading_channel and not checked)

    def on_channel_changed(self, channel_name):
        """Actualiza SNR guía, parámetros físicos habilitados y ecualizadores válidos para el canal."""
        if channel_name in CHANNEL_SNR_DEFAULTS:
            self.spin_snr_sim.setValue(CHANNEL_SNR_DEFAULTS[channel_name])

        is_fading = channel_name not in NON_FADING_CHANNELS
        self.chk_flat_fading.setEnabled(is_fading)
        self.spin_coher.setEnabled(is_fading and not self.chk_flat_fading.isChecked())
        self.spin_k_db.setEnabled(channel_name in ("rice", "shadowed_rician"))
        self.spin_sigma_db.setEnabled(channel_name in ("lognormal", "suzuki"))
        self.spin_m_nakagami.setEnabled(channel_name in ("nakagami", "shadowed_rician"))
        self.spin_imp_prob.setEnabled(channel_name == "impulsive")
        self.spin_imp_gamma.setEnabled(channel_name == "impulsive")
        self.combo_imp_demapper.setEnabled(channel_name == "impulsive")
        self.spin_csi_err.setEnabled(is_fading)
        self.update_equalizer_options(channel_name)

    def update_equalizer_options(self, channel_name):
        """Permite 'no_eq' únicamente en canales sin desvanecimiento multiplicativo."""
        no_eq_index = self.combo_eq.findText("no_eq")
        no_eq_enabled = channel_name in NON_FADING_CHANNELS
        model = self.combo_eq.model()
        if isinstance(model, QStandardItemModel) and no_eq_index >= 0:
            item = model.item(no_eq_index)
            if item is not None:
                item.setEnabled(no_eq_enabled)

        if not no_eq_enabled and self.combo_eq.currentText() == "no_eq":
            self.combo_eq.setCurrentText("mmse")

    def gather_parameters(self):
        """Reúne en un diccionario consistente todos los parámetros vigentes de la interfaz."""
        if self.combo_canal.currentText() not in NON_FADING_CHANNELS and self.combo_eq.currentText() == "no_eq":
            self.combo_eq.setCurrentText("mmse")
        return {
            "seed": self.spin_seed.value(),
            "N": self.combo_N.currentData(),
            "K": self.get_current_K(),
            "bitsxsimb": self.combo_bits.currentData(),
            "crc_degree": self.combo_crc.currentText(),
            "canal": self.combo_canal.currentText(),
            "bloq_coher": self.spin_coher.value(),
            "k_factor_db": self.spin_k_db.value(),
            "sigma_db": self.spin_sigma_db.value(),
            "m_nakagami": self.spin_m_nakagami.value(),
            "imp_prob": self.spin_imp_prob.value(),
            "imp_gamma": self.spin_imp_gamma.value(),
            "imp_demapper": self.combo_imp_demapper.currentData(),
            "ecualizador": self.combo_eq.currentText(),
            "csi_err_var": self.spin_csi_err.value(),
            "list_scl": self.combo_scl.currentData(),
            "LLR_clip": self.spin_llr.value(),
            "lote_mc": self.spin_lote_mc.value(),
            "num_iter": self.spin_num_iter.value(),
            "snr_simulacion": self.spin_snr_sim.value(),
            "ebno_min": self.spin_ebno_min.value(),
            "ebno_max": self.spin_ebno_max.value(),
            "ebno_rango": self.spin_ebno_rango.value(),
            "lote_berplot": self.spin_lote_ber.value(),
            "target_block_errors": self.spin_target_block_errors.value(),
            "max_iter_ber": self.spin_max_iter_ber.value(),
        }

    def validate_parameters(self, params, mode):
        """Valida restricciones algebraicas, estadísticas y de Sionna antes de ejecutar un hilo."""
        errors = []
        N = params["N"]
        K = params["K"]
        bits = params["bitsxsimb"]

        if not (N > 0 and (N & (N - 1)) == 0):
            errors.append(f"La longitud N ({N}) debe ser una potencia de 2 para códigos Polar.")
        if not (1 <= K < N):
            errors.append(f"Los bits de información K ({K}) deben cumplir 1 <= K < N ({N}).")
        if N % bits != 0:
            errors.append(
                f"La longitud de código N ({N}) no es divisible por Bits por símbolo ({bits}). "
                "Nota: 64-QAM (6 bits/símbolo) es incompatible con longitudes Polar potencias de 2."
            )
        if params["canal"] not in NON_FADING_CHANNELS and not (1 <= params["bloq_coher"] <= N):
            errors.append(f"El bloque de coherencia ({params['bloq_coher']}) debe estar entre 1 y N ({N}).")
        if float(params.get("m_nakagami", 1.5)) < 0.5:
            errors.append("El parámetro Nakagami m debe cumplir m >= 0.5.")
        if not (0.0 < float(params.get("imp_prob", 0.05)) < 1.0):
            errors.append("La probabilidad de impulso p debe estar en el intervalo (0, 1).")
        if float(params.get("imp_gamma", 50.0)) < 0.0:
            errors.append("La razón impulsiva Γ debe ser >= 0.")
        if float(params.get("csi_err_var", 0.0)) < 0.0:
            errors.append("La varianza de error CSI σ_e² debe ser >= 0.")

        if mode == "ber":
            if K % bits != 0:
                errors.append(
                    f"Para la curva de referencia Sin código, K ({K}) debe ser divisible por Bits por símbolo ({bits})."
                )
            if params["ebno_min"] >= params["ebno_max"]:
                errors.append(
                    f"Eb/N0 mín ({params['ebno_min']} dB) debe ser estrictamente menor que "
                    f"Eb/N0 máx ({params['ebno_max']} dB)."
                )
            if params["ebno_rango"] < 2:
                errors.append("El número de puntos de Eb/N0 debe ser al menos 2.")

        if errors:
            QMessageBox.warning(
                self,
                "Parámetros de Simulación Inválidos",
                "Corrija los siguientes problemas antes de ejecutar:\n\n• " + "\n• ".join(errors),
            )
            return False
        return True

    # -----------------------------------------------------------------
    # GESTIÓN DE RESPALDOS .NPZ Y ARCHIVOS CSV
    # -----------------------------------------------------------------

    def is_backup_meta_applicable(self, loaded_bits, saved_index, saved_meta,
                                  is_vectorized=False, completed_iters=None):
        """Verifica si los metadatos y progreso de un respaldo .npz son aplicables a la interfaz."""
        if not isinstance(saved_meta, dict) or not all(k in saved_meta for k in MC_BACKUP_CORE_KEYS):
            return False
        try:
            backup_N = int(saved_meta["N"])
            backup_K = int(saved_meta["K"])
            backup_bits = int(saved_meta["bitsxsimb"])
            backup_coher = int(saved_meta["bloq_coher"])
            backup_canal = str(saved_meta["canal"])
            backup_eq = str(saved_meta["ecualizador"])
            backup_num_iter = int(saved_meta.get("num_iter", completed_iters or 1))
        except (TypeError, ValueError):
            return False

        progress_ok = (
            (completed_iters is not None and 0 <= completed_iters <= backup_num_iter)
            if is_vectorized else (0 <= saved_index <= backup_N)
        )
        coher_ok = backup_canal in NON_FADING_CHANNELS or (1 <= backup_coher <= backup_N)

        return (
            len(loaded_bits) == backup_N
            and progress_ok
            and coher_ok
            and 1 <= backup_K < backup_N
            and self.combo_N.findData(backup_N) != -1
            and self.combo_bits.findData(backup_bits) != -1
            and self.combo_canal.findText(backup_canal) != -1
            and self.combo_eq.findText(backup_eq) != -1
        )

    def apply_backup_meta_to_ui(self, saved_meta):
        """Aplica en orden seguro los parámetros guardados en un respaldo .npz sobre la interfaz."""
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

        # Se restauran tras cambiar N y canal para no sobrescribirlos con valores por defecto.
        backup_coher = int(saved_meta["bloq_coher"])
        self.chk_flat_fading.setChecked(backup_coher == backup_N)
        self.spin_coher.setValue(backup_coher)
        self.spin_k_db.setValue(float(saved_meta["k_factor_db"]))
        self.spin_sigma_db.setValue(float(saved_meta["sigma_db"]))
        self.spin_m_nakagami.setValue(float(saved_meta.get("m_nakagami", 1.5)))
        self.spin_imp_prob.setValue(float(saved_meta.get("imp_prob", 0.05)))
        self.spin_imp_gamma.setValue(float(saved_meta.get("imp_gamma", 50.0)))
        imp_demap_idx = self.combo_imp_demapper.findData(str(saved_meta.get("imp_demapper", "mixture")))
        if imp_demap_idx >= 0:
            self.combo_imp_demapper.setCurrentIndex(imp_demap_idx)

        eq_idx = self.combo_eq.findText(str(saved_meta["ecualizador"]))
        if eq_idx >= 0:
            self.combo_eq.setCurrentIndex(eq_idx)

        self.spin_csi_err.setValue(float(saved_meta.get("csi_err_var", 0.0)))
        self.spin_llr.setValue(float(saved_meta["LLR_clip"]))
        self.spin_snr_sim.setValue(float(saved_meta["snr_simulacion"]))
        if "seed" in saved_meta:
            self.spin_seed.setValue(int(saved_meta["seed"]))
        if "lote_mc" in saved_meta:
            self.spin_lote_mc.setValue(int(saved_meta["lote_mc"]))
        if "num_iter" in saved_meta:
            self.spin_num_iter.setValue(int(saved_meta["num_iter"]))

        self.update_code_rate_label()
        self.update_mc_summary_label()
        self.update_mc_status_label()

    @staticmethod
    def _find_mc_backup_file(N):
        """Busca primero el respaldo del N actual y, si no existe, el .npz más reciente de otro N."""
        exact_path = f"BER_bits_backup_{N}.npz"
        if os.path.exists(exact_path):
            return exact_path, True
        other_backups = sorted(glob.glob("BER_bits_backup_*.npz"), key=os.path.getmtime, reverse=True)
        if other_backups:
            return other_backups[0], False
        return None, True

    def _prompt_matching_backup(self, backup_file, progress_label):
        """Diálogo cuando el respaldo coincide exactamente con la configuración actual."""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("¿Reanudar Respaldo Monte Carlo?")
        msg.setText(
            f"Se encontró un respaldo compatible '{backup_file}' ({progress_label}).\n"
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

    @staticmethod
    def _backup_meta_matches(saved_meta, current_meta):
        """Compara metadatos de respaldo de forma retrocompatible con respaldos de 10 claves."""
        if not isinstance(saved_meta, dict):
            return False
        if not all(k in saved_meta for k in MC_BACKUP_CORE_KEYS):
            return False
        keys_to_check = [k for k in MC_BACKUP_KEYS if k in saved_meta]
        return all(saved_meta.get(k) == current_meta.get(k) for k in keys_to_check)

    def _prompt_mismatched_backup(
        self, backup_file, same_N_backup, loaded_bits, saved_index,
        saved_meta, current_meta, progress_label, is_vectorized=False, completed_iters=None,
    ):
        """Diálogo cuando el respaldo difiere de la configuración actual."""
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
        can_apply = self.is_backup_meta_applicable(
            loaded_bits, saved_index, saved_meta,
            is_vectorized=is_vectorized, completed_iters=completed_iters,
        )
        fresh_label = "Descartar y empezar de cero" if same_N_backup else "Ignorar respaldo y empezar de cero"

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Diferencia en Parámetros del Respaldo")
        msg.setText(
            f"El respaldo existente '{backup_file}' ({progress_label}) "
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

    def resolve_mc_backup_action(self, params):
        """Detecta respaldos Monte Carlo previos y pregunta al usuario cómo proceder."""
        N = params["N"]
        backup_file, same_N_backup = self._find_mc_backup_file(N)
        if backup_file is not None:
            try:
                loaded_bits, saved_index, saved_meta, loaded_counts, completed_iters = (
                    load_mc_backup_state(backup_file)
                )
            except Exception as exc:
                if not same_N_backup:
                    return ("fresh", None, None, False)
                reply = QMessageBox.warning(
                    self,
                    "Respaldo Dañado Detectado",
                    f"No se pudo leer el archivo de respaldo '{backup_file}' ({exc}).\n\n"
                    "¿Descartarlo e iniciar desde cero?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                )
                if reply == QMessageBox.StandardButton.Yes:
                    return ("fresh", backup_file, None, False)
                return ("cancel", None, None, False)

            is_vectorized = loaded_counts is not None and completed_iters is not None
            saved_total_iters = int(saved_meta.get("num_iter", params["num_iter"]))
            progress_label = (
                f"iteración {completed_iters}/{saved_total_iters}"
                if is_vectorized else f"bit-canal {saved_index}/{len(loaded_bits)}"
            )
            valid_progress = (
                (0 <= completed_iters <= params["num_iter"])
                if is_vectorized else (0 <= saved_index <= N)
            )
            current_meta = extract_mc_backup_meta(params)
            if (
                same_N_backup
                and len(loaded_bits) == N
                and self._backup_meta_matches(saved_meta, current_meta)
                and valid_progress
            ):
                act, bf, sm = self._prompt_matching_backup(backup_file, progress_label)
                return (act, bf, sm, not is_vectorized)
            act, bf, sm = self._prompt_mismatched_backup(
                backup_file, same_N_backup, loaded_bits, saved_index,
                saved_meta, current_meta, progress_label,
                is_vectorized=is_vectorized, completed_iters=completed_iters,
            )
            return (act, bf, sm, not is_vectorized)

        legacy_file = f"BER_bits_backup_{N}.npy"
        if os.path.exists(legacy_file):
            reply = QMessageBox.information(
                self,
                "Respaldo Anterior Detectado",
                f"Se encontró un archivo de respaldo antiguo '{legacy_file}' (sin metadatos de parámetros).\n"
                "¿Iniciar una ejecución nueva usando el formato seguro de respaldo?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            return ("fresh", None, None, False) if reply == QMessageBox.StandardButton.Yes else (
                "cancel", None, None, False
            )

        return ("fresh", None, None, False)

    def _prompt_csv_save_path(self, dialog_title, default_path):
        """Abre un diálogo para elegir ruta CSV y garantiza extensión .csv y carpeta padre."""
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

    def save_ber_bits_csv(self):
        """Guarda el vector BER_bits actual en un archivo CSV elegido por el usuario."""
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
        except Exception as exc:
            QMessageBox.critical(self, "Error al Guardar CSV", str(exc))

    def load_ber_bits_csv(self):
        """Carga y valida desde CSV un perfil previamente generado de BER por bit-canal."""
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
                    f"Se ajustó N a {loaded_len}.",
                )
                self.combo_N.setCurrentIndex(n_index)

            self.ber_bits = loaded_bits
            self.ber_bits_source = file_path
            self.active_mc_params = None
            self.btn_save_mc_csv.setEnabled(True)
            self.update_mc_status_label()
            self.log(f"📂 BER_bits cargado ({loaded_len} bits) desde: {file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error al Cargar CSV", str(exc))

    def save_ber_curves_csv(self):
        """Exporta a CSV las curvas BER y BLER frente a Eb/N0 actualmente simuladas."""
        if self.ber_curves_ebno is None or not self.ber_curves_data:
            QMessageBox.information(
                self, "Sin Datos para Guardar", "Ejecute primero una simulación BER/BLER vs Eb/N0."
            )
            return

        p = self.active_ber_params if self.active_ber_params else self.gather_parameters()
        default_path = default_ber_curves_csv_path(p)
        try:
            file_path = self._prompt_csv_save_path("Guardar archivo CSV de curvas BER/BLER", default_path)
            if not file_path:
                return
            data = {"Eb/N0 [dB]": self.ber_curves_ebno}
            for label, ber_arr in self.ber_curves_data.items():
                data[f"{label} (BER)"] = ber_arr
                if label in self.bler_curves_data:
                    data[f"{label} (BLER)"] = self.bler_curves_data[label]
            pd.DataFrame(data).to_csv(file_path, index=False, encoding="utf-8-sig")
            self.log(f"💾 CSV de curvas BER/BLER guardado en: {file_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Error al Guardar CSV", str(exc))

    # -----------------------------------------------------------------
    # CONTROLADORES DE SIMULACIÓN Y PRE-VERIFICACIÓN DE RECURSOS
    # -----------------------------------------------------------------

    def has_active_worker(self):
        """Comprueba si existe algún hilo de simulación o cálculo activo."""
        return any(
            w is not None and w.isRunning()
            for w in (self.mc_worker, self.ber_worker, self.snr_worker)
        )

    def start_auto_design_snr(self):
        """Lanza la estimación automática del SNR de diseño por bisección de información mutua."""
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
        self.snr_worker.log_signal.connect(self.log)
        self.snr_worker.progress_signal.connect(self.update_auto_snr_progress)
        self.snr_worker.finished_signal.connect(self.on_auto_snr_finished)
        self.snr_worker.stopped_signal.connect(self.on_auto_snr_stopped)
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

    def on_auto_snr_stopped(self):
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_status_activity.setText("Estimación de SNR de diseño detenida")
        self.set_simulation_running(False)
        self.log("⚠️ Estimación de SNR de diseño cancelada.")

    def cleanup_snr_worker(self):
        if self.snr_worker is not None:
            self.snr_worker.deleteLater()
            self.snr_worker = None

    def _apply_resource_recommendations(self, recommendations):
        """Aplica sobre la interfaz los ajustes sugeridos por la verificación de recursos."""
        if "N" in recommendations:
            idx = self.combo_N.findData(int(recommendations["N"]))
            if idx >= 0:
                self.combo_N.setCurrentIndex(idx)
        if "lote_mc" in recommendations:
            self.spin_lote_mc.setValue(int(recommendations["lote_mc"]))
        if "num_iter" in recommendations:
            self.spin_num_iter.setValue(int(recommendations["num_iter"]))
        if "lote_berplot" in recommendations:
            self.spin_lote_ber.setValue(int(recommendations["lote_berplot"]))
        if "ebno_rango" in recommendations:
            self.spin_ebno_rango.setValue(int(recommendations["ebno_rango"]))
        if "list_scl" in recommendations:
            scl_idx = self.combo_scl.findData(int(recommendations["list_scl"]))
            if scl_idx >= 0:
                self.combo_scl.setCurrentIndex(scl_idx)

    def precheck_system_resources(self, params, mode, lock_n=False, lock_mc_params=False, use_legacy=False):
        """Pre-verifica CPU/GPU y RAM antes de simular y permite aceptar reducciones seguras."""
        warnings, recommendations, summary = assess_system_resources(
            params,
            mode,
            has_gpu=HAS_GPU,
            lock_n=lock_n,
            lock_mc_params=lock_mc_params,
            use_legacy=use_legacy,
        )
        if not warnings:
            return True, params

        labels = {
            "N": "Longitud N",
            "lote_mc": "Lote paralelo MC",
            "num_iter": "Iteraciones MC",
            "lote_berplot": "Tamaño de lote BER",
            "ebno_rango": "Puntos Eb/N0",
            "list_scl": "Tamaño lista SCL",
        }
        rec_lines = [f"• {labels.get(k, k)}: {params[k]} → {v}" for k, v in recommendations.items()]
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

    def start_monte_carlo(self):
        """Inicia y supervisa la generación Monte Carlo de bits congelados."""
        if self.has_active_worker():
            return

        params = self.gather_parameters()
        action, backup_file, saved_meta, is_legacy_backup = self.resolve_mc_backup_action(params)
        if action == "cancel":
            return

        if action == "update_and_resume":
            self.apply_backup_meta_to_ui(saved_meta)
            params = self.gather_parameters()
            self.log(f"🔄 Parámetros de la interfaz actualizados desde el respaldo '{backup_file}'.")

        if not self.validate_parameters(params, mode="mc"):
            return

        resume_flag = action in ("resume", "update_and_resume")
        use_legacy = resume_flag and is_legacy_backup
        proceed, params = self.precheck_system_resources(
            params,
            mode="mc",
            lock_n=resume_flag,
            lock_mc_params=resume_flag,
            use_legacy=use_legacy,
        )
        if not proceed or not self.validate_parameters(params, mode="mc"):
            return

        self.active_mc_params = dict(params)
        self.set_simulation_running(True, mode="mc")
        total_steps = params["N"] if use_legacy else params["num_iter"]
        self.progress_bar.setRange(0, total_steps)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m (%p%)")
        self.lbl_status_activity.setText(f"Ejecutando Monte Carlo (0/{total_steps})...")

        self.log(
            f"--- Iniciando generación Monte Carlo de bits congelados "
            f"(N={params['N']}, K={params['K']}, Canal={params['canal']}, "
            f"Eq={params['ecualizador']}, SNR={params['snr_simulacion']} dB) ---"
        )
        self.mc_worker = MonteCarloWorker(params, resume_backup=resume_flag, use_legacy_loop=use_legacy)
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

    def update_mc_progress(self, current, total, elapsed, eta):
        """Actualiza barra de progreso, tiempo transcurrido y ETA de Monte Carlo."""
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current)
        if elapsed > 0:
            self.lbl_status_activity.setText(
                f"Monte Carlo: {current}/{total} | "
                f"Transcurrido: {format_duration(elapsed)} | ETA: {format_duration(eta)}"
            )
        else:
            self.lbl_status_activity.setText(f"Monte Carlo: {current}/{total}")

    def on_mc_finished(self, ber_bits, filename):
        """Registra los resultados completados de Monte Carlo y restaura los controles."""
        self.ber_bits = ber_bits
        self.ber_bits_source = filename
        self.update_mc_status_label()
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.log(f"✅ ¡Monte Carlo completado! Guardado automáticamente en: {filename}")
        self.lbl_status_activity.setText("Monte Carlo completado")
        self.set_simulation_running(False)

    def on_mc_stopped(self, completed, total, backup_path):
        """Restaura la interfaz cuando Monte Carlo se detiene por solicitud del usuario."""
        self.progress_bar.setFormat("%p%")
        suffix = " (respaldo guardado)" if backup_path else ""
        self.lbl_status_activity.setText(f"Monte Carlo detenido en {completed}/{total}{suffix}")
        self.set_simulation_running(False)

    def start_ber_simulation(self):
        """Inicia el barrido BER/BLER vs Eb/N0 y reinicia la gráfica para las nuevas curvas."""
        if self.has_active_worker():
            return

        params = self.gather_parameters()
        if not self.validate_parameters(params, mode="ber"):
            return

        has_matching_bits = self.ber_bits is not None and len(self.ber_bits) == params["N"]
        proceed, params = self.precheck_system_resources(params, mode="ber", lock_n=has_matching_bits)
        if not proceed or not self.validate_parameters(params, mode="ber"):
            return

        self.active_ber_params = dict(params)
        self.ber_curves_ebno = None
        self.ber_curves_data = {}
        self.bler_curves_data = {}
        self.redraw_curves()

        self.set_simulation_running(True, mode="ber")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v / %m puntos (%p%)")
        self.lbl_status_activity.setText("Iniciando simulación BER/BLER vs Eb/N0...")

        self.log("--- Iniciando simulación BER/BLER vs Eb/N0 ---")
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
            self.lbl_status_activity.setText("Deteniendo simulación BER al finalizar el lote actual...")
            self.log("⏹ Detención solicitada. Deteniendo al finalizar el lote actual...")

    def cleanup_ber_worker(self):
        if self.ber_worker is not None:
            self.ber_worker.deleteLater()
            self.ber_worker = None

    def update_ber_progress(self, current, total, status_text):
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(current)
        self.lbl_status_activity.setText(status_text)

    def plot_ber_curve(self, legend_label, ebno_dbs, ber_values, bler_values):
        """Almacena las curvas BER y BLER recibidas y actualiza la gráfica activa."""
        self.ber_curves_ebno = np.array(ebno_dbs, copy=True)
        self.ber_curves_data[legend_label] = np.array(ber_values, copy=True)
        self.bler_curves_data[legend_label] = np.array(bler_values, copy=True)
        self.redraw_curves()

        active_vals = ber_values if self.plot_metric == "BER" else bler_values
        valid_mask = np.isfinite(active_vals) & (active_vals > 0.0)
        zero_count = int(np.sum(active_vals == 0.0))

        if np.any(valid_mask):
            if zero_count > 0:
                self.log(
                    f"✔ Curva graficada ({self.plot_metric}): {legend_label} "
                    f"({zero_count} punto(s) con valor 0 omitidos en escala logarítmica)"
                )
            else:
                self.log(f"✔ Curva graficada ({self.plot_metric}): {legend_label}")
        else:
            self.log(
                f"ℹ️ La curva '{legend_label}' tuvo 0 errores en todos los puntos Eb/N0 "
                "(sin puntos para escala logarítmica)."
            )

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

    def on_worker_error(self, err_msg, detail_tb=""):
        """Informa errores originados en los hilos y restablece el estado de la interfaz."""
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.lbl_status_activity.setText("Error de simulación")
        self.set_simulation_running(False)
        self.log(f"❌ ERROR: {err_msg}")
        if detail_tb:
            self.log(detail_tb.rstrip())
        QMessageBox.critical(self, "Error de Simulación", err_msg)

    # -----------------------------------------------------------------
    # CIERRE SEGURO DE LA VENTANA
    # -----------------------------------------------------------------

    def _on_deferred_close_check(self):
        """Completa el cierre diferido en cuanto terminan todos los hilos activos."""
        if self._pending_close and not self.has_active_worker():
            self.close()

    def closeEvent(self, event):
        """Detiene de forma segura cualquier hilo en ejecución antes de cerrar la ventana."""
        running_workers = [
            (w, timeout_ms)
            for w, timeout_ms in (
                (self.mc_worker, 10000),
                (self.ber_worker, 10000),
                (self.snr_worker, 5000),
            )
            if w is not None and w.isRunning()
        ]

        if not running_workers:
            self._pending_close = False
            event.accept()
            return

        if not self._pending_close:
            reply = QMessageBox.question(
                self,
                "Simulación en Progreso",
                "Hay una simulación o cálculo en ejecución.\n"
                "¿Desea detenerlo de forma segura (guardando respaldo Monte Carlo si aplica) y salir?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            self._pending_close = True
            self.lbl_status_activity.setText("Deteniendo simulación activa antes de salir...")
            QApplication.processEvents()

            still_running = []
            for worker, timeout_ms in running_workers:
                worker.request_stop()
                if not worker.wait(timeout_ms) or worker.isRunning():
                    still_running.append(worker)
        else:
            still_running = [w for w, _ in running_workers]

        if still_running:
            for worker in still_running:
                try:
                    worker.finished.connect(
                        self._on_deferred_close_check, Qt.ConnectionType.UniqueConnection
                    )
                except TypeError:
                    pass
            self.lbl_status_activity.setText(
                "Esperando finalización del kernel GPU/CPU para cerrar de forma segura..."
            )
            QMessageBox.warning(
                self,
                "Cierre Diferido en Curso",
                "Un cálculo en GPU/CPU aún está finalizando el lote actual.\n\n"
                "Espere unos segundos: la ventana se cerrará automáticamente de forma segura "
                "en cuanto termine el hilo activo.",
            )
            event.ignore()
            return

        self._pending_close = False
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PolarSimGUI()
    window.show()
    sys.exit(app.exec())
