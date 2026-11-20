import os
import sys
import gc
import psutil
from datetime import datetime
import numpy as np
import pandas as pd

# Configura la GPU y los mensajes de TensorFlow antes de importar TF/Sionna.
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
tf.get_logger().setLevel('ERROR')

import sionna

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QProgressBar, QTextEdit, QFileDialog, QMessageBox,
    QLabel, QSplitter, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# =====================================================================
# 1. MODELOS DE CANAL Y ECUALIZADORES (compilados con XLA)
# =====================================================================

# Canal AWGN: ganancia plana unitaria y ruido gaussiano complejo.
@tf.function(jit_compile=True)
def canalAWGN(ebno_db, bitsxsimb, coderate, x):
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    awgn_real = sionna.phy.config.tf_rng.normal(tf.shape(x), mean=0.0, stddev=tf.sqrt(no / 2.0))
    awgn_imag = sionna.phy.config.tf_rng.normal(tf.shape(x), mean=0.0, stddev=tf.sqrt(no / 2.0))
    awgn = tf.complex(awgn_real, awgn_imag)
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
    repeticiones = tf.cast(tf.math.floordiv(N + bloq_coher - 1, bloq_coher), tf.int32)
    
    awgn_real = sionna.phy.config.tf_rng.normal(tf.shape(x), mean=0.0, stddev=tf.sqrt(no / 2.0))
    awgn_imag = sionna.phy.config.tf_rng.normal(tf.shape(x), mean=0.0, stddev=tf.sqrt(no / 2.0))
    awgn = tf.complex(awgn_real, awgn_imag)
    
    h_real = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], stddev=tf.sqrt(0.5), dtype=tf.float32)
    h_imag = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], stddev=tf.sqrt(0.5), dtype=tf.float32)
    h = tf.repeat(tf.complex(h_real, h_imag), repeats=bloq_coher, axis=1)
    h = h[:, :N]
    y = h * x + awgn
    return y, h, no

# Canal Rice: combina componente de línea de vista y desvanecimiento disperso.
@tf.function(jit_compile=True)
def canalRice(bloq_coher, k_factor_db, ebno_db, bitsxsimb, coderate, x):
    batch_size = tf.shape(x)[0]
    bloq_coher = tf.cast(bloq_coher, tf.int32)
    N = tf.shape(x)[-1]
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    repeticiones = tf.cast(tf.math.floordiv(N + bloq_coher - 1, bloq_coher), tf.int32)
    
    awgn_real = sionna.phy.config.tf_rng.normal(tf.shape(x), mean=0.0, stddev=tf.sqrt(no / 2.0))
    awgn_imag = sionna.phy.config.tf_rng.normal(tf.shape(x), mean=0.0, stddev=tf.sqrt(no / 2.0))
    awgn = tf.complex(awgn_real, awgn_imag)
    
    k_factor_lin = tf.pow(10.0, tf.cast(k_factor_db, tf.float32) / 10.0)
    h_nlos_real = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], stddev=tf.sqrt(0.5), dtype=tf.float32)
    h_nlos_imag = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], stddev=tf.sqrt(0.5), dtype=tf.float32)
    h_nlos = tf.complex(h_nlos_real, h_nlos_imag)
    
    phi = sionna.phy.config.tf_rng.uniform([batch_size, repeticiones], minval=0.0, maxval=tf.constant(2.0 * np.pi, tf.float32))
    h_los = tf.complex(tf.cos(phi), tf.sin(phi))
    
    a = tf.sqrt(k_factor_lin / (k_factor_lin + 1.0))
    b = tf.sqrt(1.0 / (k_factor_lin + 1.0))
    h_bloque = tf.cast(a, tf.complex64) * tf.cast(h_los, tf.complex64) + tf.cast(b, tf.complex64) * tf.cast(h_nlos, tf.complex64)
    h = tf.repeat(h_bloque, repeats=bloq_coher, axis=1)
    h = h[:, :N]
    y = h * x + awgn
    return y, h, no

# Canal lognormal: aplica desvanecimiento multiplicativo por sombreado.
@tf.function(jit_compile=True)
def canalLognormal(bloq_coher, sigma_db, ebno_db, bitsxsimb, coderate, x):
    batch_size = tf.shape(x)[0]
    bloq_coher = tf.cast(bloq_coher, tf.int32)
    N = tf.shape(x)[-1]
    no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=coderate)
    repeticiones = tf.cast(tf.math.floordiv(N + bloq_coher - 1, bloq_coher), tf.int32)
    
    awgn_real = sionna.phy.config.tf_rng.normal(tf.shape(x), mean=0.0, stddev=tf.sqrt(no / 2.0))
    awgn_imag = sionna.phy.config.tf_rng.normal(tf.shape(x), mean=0.0, stddev=tf.sqrt(no / 2.0))
    awgn = tf.complex(awgn_real, awgn_imag)
    
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
    repeticiones = tf.cast(tf.math.floordiv(N + bloq_coher - 1, bloq_coher), tf.int32)
    
    awgn_real = sionna.phy.config.tf_rng.normal(tf.shape(x), mean=0.0, stddev=tf.sqrt(no / 2.0))
    awgn_imag = sionna.phy.config.tf_rng.normal(tf.shape(x), mean=0.0, stddev=tf.sqrt(no / 2.0))
    awgn = tf.complex(awgn_real, awgn_imag)
    
    g_real = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], stddev=tf.sqrt(0.5), dtype=tf.float32)
    g_imag = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], stddev=tf.sqrt(0.5), dtype=tf.float32)
    g = tf.complex(g_real, g_imag)
    
    sigma_lin = tf.cast(tf.math.log(10.0) / 20.0 * sigma_db, tf.float32)
    mu_ln = -tf.square(sigma_lin)
    z = sionna.phy.config.tf_rng.normal([batch_size, repeticiones], mean=mu_ln, stddev=sigma_lin, dtype=tf.float32)
    a_c = tf.complex(tf.exp(z), tf.zeros_like(z))
    
    h_comp = g * a_c
    h = tf.repeat(h_comp, repeats=bloq_coher, axis=1)[:, :N]
    y = h * x + awgn
    return y, h, no

# Sin ecualización: conserva la señal recibida para canales AWGN.
@tf.function(jit_compile=True)
def sin_ecualizar(y, h, no):
    return y, no

# Ecualizador ZF: cancela la ganancia del canal.
@tf.function(jit_compile=True)
def zf(y, h, no, eps=1e-12):
    h2 = tf.maximum(tf.math.square(tf.abs(h)), eps)
    x_hat = y * tf.math.conj(h) / tf.cast(h2, tf.complex64)
    no_eff = no / tf.maximum(h2, eps)
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
        c = self.p['canal']
        if c == "awgn":
            y, h, no = canalAWGN(ebno_db, self.p['bitsxsimb'], coderate, x)
        elif c == "rayleigh":
            y, h, no = canalRayleigh(self.p['bloq_coher'], ebno_db, self.p['bitsxsimb'], coderate, x)
        elif c == "rice":
            y, h, no = canalRice(self.p['bloq_coher'], self.p['k_factor_db'], ebno_db, self.p['bitsxsimb'], coderate, x)
        elif c == "lognormal":
            y, h, no = canalLognormal(self.p['bloq_coher'], self.p['sigma_db'], ebno_db, self.p['bitsxsimb'], coderate, x)
        elif c == "suzuki":
            y, h, no = canalRayleighLognormal(self.p['bloq_coher'], self.p['sigma_db'], ebno_db, self.p['bitsxsimb'], coderate, x)
        else:
            raise ValueError(f"Invalid channel: {c}")

        eq = self.p['ecualizador']
        if eq == "no_eq":
            x_hat, no_eff = sin_ecualizar(y, h, no)
        elif eq == "zf":
            x_hat, no_eff = zf(y, h, no)
        elif eq == "mmse":
            x_hat, no_eff = mmse(y, h, no)
        else:
            raise ValueError(f"Invalid equalizer: {eq}")

        llr_prev = tf.clip_by_value(self.cb['demapeador'](x_hat, no_eff), -self.p['LLR_clip'], self.p['LLR_clip'])
        llr = self.cb['deinterleaver'](llr_prev)
        return llr


# Sistema de referencia sin código de canal.
class SinCodigo(BaseSystemBlock):
    def __init__(self, params, common_blocks):
        super().__init__(params, common_blocks)

    def call(self, batch_size, ebno_db):
        b = self.cb['datos']([batch_size, self.p['K']])
        b_int = self.cb['interleaver'](b)
        x = self.cb['mapeador'](b_int)
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
        bits_codif = self.polar_codif(b)
        bits_int = self.cb['interleaver'](bits_codif)
        x = self.cb['mapeador'](bits_int)
        llr = self.apply_channel_and_eq(x, ebno_db, coderate=self.p['K'] / self.p['N'])
        b_hat = self.polar_decodif(llr)
        return b, b_hat


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
        b_crc = self.cb['crc_codif'](b)
        bits_codif = self.polar_codif(b_crc)
        bits_int = self.cb['interleaver'](bits_codif)
        x = self.cb['mapeador'](bits_int)
        llr = self.apply_channel_and_eq(x, ebno_db, coderate=self.p['K'] / self.p['N'])
        b_crc_hat = self.polar_decodif(llr)
        b_hat = b_crc_hat[..., :K_efec]
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
        bits_codif = self.ldpc_codif(b)
        bits_int = self.cb['interleaver'](bits_codif)
        x = self.cb['mapeador'](bits_int)
        llr = self.apply_channel_and_eq(
            x, ebno_db, coderate=self.p['K'] / self.p['N']
        )
        b_hat = self.ldpc_decodif(llr)
        return b, b_hat


# =====================================================================
# 3. HILOS DE TRABAJO EN SEGUNDO PLANO
# =====================================================================

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
        'crc_long': crc_codif.crc_length
    }


# Mide la confiabilidad de cada canal de bit Polar mediante Monte Carlo.
class MonteCarloWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(np.ndarray, str)
    error_signal = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.p = params
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            cb = build_common_blocks(self.p)
            N = self.p['N']
            K = self.p['K']
            lote_mc = self.p['lote_mc']
            num_iter = self.p['num_iter']
            snr_sim = tf.constant(self.p['snr_simulacion'], dtype=tf.float32)
            coderate_sim = float(K / N)

            backup_file = f"BER_bits_backup_{N}.npy"
            if os.path.exists(backup_file):
                BER_bits = np.load(backup_file)
                if len(BER_bits) == N:
                    start_index = int(np.count_nonzero(BER_bits))
                    self.log_signal.emit(f"Recovered backup. Resuming from bit-channel {start_index}/{N}")
                else:
                    BER_bits = np.zeros(N)
                    start_index = 0
            else:
                BER_bits = np.zeros(N)
                start_index = 0

            for i in range(start_index, N):
                if self._stop_requested:
                    np.save(backup_file, BER_bits)
                    self.log_signal.emit(f"⚠️ Simulation stopped by user at channel {i+1}/{N}. Backup saved.")
                    return

                bits_congelados = np.array([j for j in range(N) if j != i])
                codificador_temp = sionna.phy.fec.polar.encoding.PolarEncoder(bits_congelados, N)
                decodificador_temp = sionna.phy.fec.polar.decoding.PolarSCLDecoder(bits_congelados, N, list_size=1)

                @tf.function(jit_compile=True)
                def simulacion_batch_size():
                    bits = cb['datos']([lote_mc, 1])
                    bits_codif = codificador_temp(bits)
                    bits_int = cb['interleaver'](bits_codif)
                    x = cb['mapeador'](bits_int)

                    c = self.p['canal']
                    if c == "awgn":
                        y, h, no = canalAWGN(snr_sim, self.p['bitsxsimb'], coderate_sim, x)
                    elif c == "rayleigh":
                        y, h, no = canalRayleigh(self.p['bloq_coher'], snr_sim, self.p['bitsxsimb'], coderate_sim, x)
                    elif c == "rice":
                        y, h, no = canalRice(self.p['bloq_coher'], self.p['k_factor_db'], snr_sim, self.p['bitsxsimb'], coderate_sim, x)
                    elif c == "lognormal":
                        y, h, no = canalLognormal(self.p['bloq_coher'], self.p['sigma_db'], snr_sim, self.p['bitsxsimb'], coderate_sim, x)
                    elif c == "suzuki":
                        y, h, no = canalRayleighLognormal(self.p['bloq_coher'], self.p['sigma_db'], snr_sim, self.p['bitsxsimb'], coderate_sim, x)

                    eq = self.p['ecualizador']
                    if eq == "no_eq":
                        x_hat, no_eff = sin_ecualizar(y, h, no)
                    elif eq == "zf":
                        x_hat, no_eff = zf(y, h, no)
                    elif eq == "mmse":
                        x_hat, no_eff = mmse(y, h, no)

                    llr_prev = tf.clip_by_value(cb['demapeador'](x_hat, no_eff), -self.p['LLR_clip'], self.p['LLR_clip'])
                    llr = cb['deinterleaver'](llr_prev)
                    bits_decodif = decodificador_temp(llr)

                    err = tf.math.count_nonzero(tf.not_equal(bits, bits_decodif), dtype=tf.int32)
                    return tf.cast(err, tf.int64), tf.constant(lote_mc, dtype=tf.int64)

                errores_totales = 0
                bits_totales = 0
                for _ in range(num_iter):
                    err, total = simulacion_batch_size()
                    errores_totales += int(err.numpy())
                    bits_totales += int(total.numpy())

                BER_bits[i] = errores_totales / bits_totales
                self.progress_signal.emit(i + 1, N)
                self.log_signal.emit(f"Bit-channel {i+1}/{N} | BER: {BER_bits[i]:.2e}")

                if i % 10 == 0:
                    np.save(backup_file, BER_bits)

                del simulacion_batch_size, codificador_temp, decodificador_temp
                tf.keras.backend.clear_session()
                gc.collect()

                mem = psutil.virtual_memory()
                if mem.percent > 85.0:
                    np.save(backup_file, BER_bits)
                    self.error_signal.emit(f"RAM usage exceeded 85% ({mem.percent}%). Emergency backup saved at channel {i}.")
                    return

            # Guarda los resultados finales de confiabilidad de los canales de bits.
            output_dir = f"results_{N}"
            os.makedirs(output_dir, exist_ok=True)
            fecha = datetime.now().strftime("%y%m%d")
            filename = os.path.join(
                output_dir,
                f"{fecha}_{self.p['canal']}_{self.p['ecualizador']}_{self.p['snr_simulacion']}SNR_BER_bits_{N}_{K}_{num_iter}iter.csv"
            )
            df = pd.DataFrame({"BER bits": BER_bits})
            df.to_csv(filename, index=True)

            if os.path.exists(backup_file):
                os.remove(backup_file)

            self.finished_signal.emit(BER_bits, filename)

        except Exception as e:
            self.error_signal.emit(str(e))


# Ejecuta las barridas BER frente a Eb/N0 para los distintos sistemas.
class BERSimWorker(QThread):
    log_signal = pyqtSignal(str)
    curve_ready_signal = pyqtSignal(str, np.ndarray, np.ndarray)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, params, ber_bits=None):
        super().__init__()
        self.p = params
        self.ber_bits = ber_bits

    def run(self):
        try:
            cb = build_common_blocks(self.p)
            ebno_dbs = np.linspace(self.p['ebno_min'], self.p['ebno_max'], self.p['ebno_rango'])
            ber_plots = sionna.phy.utils.PlotBER(f"Comparacion canal {self.p['canal']}")

            # 1. Sistema sin código de canal
            self.log_signal.emit("Running BER simulation: Uncoded (Sin código)...")
            sin_codigo = SinCodigo(self.p, cb)
            ber_uncoded, _ = ber_plots.simulate(
                sin_codigo,
                ebno_dbs=ebno_dbs,
                batch_size=self.p['lote_berplot'],
                num_target_block_errors=100,
                legend="Sin codigo",
                soft_estimates=True,
                max_mc_iter=100,
                show_fig=False
            )
            self.curve_ready_signal.emit("Sin código", ebno_dbs, np.array(ber_uncoded))

            # Si existen bits congelados, ejecuta las curvas Polar y Polar + CRC.
            if self.ber_bits is not None and len(self.ber_bits) == self.p['N']:
                indices_ordenados = np.argsort(self.ber_bits)
                bits_congelados = indices_ordenados[-(self.p['N'] - self.p['K']):]

                # 2. Código Polar
                self.log_signal.emit("Running BER simulation: Polar Code (SCL)...")
                codigo_polar = CodigoPolar(self.p, cb, bits_congelados)
                ber_polar, _ = ber_plots.simulate(
                    codigo_polar,
                    ebno_dbs=ebno_dbs,
                    batch_size=self.p['lote_berplot'],
                    num_target_block_errors=250,
                    legend="Codigo Polar",
                    soft_estimates=False,
                    max_mc_iter=25,
                    show_fig=False,
                    forward_keyboard_interrupt=False
                )
                self.curve_ready_signal.emit("Código Polar", ebno_dbs, np.array(ber_polar))

                # 3. Código Polar con CRC
                if self.p['K'] > cb['crc_long']:
                    self.log_signal.emit(f"Running BER simulation: Polar Code + {self.p['crc_degree']}...")
                    codigo_polar_crc = CodigoPolarCRC(self.p, cb, bits_congelados)
                    ber_polar_crc, _ = ber_plots.simulate(
                        codigo_polar_crc,
                        ebno_dbs=ebno_dbs,
                        batch_size=self.p['lote_berplot'],
                        num_target_block_errors=250,
                        legend="Codigo Polar + CRC",
                        soft_estimates=False,
                        max_mc_iter=25,
                        show_fig=False,
                        forward_keyboard_interrupt=False
                    )
                    self.curve_ready_signal.emit(f"Código Polar + {self.p['crc_degree']}", ebno_dbs, np.array(ber_polar_crc))
                else:
                    self.log_signal.emit("⚠️ Skipping Polar+CRC: K must be greater than CRC length.")
            else:
                self.log_signal.emit("ℹ️ No frozen bits loaded/generated. Simulated Uncoded curve only.")

            # 4. Código LDPC 5G NR
            self.log_signal.emit("Running BER simulation: LDPC 5G NR...")
            codigo_ldpc = CodigoLDPC(self.p, cb)
            ber_ldpc, _ = ber_plots.simulate(
                codigo_ldpc,
                ebno_dbs=ebno_dbs,
                batch_size=self.p['lote_berplot'],
                num_target_block_errors=250,
                legend="Codigo LDPC",
                soft_estimates=False,
                max_mc_iter=25,
                show_fig=False,
                forward_keyboard_interrupt=False
            )
            self.curve_ready_signal.emit("Código LDPC", ebno_dbs, np.array(ber_ldpc))

            self.finished_signal.emit()

        except Exception as e:
            self.error_signal.emit(str(e))


# =====================================================================
# 4. VENTANA PRINCIPAL PYQT6
# =====================================================================

# Ventana principal con controles, progreso, registros y gráfica BER.
class PolarSimGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sionna Polar/LDPC Code Simulator & Monte Carlo Designer")
        self.resize(1350, 860)

        self.ber_bits = None
        self.mc_worker = None
        self.ber_worker = None

        self.init_ui()

    def init_ui(self):
        # Divide la interfaz entre parámetros de entrada y visualización de resultados.
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)

        # ---------------- PANEL IZQUIERDO: PARÁMETROS ----------------
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # Grupo 1: parámetros básicos y configuración del código
        grp_basic = QGroupBox("1. Basic & Code Parameters")
        form_basic = QFormLayout(grp_basic)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 999999)
        self.spin_seed.setValue(0)

        self.combo_N = QComboBox()
        for exp in range(5, 14):  # 32 to 8192
            self.combo_N.addItem(str(2**exp), 2**exp)
        self.combo_N.setCurrentText("1024")
        self.combo_N.currentIndexChanged.connect(self.on_N_changed)

        self.spin_K = QSpinBox()
        self.spin_K.setRange(1, 8192)
        self.spin_K.setValue(512)

        self.combo_bits = QComboBox()
        self.combo_bits.addItem("1 (BPSK - PAM)", 1)
        self.combo_bits.addItem("2 (QPSK - QAM)", 2)
        self.combo_bits.addItem("4 (16-QAM)", 4)
        self.combo_bits.addItem("6 (64-QAM)", 6)

        self.combo_crc = QComboBox()
        for crc in ["CRC24A", "CRC24B", "CRC24C", "CRC16", "CRC11", "CRC6"]:
            self.combo_crc.addItem(crc)

        form_basic.addRow("Random Seed:", self.spin_seed)
        form_basic.addRow("Codeword Length (N):", self.combo_N)
        form_basic.addRow("Info Bits (K):", self.spin_K)
        form_basic.addRow("Bits per Symbol:", self.combo_bits)
        form_basic.addRow("CRC Polynomial:", self.combo_crc)
        left_layout.addWidget(grp_basic)

        # Grupo 2: parámetros del modelo de canal
        grp_chan = QGroupBox("2. Channel Parameters")
        form_chan = QFormLayout(grp_chan)

        self.combo_canal = QComboBox()
        for c in ["suzuki", "awgn", "rayleigh", "rice", "lognormal"]:
            self.combo_canal.addItem(c)
        self.combo_canal.currentTextChanged.connect(self.on_channel_changed)

        self.spin_coher = QSpinBox()
        self.spin_coher.setRange(1, 8192)
        self.spin_coher.setValue(64)

        self.spin_k_db = QDoubleSpinBox()
        self.spin_k_db.setRange(-20.0, 50.0)
        self.spin_k_db.setValue(6.0)

        self.spin_sigma_db = QDoubleSpinBox()
        self.spin_sigma_db.setRange(0.1, 30.0)
        self.spin_sigma_db.setValue(6.0)

        form_chan.addRow("Channel Model:", self.combo_canal)
        form_chan.addRow("Coherence Block (bloq_coher):", self.spin_coher)
        form_chan.addRow("Rice K-Factor [dB]:", self.spin_k_db)
        form_chan.addRow("Lognormal/Suzuki σ [dB]:", self.spin_sigma_db)
        left_layout.addWidget(grp_chan)

        # Grupo 3: parámetros del receptor
        grp_rx = QGroupBox("3. Receiver Parameters")
        form_rx = QFormLayout(grp_rx)

        self.combo_eq = QComboBox()
        for eq in ["mmse", "zf", "no_eq"]:
            self.combo_eq.addItem(eq)
        self.update_equalizer_options(self.combo_canal.currentText())

        self.combo_scl = QComboBox()
        for l in [1, 2, 4, 8, 16, 32]:
            self.combo_scl.addItem(str(l), l)
        self.combo_scl.setCurrentText("8")

        self.spin_llr = QDoubleSpinBox()
        self.spin_llr.setRange(10.0, 5000.0)
        self.spin_llr.setValue(500.0)

        form_rx.addRow("Equalizer:", self.combo_eq)
        form_rx.addRow("SCL List Size (list_scl):", self.combo_scl)
        form_rx.addRow("LLR Clipping Value:", self.spin_llr)
        left_layout.addWidget(grp_rx)

        # Grupo 4: generación Monte Carlo de bits congelados
        grp_mc = QGroupBox("4. Monte Carlo Design (In[2] & In[8])")
        form_mc = QFormLayout(grp_mc)

        self.spin_lote_mc = QSpinBox()
        self.spin_lote_mc.setRange(10, 50000)
        self.spin_lote_mc.setSingleStep(500)
        self.spin_lote_mc.setValue(2000)

        self.spin_num_iter = QSpinBox()
        self.spin_num_iter.setRange(1, 50000)
        self.spin_num_iter.setSingleStep(500)
        self.spin_num_iter.setValue(2000)

        self.spin_snr_sim = QDoubleSpinBox()
        self.spin_snr_sim.setRange(-20.0, 40.0)
        self.spin_snr_sim.setSingleStep(0.5)
        self.spin_snr_sim.setValue(5.0)

        self.lbl_mc_status = QLabel("Frozen Bits: Not loaded")
        self.lbl_mc_status.setStyleSheet("color: #d9534f; font-weight: bold;")

        form_mc.addRow("Parallel Batch (lote_mc):", self.spin_lote_mc)
        form_mc.addRow("Sequential Iters (num_iter):", self.spin_num_iter)
        form_mc.addRow("Design SNR [dB]:", self.spin_snr_sim)
        form_mc.addRow("Status:", self.lbl_mc_status)
        left_layout.addWidget(grp_mc)

        # Grupo 5: parámetros de la gráfica BER
        grp_ber = QGroupBox("5. BER vs Eb/N0 Plot Parameters")
        form_ber = QFormLayout(grp_ber)

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

        form_ber.addRow("Eb/N0 Min [dB]:", self.spin_ebno_min)
        form_ber.addRow("Eb/N0 Max [dB]:", self.spin_ebno_max)
        form_ber.addRow("Number of Points:", self.spin_ebno_rango)
        form_ber.addRow("Plot Batch Size (lote_berplot):", self.spin_lote_ber)
        left_layout.addWidget(grp_ber)

        # Botones de acciones principales
        btn_layout = QVBoxLayout()
        self.btn_run_mc = QPushButton("▶ 1. Run Monte Carlo (Generate Frozen Bits)")
        self.btn_run_mc.setStyleSheet("background-color: #0275d8; color: white; font-weight: bold; padding: 8px;")
        self.btn_run_mc.clicked.connect(self.start_monte_carlo)

        self.btn_stop_mc = QPushButton("⏹ Stop Monte Carlo")
        self.btn_stop_mc.setEnabled(False)
        self.btn_stop_mc.clicked.connect(self.stop_monte_carlo)

        self.btn_load_csv = QPushButton("📂 Load BER_bits from CSV...")
        self.btn_load_csv.clicked.connect(self.load_ber_bits_csv)

        self.btn_run_ber = QPushButton("📈 2. Simulate & Plot BER Curves")
        self.btn_run_ber.setStyleSheet("background-color: #5cb85c; color: white; font-weight: bold; padding: 8px;")
        self.btn_run_ber.clicked.connect(self.start_ber_simulation)

        btn_layout.addWidget(self.btn_run_mc)
        btn_layout.addWidget(self.btn_stop_mc)
        btn_layout.addWidget(self.btn_load_csv)
        btn_layout.addWidget(self.btn_run_ber)
        left_layout.addLayout(btn_layout)

        left_layout.addStretch()
        scroll_area.setWidget(left_widget)
        splitter.addWidget(scroll_area)

        # ---------------- PANEL DERECHO: GRÁFICA Y REGISTROS ----------------
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
        right_layout.addWidget(self.progress_bar)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #1e1e1e; color: #d4d4d4; font-family: monospace;")
        right_layout.addWidget(self.log_console, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setSizes([420, 930])

    # =================================================================
    # EVENTOS Y FUNCIONES AUXILIARES DE LA INTERFAZ
    # =================================================================

    # Configura los ejes de la gráfica logarítmica de BER.
    def setup_plot_axes(self, title="BER vs Eb/N0"):
        self.ax.clear()
        self.ax.set_yscale('log')
        self.ax.set_xlabel(r"$E_b/N_0$ [dB]", fontsize=11)
        self.ax.set_ylabel("BER", fontsize=11)
        self.ax.set_title(title, fontsize=13, fontweight='bold')
        self.ax.set_ylim([1e-6, 1.0])
        self.ax.grid(True, which="both", linestyle="--", alpha=0.6)

    def log(self, message):
        self.log_console.append(message)
        self.log_console.verticalScrollBar().setValue(self.log_console.verticalScrollBar().maximum())

    def on_N_changed(self):
        N = self.combo_N.currentData()
        self.spin_K.setMaximum(N - 1)
        self.spin_K.setValue(N // 2)
        self.spin_coher.setMaximum(N)
        self.spin_coher.setValue(max(1, N // 16))

    # Actualiza los valores SNR y las opciones válidas al cambiar de canal.
    def on_channel_changed(self, channel_name):
        snr_defaults = {
            "awgn": -1.2,
            "rayleigh": 1.0,
            "rice": -0.3,
            "lognormal": 2.4,
            "suzuki": 5.0
        }
        if channel_name in snr_defaults:
            self.spin_snr_sim.setValue(snr_defaults[channel_name])
        self.update_equalizer_options(channel_name)

    # Permite sin_ecualizar solo con el canal AWGN.
    def update_equalizer_options(self, channel_name):
        no_eq_index = self.combo_eq.findText("no_eq")
        no_eq_enabled = (channel_name == "awgn")
        model = self.combo_eq.model()
        model.setData(
            model.index(no_eq_index, 0),
            no_eq_enabled,
            Qt.ItemDataRole.UserRole - 1
        )

        if not no_eq_enabled and self.combo_eq.currentText() == "no_eq":
            self.combo_eq.setCurrentText("mmse")

    # Reúne una configuración consistente antes de iniciar una simulación.
    def gather_parameters(self):
        if self.combo_canal.currentText() != "awgn" and self.combo_eq.currentText() == "no_eq":
            self.combo_eq.setCurrentText("mmse")
        return {
            'seed': self.spin_seed.value(),
            'N': self.combo_N.currentData(),
            'K': self.spin_K.value(),
            'bitsxsimb': self.combo_bits.currentData(),
            'crc_degree': self.combo_crc.currentText(),
            'canal': self.combo_canal.currentText(),
            'bloq_coher': self.spin_coher.value(),
            'k_factor_db': self.spin_k_db.value(),
            'sigma_db': self.spin_sigma_db.value(),
            'ecualizador': self.combo_eq.currentText(),
            'list_scl': self.combo_scl.currentData(),
            'LLR_clip': self.spin_llr.value(),
            'lote_mc': self.spin_lote_mc.value(),
            'num_iter': self.spin_num_iter.value(),
            'snr_simulacion': self.spin_snr_sim.value(),
            'ebno_min': self.spin_ebno_min.value(),
            'ebno_max': self.spin_ebno_max.value(),
            'ebno_rango': self.spin_ebno_rango.value(),
            'lote_berplot': self.spin_lote_ber.value()
        }

    # =================================================================
    # CONTROLADORES DE SIMULACIÓN
    # =================================================================

    # Inicia y supervisa la generación Monte Carlo de bits congelados.
    def start_monte_carlo(self):
        params = self.gather_parameters()
        self.btn_run_mc.setEnabled(False)
        self.btn_run_ber.setEnabled(False)
        self.btn_stop_mc.setEnabled(True)
        self.progress_bar.setRange(0, params['N'])
        self.progress_bar.setValue(0)

        self.log(f"\n--- Starting Monte Carlo Frozen Bit Generation (N={params['N']}, K={params['K']}, Channel={params['canal']}) ---")
        self.mc_worker = MonteCarloWorker(params)
        self.mc_worker.log_signal.connect(self.log)
        self.mc_worker.progress_signal.connect(self.update_mc_progress)
        self.mc_worker.finished_signal.connect(self.on_mc_finished)
        self.mc_worker.error_signal.connect(self.on_worker_error)
        self.mc_worker.start()

    def stop_monte_carlo(self):
        if self.mc_worker and self.mc_worker.isRunning():
            self.mc_worker.request_stop()
            self.btn_stop_mc.setEnabled(False)
            self.btn_run_mc.setEnabled(True)
            self.btn_run_ber.setEnabled(True)

    # Actualiza el progreso de la simulación de canales de bits.
    def update_mc_progress(self, current, total):
        self.progress_bar.setValue(current)

    # Guarda en la interfaz los resultados completados de Monte Carlo.
    def on_mc_finished(self, ber_bits, filename):
        self.ber_bits = ber_bits
        self.lbl_mc_status.setText(f"Ready ({len(ber_bits)} bits)")
        self.lbl_mc_status.setStyleSheet("color: #5cb85c; font-weight: bold;")
        self.log(f"✅ Monte Carlo completed! Saved to: {filename}")
        self.btn_run_mc.setEnabled(True)
        self.btn_run_ber.setEnabled(True)
        self.btn_stop_mc.setEnabled(False)

    # Carga desde CSV un conjunto previamente generado de valores BER por bit.
    def load_ber_bits_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select BER_bits CSV File", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            try:
                df = pd.read_csv(file_path)
                if "BER bits" not in df.columns:
                    raise ValueError("CSV does not contain a 'BER bits' column.")
                loaded_bits = df["BER bits"].to_numpy()
                current_N = self.combo_N.currentData()
                if len(loaded_bits) != current_N:
                    QMessageBox.warning(
                        self, "Length Mismatch",
                        f"Loaded CSV has {len(loaded_bits)} bits, but N is set to {current_N}.\n"
                        f"Adjusting N to {len(loaded_bits)}."
                    )
                    self.combo_N.setCurrentText(str(len(loaded_bits)))
                self.ber_bits = loaded_bits
                self.lbl_mc_status.setText(f"Loaded ({len(loaded_bits)} bits)")
                self.lbl_mc_status.setStyleSheet("color: #5cb85c; font-weight: bold;")
                self.log(f"📂 Loaded BER_bits from: {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error Loading CSV", str(e))

    # Inicia la barrida BER y prepara la gráfica para las nuevas curvas.
    def start_ber_simulation(self):
        params = self.gather_parameters()
        self.setup_plot_axes(f"Comparación Canal: {params['canal'].capitalize()} (N={params['N']}, K={params['K']}, Eq={params['ecualizador'].upper()})")
        self.ax.set_xlim([params['ebno_min'], params['ebno_max']])
        self.canvas.draw()

        self.btn_run_mc.setEnabled(False)
        self.btn_run_ber.setEnabled(False)
        self.progress_bar.setRange(0, 0)  # Indeterminate busy indicator

        self.log(f"\n--- Starting BER vs Eb/N0 Simulation ---")
        self.ber_worker = BERSimWorker(params, self.ber_bits)
        self.ber_worker.log_signal.connect(self.log)
        self.ber_worker.curve_ready_signal.connect(self.plot_ber_curve)
        self.ber_worker.finished_signal.connect(self.on_ber_finished)
        self.ber_worker.error_signal.connect(self.on_worker_error)
        self.ber_worker.start()

    # Dibuja cada curva BER recibida y actualiza su leyenda.
    def plot_ber_curve(self, legend_label, ebno_dbs, ber_values):
        markers = {
            "Sin código": "o-",
            "Código Polar": "s-",
            f"Código Polar + {self.combo_crc.currentText()}": "^-",
            "Código LDPC": "D-"
        }
        style = markers.get(legend_label, "d-")
        self.ax.plot(ebno_dbs, ber_values, style, linewidth=2, markersize=6, label=legend_label)
        self.ax.legend(loc="best", fontsize=10)
        self.canvas.draw()
        self.log(f"✔ Plotted curve: {legend_label}")

    # Restaura los controles al finalizar todas las simulaciones BER.
    def on_ber_finished(self):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        self.btn_run_mc.setEnabled(True)
        self.btn_run_ber.setEnabled(True)
        self.log("✅ All BER curves simulated and plotted.")

    # Informa errores de los hilos y restablece el estado de la interfaz.
    def on_worker_error(self, err_msg):
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.btn_run_mc.setEnabled(True)
        self.btn_run_ber.setEnabled(True)
        self.btn_stop_mc.setEnabled(False)
        self.log(f"❌ ERROR: {err_msg}")
        QMessageBox.critical(self, "Simulation Error", err_msg)


# Punto de entrada: crea la aplicación Qt y muestra la ventana principal.
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PolarSimGUI()
    window.show()
    sys.exit(app.exec())