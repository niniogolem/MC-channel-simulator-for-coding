# Ejemplo simple de canal plano Rayleigh con ecualizador ZF
# --- Parámetros del sistema ---
batch_size = 4
N = 8  # símbolos por bloque
bitsxsimb = 2
mod = sionna.phy.mapping.Constellation("qam", bitsxsimb)
mapper = sionna.phy.mapping.Mapper(constellation=mod)
demapper = sionna.phy.mapping.Demapper("app", constellation=mod)

# --- Canal plano Rayleigh + AWGN ---
generador_rayleigh = sionna.phy.channel.GenerateFlatFadingChannel(num_tx_ant=1, num_rx_ant=1)
canal_awgn = sionna.phy.channel.AWGN()

# --- Eb/No y densidad espectral de ruido ---
ebno_db = tf.constant(30.0, dtype=tf.float32)
no = sionna.phy.utils.ebnodb2no(ebno_db, bitsxsimb, coderate=1.0)  # tf.float32
no_c = tf.cast(no, tf.complex64)

# --- Bits aleatorios y mapeo a símbolos ---
bits = tf.random.uniform([batch_size, N * bitsxsimb], maxval=2, dtype=tf.int32)
x = mapper(bits)  # [batch_size, N] tf.complex64

# --- Canal plano por bloque ---
h = generador_rayleigh(batch_size)  # [batch_size,1,1]
y_pre = h * x  # difusión automática
y = canal_awgn(y_pre, no)  # señal recibida con ruido

# --- ECUALIZADOR ZF ---
h_plano = tf.reshape(h, [-1, 1])  # [batch_size, 1]
abs_h2_c = tf.cast(tf.abs(h_plano)**2, tf.complex64)
zf = tf.math.conj(h_plano) / abs_h2_c
x_sombrero = zf * y  # señal ecualizada

# --- Varianza de ruido post-ZF ---
no_efec = no_c / abs_h2_c
no_efec_real = tf.cast(tf.math.real(no_efec), tf.float32)
no_efec_broadcast = tf.ones_like(x_sombrero, dtype=tf.float32) * no_efec_real

# --- Demapeo suave (LLRs) ---
llr = demapper(x_sombrero, no_efec_broadcast)

# --- Decisión dura (0 o 1 por bit) ---
bits_estimados = tf.cast(llr < 0.0, tf.int32)

# --- Cálculo de errores ---
errores = tf.reduce_sum(tf.cast(bits != bits_estimados, tf.int32), axis=1)
ber = tf.reduce_mean(tf.cast(errores, tf.float32)) / (N * bitsxsimb)

# --- Impresión de resultados ---
tf.print("\nBits reales:", bits[0])
tf.print("Bits estimados:", bits_estimados[0])
tf.print("Símbolos reales:", x[0][:5])
tf.print("Símbolos recibidos y ecualizados:", x_sombrero[0][:5])
tf.print("BER promedio:", ber)
