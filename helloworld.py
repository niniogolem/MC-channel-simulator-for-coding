import numpy as np
import keras
from keras import layers
from keras import ops
import sionna.phy
import tensorflow as tf

batch_size = 20000
n = 1000 # codeword length
k = 500 # information bits per codeword
m = 4 # bits per symbol
snr = 10

b = sionna.phy.mapping.BinarySource()([batch_size, k])
c = sionna.phy.fec.ldpc.encoding.LDPC5GEncoder(k, n)(b)
x = sionna.phy.mapping.Mapper("qam", m)(c)
y = sionna.phy.channel.AWGN()([x, 1/snr])
llr = sionna.phy.mapping.Demapper("app", "qam", m)([y, 1/snr])
b_hat = sionna.phy.fec.ldpc.decoding.LDPC5GDecoder(sionna.phy.fec.ldpc.encoding.LDPC5GEncoder(k, n))(llr)

class E2E(tf.keras.Model):
    def __init__(self):
        super().__init__
        self.binary_source = sionna.phy.mapping.BinarySource()
        self.encoder = sionna.phy.fec.ldpc.encoding.LDPC5GEncoder(k, n)
        self.decoder = sionna.phy.fec.ldpc.decoding.LDPC5GDecoder(self.encoder)
        self.mapper = sionna.phy.mapping.Mapper("qam", m)
        self.demapper = sionna.phy.mapping.Demapper("app", "qam", m)
        self.channel = sionna.phy.channel.AWGN()

    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        b = self.binary_source([batch_size, k])
        c = self.encoder(b)
        x = self.mapper(c)
        no = sionna.phy.utils.ebnodb2no(ebno_db, m, k/n)
        y = self.channel([x, no])
        llr = self.demapper([y, no])
        b_hat = self.decoder(llr)
        return b, b_hat

compute_ber(*e2e(20000, 5))

ber_plot = PlotBER("AWGN Channel")
ber_plot.simulate(e2e,
                  ebno_dbs = np.arange(0, 6),
                  batch_size = 20000,
                  num_target_blocks = 100,
                  legend = "5G LDPC 16QAM",
                  soft_estimates = True,
                  max_mc_iter = 1000,
                  forward_keyboard_interrupt = False,
                  show_fig = True);
