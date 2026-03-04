# Monte Carlo channel simulator for bit channel selection in Polar coding

Python Notebook for simulation of a Wireless communication channel and Monte Carlo simulation of said channels for frozen bit-channel selection (which under Polar coding, changes with each channel). Some platform features include:
- Support for multiple channels, currently: AWGN, Rayleigh, Rice, Lognormal and Suzuki
- Support for ZF, MMSE and no equalizer on reception side
- Monte Carlo simulation with GPU/CPU capability for frozen bit channel selection in Polar Codes
- Plot BER vs EbNo curves, with comparison with uncoded and LDPC transmission

Copyright (c) 2026 Sergio Huaman Kemper

Part of this code uses Python Sionna modules, which are distributed under Apache-2.0 license:
Hoydis, J., Cammerer, S., Ait Aoudia, F., Nimier-David, M., Maggi, L., Marcus, G., Vem, A., & Keller, A. (2022). Sionna (Version 1.2.1) [Software]. https://nvlabs.github.io/sionna/
