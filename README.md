# Monte Carlo channel simulator for bit channel selection in Polar coding

Python Notebook for simulation of a Wireless communication channel and Monte Carlo simulation of said channels for frozen bit-channel selection (which under Polar coding, changes with each channel). Some platform features include:
- Support for multiple channels, currently: AWGN, Rayleigh, Rice, Lognormal and Suzuki
- Support for ZF, MMSE and no equalizer on reception side
- Monte Carlo simulation with GPU/CPU capability for frozen bit channel selection in Polar Codes
- Plot BER vs EbNo curves, with comparison with uncoded and LDPC transmission

Copyright (c) 2026 Sergio Huaman Kemper

El desarrollador empleo Sionna para el desarrollo del presente simulador:
@software{sionna,
 title = {Sionna},
 author = {Hoydis, Jakob and Cammerer, Sebastian and {Ait Aoudia}, Fayçal and Nimier-David, Merlin and Maggi, Lorenzo and Marcus, Guillermo and Vem, Avinash and Keller, Alexander},
 note = {<https://nvlabs.github.io/sionna/>},
 year = {2022},
 version = {1.2.1}
}
