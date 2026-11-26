# Monte Carlo Channel Simulator for Bit-Channel Selection in Polar Coding

Wireless channel simulation platform and Monte Carlo bit-channel reliability designer for **Polar codes** (where optimal frozen bit-channel selection depends on the underlying channel). Built on **TensorFlow** and **NVIDIA Sionna**, the software is available in two versions:

- **Desktop GUI (`gui.py`):** Full PyQt6 graphical interface (in Spanish) for configuring parameters, running Monte Carlo designs, simulating BER curves and exporting CSV datasets without editing code.
- **Jupyter Notebook (`main.ipynb`):** Interactive step-by-step notebook version (with Google Colab support) for research, prototyping, and custom experiments.

---

## Key Features

### Channel & Receiver Models (XLA-Accelerated)
- **8 wireless channel models:**
  - **AWGN** — Additive white Gaussian noise
  - **Rayleigh** — Multipath fading (NLOS)
  - **Rice** — Multipath fading with LOS component ($K$-factor)
  - **Lognormal** — Slow shadowing ($\sigma$)
  - **Suzuki** — Composite Rayleigh × Lognormal (fast + slow fading)
  - **Nakagami-$m$** — Generalized fading ($m \ge 0.5$, XLA-native Marsaglia–Tsang sampler)
  - **Shadowed-Rician** — Satellite LMS / 5G NTN model (Nakagami-$m$ shadowed LOS + Rayleigh NLOS)
  - **Impulsive** — Bernoulli–Gaussian impulsive noise mixture (normalized to total noise power $N_0$)
- **Coherence block control:** Configurable sub-block fading (default `N // 16`) or one-click **flat fading (`bloq_coher = N`)** across the entire codeword.
- **Equalizers:** `MMSE`, `ZF`, and `no_eq` (automatically restricted to non-fading channels).
- **Modulation & coding:** BPSK, QPSK, and 16-QAM; codeword lengths $N \in [32, 8192]$; standard predefined code rates (`1/8` to `7/8`); configurable LLR clipping and SCL list size (`1` to `32`).

### Monte Carlo Polar Design & BER Evaluation
- **Automatic Design SNR (`⚡ Auto` in GUI):** Estimates the Monte Carlo design $E_b/N_0$ via bit-level Mutual Information bisection across any channel, fading parameters, modulation, equalizer, and code rate.
- **Crash-safe backups (`.npz`):** Periodically saves progress with explicit bit-channel indexing and full parameter metadata; supports resuming, discarding, or updating GUI parameters directly from an existing backup file.
- **BER vs. $E_b/N_0$ comparison:** Simulates and plots **Uncoded**, **Polar (SCL)**, **Polar + CRC** (3GPP 5G NR `CRC24A/B/C`, `CRC16`, `CRC11`, `CRC6`), and **5G NR LDPC** curves.
- **CSV export & import:** Load/save `BER_bits` CSV files and export simulated BER vs. $E_b/N_0$ curves to Excel-compatible UTF-8 CSVs.
- **Hardware & stability safeguards:** GPU/CPU automatic detection (plus `--cpu` mode), pre-run compute/RAM resource checks recommending scaled-down parameters on CPU-only or low-RAM machines, live RAM monitoring with emergency backup guard, and clean cancellation/shutdown.

---

## Repository Structure

- `gui.py` — Standalone PyQt6 desktop application
- `main.ipynb` — Interactive Jupyter Notebook version
- `frozen_bit_analysis.py` — Offline analysis and visualization of frozen-bit distributions
- `resources/sionna_env_linux.yml` — Minimal Conda environment for Linux
- `resources/sionna_env_windows.yml` — Minimal Conda environment for Windows (WSL2)
- `install_instructions_en.txt` — Step-by-step setup guide (English) for clean Windows + WSL2 machines
- `install_instructions.txt` — Step-by-step setup guide (Spanish) for clean Windows + WSL2 machines
- `results_*/` — Saved reference `BER_bits` CSV datasets and comparison plots

---

## Installation & Usage

### Linux
```bash
conda env create -f resources/sionna_env_linux.yml
conda activate sionna
```

### Windows (WSL2 — recommended for NVIDIA GPU acceleration)
```bash
conda env create -f resources/sionna_env_windows.yml
conda activate sionna
```
See [`install_instructions_en.txt`](install_instructions_en.txt) (English) or [`install_instructions.txt`](install_instructions.txt) (Spanish) for complete step-by-step Windows + WSL2 setup instructions (including Qt6/WSLg dependencies and troubleshooting).

### Running the Software
- **Desktop GUI (GPU if available, otherwise CPU):**
  ```bash
  python gui.py
  ```
- **Desktop GUI forcing CPU mode:**
  ```bash
  python gui.py --cpu
  ```
- **Jupyter Notebook:**
  ```bash
  jupyter lab main.ipynb
  ```

---

## License & Citation

Copyright (c) 2026 Sergio Huaman Kemper. See [`LICENSE`](LICENSE) for details.

Part of this code uses Python Sionna modules, which are distributed under the Apache-2.0 license:
Hoydis, J., Cammerer, S., Ait Aoudia, F., Nimier-David, M., Maggi, L., Marcus, G., Vem, A., & Keller, A. (2022). *Sionna* (Version 1.2.2) [Software]. https://nvlabs.github.io/sionna/
