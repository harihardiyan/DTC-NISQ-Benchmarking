This is a formal, academic-grade GitHub README draft for your project. The tone is objective, technical, and free of marketing fluff.

---

# DTC-NISQ-Benchmarking: Quantitative Stability Analysis of Discrete Time Crystals under Synchronized Noise

## Abstract
This repository provides a numerical framework for evaluating the stability of **Discrete Time Crystal (DTC)** phases on **Noisy Intermediate-Scale Quantum (NISQ)** architectures. The simulation focuses on the noise sensitivity of DTC lifetimes ($\tau$) within the **Many-Body Localization (MBL)** regime. Unlike idealized simulations, this project implements a synchronized noise model ($P_2 = 2P_1$) to replicate the coherence limitations of contemporary quantum hardware.

## Core Features
1. **Physical Integrity:** No artificial penalty terms or convergence manipulation. System decay is reported as observed under open-system dynamics.
2. **Synchronized Noise Floor:** Eliminates simulation bias by locking 1-qubit and 2-qubit error probabilities ($P_2 = 2P_1$).
3. **Exponential Envelope Fitting:** Extracts $\tau$ via log-linear regression with a grid search for the offset constant ($C$), accounting for the period-2 oscillation of the DTC order parameter.
4. **Self-Auditing Mechanism:** Automated calculation of the log-log slope ($\tau$ vs. Noise). A gradient near -1.0 validates the model's consistency with Markovian Depolarization theory.

## Methodology

### 1. Circuit Architecture
The system utilizes a 6-qubit Floquet chain with the following Hamiltonian cycle repeated for 30 steps:
- **Interaction:** $R_{ZZ}$ gates with coupling $J=0.25$.
- **Disorder:** On-site $R_Z$ rotations with random amplitudes $h \in [-4.0, 4.0]$.
- **Drive:** Imperfect $\pi$-pulses ($R_X$) with rotation error $\epsilon=0.02$.

### 2. Observables
The stability is measured via **Staggered Magnetization ($M$):**
$$M(t) = \frac{1}{N} \sum_{i=1}^{N} (-1)^i \langle Z_i(t) \rangle$$
The envelope of $|M(t)|$ is used to determine the lifetime $\tau$.

### 3. Noise Model
Depolarizing channels are applied to every gate:
- $P_1$ (Single-qubit probability): Swept from $10^{-4}$ to $10^{-2}$.
- $P_2$ (Two-qubit probability): $2 \times P_1$.

## Empirical Results (Audit Trail)

Experimental execution yields the following benchmark data:

| Noise $P_1$ | Lifetime $\tau$ (Cycles) | Status |
| :--- | :--- | :--- |
| 0.00010 | 1627.190 | GREEN (Coherent DTC) |
| 0.00129 | 131.745 | GREEN (MBL Protected) |
| 0.01000 | 18.164 | RED (Thermalized) |

### Audit Ledger Summary
- **Log-log Slope:** `-0.979` (Matches theoretical $P^{-1}$ prediction within 3% deviation).
- **Energy Conservation:** FAILED (Expected; open Floquet systems are inherently non-conservative).
- **Inference:** The DTC lifetime follows a power-law decay relative to noise strength, confirming the fragility of the MBL phase under depolarizing noise in NISQ devices.

## Installation & Usage

### Prerequisites
- Python 3.10+
- `qiskit`, `qiskit-aer`
- `numpy`, `matplotlib`, `pandas`

### Running the Benchmark
```bash
python mark5_dtc_benchmarking.py
```

## Repository Structure
- `mark5_dtc_benchmarking.py`: Core simulation and fitting engine.
- `mark5_tau_ledger.csv`: Raw data from the parameter sweep.
- `plots/`: Visualizations of the decay curves and log-log analysis.

## Citation/Contact
**Independent Researcher / AI Handler**
*Simulations executed using single-pass density matrix snapshots for resource efficiency on restricted hardware environments.*

---
*End of Audit Report.* ☕️🚬🫡
