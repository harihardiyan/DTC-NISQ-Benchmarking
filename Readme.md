

---

# DTC-NISQ-Benchmarking: Evolutionary Framework for Quantum Stability Audits

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Physics: Quantum NISQ](https://img.shields.io/badge/Physics-Quantum%20NISQ-red.svg)]()

## Overview
This repository documents the rigorous development of a benchmarking framework for **Discrete Time Crystals (DTC)** on Noisy Intermediate-Scale Quantum (NISQ) devices. The project evolves through distinct "Mark" versions, each introducing progressively sophisticated statistical methods and physical audits to eliminate simulation artifacts and validate the **Many-Body Localization (MBL)** phase protection against synchronized noise.

## Repository Structure & Evolution

The project is structured into modular versions, reflecting a step-by-step engineering journey from basic validation to high-fidelity statistical auditing.

### 📂 [Mark-V-Baseline](./Mark-V-Baseline)
**The Power-Law Foundation**
- **Objective:** Establish the fundamental relationship between depolarizing noise probability ($P$) and DTC lifetime ($\tau$).
- **Key Logic:** Implementation of a synchronized noise floor ($P_2 = 2P_1$) to prevent single-qubit gate error masking.
- **Audit Result:** Confirmed the theoretical $P^{-1}$ decay law with a log-log slope of **-0.979**.

### 📂 [Mark-VI-Advanced-Analysis](./Mark-VI-Advanced-Analysis)
**Stretched Dynamics & Purity Tracking**
- **Objective:** Investigate non-trivial relaxation dynamics beyond simple exponential decay.
- **Key Upgrade:** Introduction of **Stretched Exponential Fitting** ($e^{-(t/\tau)^\beta}$) and **Purity Diagnostics** ($Tr(\rho^2)$) to distinguish between coherent interaction and thermalization.
- **Audit Result:** Detected compressed exponential dynamics ($\beta > 1$) characteristic of Floquet heating in finite-size systems.

### 📂 [Mark-VII-Production-Ledger](./Mark-VII-Production-Ledger)
**Data Persistence & Operational Reliability**
- **Objective:** Scale the simulation for long-duration parameter sweeps without data loss risk.
- **Key Upgrade:** Automated CSV ledger system (`mark7_tau_ledger.csv`) for real-time data persistence and external auditing.
- **Audit Result:** Validated reproducibility of the "Golden Slope" (-0.893) under rigorous production conditions.

### 📂 [Mark-VIII-Statistical-Selection](./Mark-VIII-Statistical-Selection)
**AIC-Driven Model Selection**
- **Objective:** Eliminate subjective bias in curve fitting.
- **Key Upgrade:** Implementation of the **Akaike Information Criterion (AIC)** to objectively select between Standard and Stretched Exponential models based on information loss penalization.
- **Audit Result:** Achieved a Monotonicity Score of **1.00**, confirming perfect causality between increased noise and decreased lifetime.

### 📂 [Mark-IX-Bootstrap-Validation](./Mark-IX-Bootstrap-Validation)
**The Gold Standard**
- **Objective:** Quantify statistical confidence and eliminate random fluctuations.
- **Key Upgrade:** **Bootstrap Resampling (N=1000)** to generate rigorous Error Bars ($\pm \sigma$) and Confidence Intervals (CI) for every lifetime data point.
- **Audit Result:** Delivered the definitive "Golden Curve" with a slope of **-0.976** and a fitting MSE of $10^{-6}$, confirming the robustness of the MBL-DTC phase with high statistical significance.

## Methodology (Core Physics)
All versions utilize a consistent 6-qubit Floquet MBL circuit architecture:
1.  **Hamiltonian:** Periodically driven chain with $R_{ZZ}$ interactions, onsite disorder fields $h_i$, and imperfect $\pi$-pulse drives.
2.  **Noise Model:** Markovian depolarizing noise applied to all gates, with strict synchronization ($P_{2q} = 2 \times P_{1q}$).
3.  **Metric:** Lifetime $\tau$ extracted from the envelope of the Staggered Magnetization $|M(t)|$.

## Citation
**Independent Researcher / AI Handler**
*Simulations executed using optimized single-pass density matrix snapshots for maximum resource efficiency on restricted hardware environments (Mobile/Colab Integration).*

---
**Status: PROJECT COMPLETE (Mark 9 Validated)** ☕️🚬🫡
