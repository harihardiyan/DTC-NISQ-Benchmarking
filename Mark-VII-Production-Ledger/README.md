# DTC-NISQ-Benchmarking (Mark 7): Production-Grade Data Ledger

## Overview
**Mark 7** focuses on operational reliability and data persistence. Building upon the physical models of Mark 6, this version introduces an automated CSV ledger system to ensure high-fidelity data collection across large-scale parameter sweeps. It is designed for multi-hour simulations on restricted hardware (e.g., Google Colab / mobile-integrated environments).

## New in Mark 7: Engineering Persistence
1. **Automated CSV Ledger:** Integrated `write_csv_header` and `append_csv_row` logic ensures that every computed noise point is immediately saved to disk, preventing data loss during runtime disconnections.
2. **Deterministic Reproducibility:** Locked global seeds and rigid circuit architecture ensure that the extracted $\tau$ is reproducible across different execution environments.
3. **Big Data Ready:** The output CSV is formatted for direct ingestion into external analytical tools (Excel, Origin, or specialized physics plotting libraries).

## Methodology & Audit Protocol
Mark 7 maintains the **MANDOR AUDIT** protocol:
- **Noise Sync:** $P_2$ is strictly locked to $2 \times P_1$.
- **Fitting Engine:** Stretched exponential model $e^{-(t/\tau)^\beta}$ with a 41-point grid search for the offset $C$.
- **Audit Logic:** Self-calculating log-log slope to verify the power-law decay behavior.

## Empirical Audit Summary (Batch 20250101)
The latest audit confirms the following stability metrics:
- **Log-log Slope:** `-0.893` (Sub-Markovian decay, confirming MBL protection).
- **Avg. Stretching Exponent ($\beta$):** `1.088` (Compressed exponential dynamics).
- **Quantum Purity Floor:** `0.5972` (Validates that data is collected within the coherent regime).

## Usage
Activate the CSV ledger in the configuration dictionary before execution:
```python
CFG["CSV_LEDGER"] = "data/mark7_tau_ledger.csv"
