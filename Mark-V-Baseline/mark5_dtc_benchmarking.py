import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import DensityMatrix, partial_trace

# --- Configuration Ledger ---
CFG = {
    "N_QUBITS": 6,
    "N_CYCLES": 30,
    "SAMPLES": 5,
    "EPSILON": 0.02,
    "H_STRENGTH": 4.0,
    "J_INT": 0.25,
    "SEED_GLOBAL": 1234,
    "NOISE_P1_VALUES": np.logspace(-4, -2, 10, dtype=np.float64),
    "OPTIMIZATION_LEVEL": 0,
    "OUTPUT_DIR": "data",
    "PLOT_DIR": "plots"
}

# Strict numerical settings
np.set_printoptions(precision=6, suppress=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_noise_model_synced(p1: float):
    p2 = 2.0 * p1
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rx", "rz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["rzz"])
    return nm

def build_circuit_with_snapshots(cfg, h_fields: np.ndarray) -> QuantumCircuit:
    n = cfg["N_QUBITS"]
    qc = QuantumCircuit(n)
    for i in range(1, n, 2): qc.x(i) # Neel state
    
    for t in range(cfg["N_CYCLES"]):
        # Floquet cycle
        for q in range(n): qc.rx(np.pi * (1.0 - cfg["EPSILON"]), q)
        for q in range(n): qc.rz(2.0 * h_fields[q], q)
        for q in range(n - 1): qc.rzz(2.0 * cfg["J_INT"], q, q + 1)
        qc.save_density_matrix(label=f"rho_{t}")
    return qc

def staggered_magnetization(rho_dm: DensityMatrix, n_qubits: int) -> float:
    z_op = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    mag = 0.0
    for i in range(n_qubits):
        rho_i = partial_trace(rho_dm, [j for j in range(n_qubits) if j != i])
        mag += ((-1) ** i) * np.real(np.trace(rho_i.data @ z_op))
    return float(mag / n_qubits)

def fit_exponential_envelope(abs_mags: np.ndarray) -> float:
    x = np.arange(len(abs_mags), dtype=np.float64)
    y = abs_mags.astype(np.float64)
    y_min = float(np.min(y))
    best_tau, best_err = None, np.inf
    
    C_candidates = np.linspace(0.0, 0.9 * y_min, 41, dtype=np.float64)
    for C in C_candidates:
        y_shift = y - C
        if np.any(y_shift <= 0.0): continue
        ln_y = np.log(y_shift)
        slope, intercept = np.polyfit(x, ln_y, 1)
        if slope >= 0.0 or not np.isfinite(slope): continue
        tau = -1.0 / slope
        y_fit = np.exp(intercept) * np.exp(-x / tau) + C
        err = float(np.mean((y_fit - y) ** 2))
        if err < best_err:
            best_err, best_tau = err, tau
    return float(best_tau) if best_tau is not None else np.nan

def main():
    os.makedirs(CFG["OUTPUT_DIR"], exist_ok=True)
    os.makedirs(CFG["PLOT_DIR"], exist_ok=True)
    
    taus_fit = []
    p1_vals = CFG["NOISE_P1_VALUES"]

    for p1 in p1_vals:
        noise, _ = get_noise_model_synced(float(p1))
        sim = AerSimulator(method="density_matrix", noise_model=noise)
        
        mags_all = []
        for s in range(CFG["SAMPLES"]):
            rng = np.random.default_rng(CFG["SEED_GLOBAL"] + s)
            h_rand = rng.uniform(-CFG["H_STRENGTH"], CFG["H_STRENGTH"], CFG["N_QUBITS"])
            qc = build_circuit_with_snapshots(CFG, h_rand)
            tqc = transpile(qc, sim, optimization_level=CFG["OPTIMIZATION_LEVEL"])
            data = sim.run(tqc).result().data()
            mags = [staggered_magnetization(data[f"rho_{t}"], CFG["N_QUBITS"]) for t in range(CFG["N_CYCLES"])]
            mags_all.append(mags)
            
        mean_abs_mag = np.abs(np.mean(mags_all, axis=0))
        tau = fit_exponential_envelope(mean_abs_mag)
        taus_fit.append(tau)
        print(f"P1={p1:.5f} | tau={tau:.3f}")

    taus_fit = np.array(taus_fit)
    valid = np.isfinite(taus_fit)
    lx, ly = np.log(p1_vals[valid]), np.log(taus_fit[valid])
    slope, _ = np.polyfit(lx, ly, 1)

    print("\n--- Audit Ledger ---")
    print(f"Log-log Slope: {slope:.3f} (Ideal: -1.000)")
    print(f"Status: {'PASSED' if abs(slope + 1.0) < 0.1 else 'INVESTIGATE'}")

    plt.figure(figsize=(8, 6))
    plt.loglog(p1_vals, taus_fit, 's-', lw=2)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.xlabel("P1 Noise"); plt.ylabel("Lifetime tau"); plt.title("DTC Noise Benchmarking")
    plt.savefig(f"{CFG['PLOT_DIR']}/bifurcation_plot.png")

if __name__ == "__main__":
    main()
