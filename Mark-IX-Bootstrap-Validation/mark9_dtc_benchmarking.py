
# Mark 9: Pure Physics DTC Lifetime vs Noise with Bootstrap Error Bars (EPSILON=0.02, NOISE_P2=2*NOISE_P1)
# Pure Science & Anti-Cheat: no penalty terms, no magic constants; failures are logged explicitly.
# Rigoritas Numerik: float64/complex128 default (NumPy/Qiskit).
# Self-Audit Ledger at end; HALUSINASI DETECTED if trend flat or illogical.

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import DensityMatrix, partial_trace, purity

# =========================
# Konfigurasi Mark 9
# =========================
CFG = {
    "N_QUBITS": 6,
    "N_CYCLES": 30,
    "SAMPLES": 16,             # lebih banyak sample untuk bootstrap yang stabil
    "EPSILON": 0.02,
    "H_STRENGTH": 4.0,
    "J_INT": 0.25,
    "SEED_GLOBAL": 20250101,
    "NOISE_P1_VALUES": np.logspace(-4, -2, 12, dtype=np.float64),
    "OPTIMIZATION_LEVEL": 0,
    "ENVELOPE_MODE": "abs",    # "abs" | "demod"
    "TAU_MIN_GREEN": 100.0,
    "PLOT": True,
    "CSV_LEDGER": "data/mark9_tau_ledger.csv",
    "BOOTSTRAP_N": 1000,       # jumlah bootstrap resamples
    "BOOTSTRAP_CI": 0.68,      # interval kepercayaan ~1-sigma
}

# =========================
# Numerik ketat
# =========================
np.set_printoptions(precision=6, suppress=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.makedirs("data", exist_ok=True)

# =========================
# Noise model sinkron (NOISE_P2=2*NOISE_P1)
# =========================
def get_noise_model_synced(p1: float):
    p2 = 2.0 * p1
    nm = NoiseModel()
    nm.add_all_qubit_quantum_error(depolarizing_error(p1, 1), ["rx", "rz"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["rzz"])
    return nm, p2

# =========================
# Physics builders (Floquet)
# =========================
def build_initial_neel(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for i in range(1, n_qubits, 2):
        qc.x(i)
    return qc

def append_floquet_cycle(qc: QuantumCircuit,
                         epsilon: float,
                         h_fields: np.ndarray,
                         J_int: float):
    n = qc.num_qubits
    for q in range(n):
        qc.rx(np.pi * (1.0 - epsilon), q)
    for q in range(n):
        qc.rz(2.0 * h_fields[q], q)
    for q in range(n - 1):
        qc.rzz(2.0 * J_int, q, q + 1)

def build_circuit_with_snapshots(cfg, h_fields: np.ndarray) -> QuantumCircuit:
    qc = build_initial_neel(cfg["N_QUBITS"])
    for t in range(cfg["N_CYCLES"]):
        append_floquet_cycle(qc, cfg["EPSILON"], h_fields, cfg["J_INT"])
        qc.save_density_matrix(label=f"rho_{t}")
    return qc

# =========================
# Observables
# =========================
def single_qubit_z_expectation(rho_dm: DensityMatrix, i: int, n_qubits: int) -> float:
    z_op = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    rho_i = partial_trace(rho_dm, [j for j in range(n_qubits) if j != i])
    val = np.real(np.trace(rho_i.data @ z_op))
    return float(val)

def staggered_magnetization(rho_dm: DensityMatrix, n_qubits: int) -> float:
    mag = 0.0
    for i in range(n_qubits):
        val = single_qubit_z_expectation(rho_dm, i, n_qubits)
        mag += ((-1) ** i) * val
    return mag / n_qubits

def half_chain_purity(rho_dm: DensityMatrix, n_qubits: int) -> float:
    left = list(range(n_qubits // 2))
    right = list(range(n_qubits // 2, n_qubits))
    rho_left = partial_trace(rho_dm, right)
    return float(purity(rho_left))

# =========================
# Runner per sample
# =========================
def run_sample(cfg, sim: AerSimulator, sample_id: int):
    seed = cfg["SEED_GLOBAL"] + sample_id
    rng = np.random.default_rng(seed)
    h_rand = rng.uniform(-cfg["H_STRENGTH"], cfg["H_STRENGTH"], cfg["N_QUBITS"]).astype(np.float64)

    qc = build_circuit_with_snapshots(cfg, h_rand)
    tqc = transpile(qc, sim, optimization_level=cfg["OPTIMIZATION_LEVEL"])
    result = sim.run(tqc).result()
    data = result.data()

    mags = np.empty(cfg["N_CYCLES"], dtype=np.float64)
    purities = np.empty(cfg["N_CYCLES"], dtype=np.float64)

    for t in range(cfg["N_CYCLES"]):
        rho = data[f"rho_{t}"]
        rho_dm = rho if isinstance(rho, DensityMatrix) else DensityMatrix(rho)
        mags[t] = staggered_magnetization(rho_dm, cfg["N_QUBITS"])
        purities[t] = half_chain_purity(rho_dm, cfg["N_QUBITS"])

    return mags, purities

# =========================
# Envelope helpers
# =========================
def dtc_envelope(mags: np.ndarray, mode: str = "abs") -> np.ndarray:
    if mode == "abs":
        return np.abs(mags)
    elif mode == "demod":
        t = np.arange(len(mags), dtype=np.int64)
        return np.abs(((-1) ** t) * mags)
    else:
        return np.abs(mags)

# =========================
# Lifetime estimator (no penalty/fudge)
# =========================
def fit_exponential_envelope(abs_mags: np.ndarray, t_arr: np.ndarray = None):
    if t_arr is None:
        t_arr = np.arange(len(abs_mags), dtype=np.float64)
    x = t_arr.astype(np.float64)
    y = abs_mags.astype(np.float64)

    y_min = float(np.min(y))
    best_tau = None
    best_err = np.inf
    best_model = ("exp", 1.0, 0.0)

    C_candidates = np.linspace(0.0, 0.9 * y_min, 41, dtype=np.float64)
    for C in C_candidates:
        y_shift = y - C
        if np.any(y_shift <= 0.0):
            continue
        ln_y = np.log(y_shift)
        slope, intercept = np.polyfit(x, ln_y, 1)
        if slope >= 0.0 or not np.isfinite(slope):
            continue
        tau = -1.0 / slope
        if not np.isfinite(tau) or tau <= 0.0:
            continue
        y_fit = np.exp(intercept) * np.exp(-x / tau) + C
        err = float(np.mean((y_fit - y) ** 2))
        if err < best_err:
            best_err = err
            best_tau = tau
            best_model = ("exp", 1.0, C)

    return float(best_tau) if best_tau is not None else np.nan, best_model, best_err

# =========================
# Bootstrap estimator for τ
# =========================
def bootstrap_tau(env_samples: np.ndarray, fit_fn, n_boot: int, rng: np.random.Generator):
    S, T = env_samples.shape
    taus = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, S, size=S)  # resample indices with replacement
        env_mean = np.mean(env_samples[idx], axis=0)
        tau_b, _, _ = fit_fn(env_mean)
        taus[b] = tau_b if np.isfinite(tau_b) else np.nan
    taus = taus[np.isfinite(taus)]
    if len(taus) == 0:
        return np.nan, np.nan, (np.nan, np.nan)
    tau_mean = float(np.mean(taus))
    tau_std = float(np.std(taus, ddof=1))
    alpha = (1.0 - CFG["BOOTSTRAP_CI"]) / 2.0
    lo = float(np.quantile(taus, alpha))
    hi = float(np.quantile(taus, 1.0 - alpha))
    return tau_mean, tau_std, (lo, hi)

# =========================
# CSV ledger
# =========================
def write_csv_header(path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "NOISE_P1","NOISE_P2","EPSILON","tau_mean","tau_std","tau_ci_lo","tau_ci_hi",
            "model","C","fit_mse","final_purity"
        ])

def append_csv_row(path, p1, p2, epsilon, tau_mean, tau_std, tau_ci, model, C, mse, final_purity):
    import csv
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            f"{p1:.8f}", f"{p2:.8f}", f"{epsilon:.4f}",
            f"{tau_mean:.6f}", f"{tau_std:.6f}", f"{tau_ci[0]:.6f}", f"{tau_ci[1]:.6f}",
            model, f"{C:.6f}", f"{mse:.6e}", f"{final_purity:.6f}"
        ])

# =========================
# Sweep NOISE_P1 dan plotting
# =========================
def main():
    os.makedirs("data", exist_ok=True)

    cfg = CFG.copy()
    print("=== Mark 9 Parameter Ledger ===")
    for k, v in cfg.items():
        if k == "NOISE_P1_VALUES":
            print(f"- {k}: [{v[0]:.5f} .. {v[-1]:.5f}] ({len(v)} pts)")
        else:
            print(f"- {k}: {v}")
    print("================================")

    p1_values = cfg["NOISE_P1_VALUES"].astype(np.float64)
    tau_mean_list = []
    tau_std_list = []
    tau_ci_list = []
    model_types = []
    Cs = []
    mse_list = []
    final_purities = []
    mag_first = []
    mag_last = []

    if cfg["CSV_LEDGER"]:
        write_csv_header(cfg["CSV_LEDGER"])

    rng_boot = np.random.default_rng(cfg["SEED_GLOBAL"] + 777)

    for p1 in p1_values:
        noise_model, p2 = get_noise_model_synced(float(p1))
        sim = AerSimulator(method="density_matrix", noise_model=noise_model)

        mags_all = np.empty((cfg["SAMPLES"], cfg["N_CYCLES"]), dtype=np.float64)
        pur_all = np.empty((cfg["SAMPLES"], cfg["N_CYCLES"]), dtype=np.float64)
        for s in range(cfg["SAMPLES"]):
            mags, purities = run_sample(cfg, sim, sample_id=s)
            mags_all[s] = mags
            pur_all[s] = purities

        env_samples = np.abs(mags_all) if cfg["ENVELOPE_MODE"] == "abs" else np.abs(((-1) ** np.arange(cfg["N_CYCLES"])) * mags_all)
        env_mean = np.mean(env_samples, axis=0)

        tau_fit, model_info, mse = fit_exponential_envelope(env_mean)
        tau_mean, tau_std, tau_ci = bootstrap_tau(env_samples, fit_exponential_envelope, cfg["BOOTSTRAP_N"], rng_boot)

        tau_mean_list.append(tau_mean)
        tau_std_list.append(tau_std)
        tau_ci_list.append(tau_ci)
        model_types.append(model_info[0])
        Cs.append(model_info[2])
        mse_list.append(mse)
        final_purities.append(float(np.mean(pur_all, axis=0)[-1]))
        mag_first.append(float(env_mean[0]))
        mag_last.append(float(env_mean[-1]))

        status = "GREEN" if (np.isfinite(tau_mean) and tau_mean >= cfg["TAU_MIN_GREEN"]) else "RED"
        pm = f"±{tau_std:.3f}" if np.isfinite(tau_std) else "±nan"
        print(f"NOISE_P1={p1:.5f} | NOISE_P2={2.0*p1:.5f} | τ={tau_mean:.3f} {pm} | CI68=({tau_ci[0]:.3f},{tau_ci[1]:.3f}) | model={model_info} | MSE={mse:.3e} | status={status}")

        if cfg["CSV_LEDGER"]:
            append_csv_row(cfg["CSV_LEDGER"], p1, p2, cfg["EPSILON"], tau_mean, tau_std, tau_ci, model_info[0], model_info[2], mse, float(np.mean(pur_all, axis=0)[-1]))

    tau_mean_arr = np.array(tau_mean_list, dtype=np.float64)
    tau_std_arr = np.array(tau_std_list, dtype=np.float64)
    mse_arr = np.array(mse_list, dtype=np.float64)
    final_purities = np.array(final_purities, dtype=np.float64)
    mag_first = np.array(mag_first, dtype=np.float64)
    mag_last = np.array(mag_last, dtype=np.float64)

    if cfg["PLOT"]:
        plt.figure(figsize=(9, 6))
        plt.errorbar(p1_values, tau_mean_arr, yerr=tau_std_arr, fmt="s", capsize=3, lw=1.5)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("NOISE_P1 (depolarizing prob per 1-qubit gate)")
        plt.ylabel("Lifetime τ (cycles)")
        plt.title("Mark 9: DTC lifetime vs noise with bootstrap error bars")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

    # =========================
    # Audit Ledger (Self-Audit)
    # =========================
    valid = np.isfinite(tau_mean_arr) & (tau_mean_arr > 0.0)
    print("\n=== Audit Ledger ===")
    print(f"- Points used: {int(np.sum(valid))} / {len(p1_values)}")
    if np.sum(valid) >= 4:
        lx = np.log(p1_values[valid])
        ly = np.log(tau_mean_arr[valid])
        slope, intercept = np.polyfit(lx, ly, 1)
        avg_tau = float(np.mean(tau_mean_arr[valid]))
        min_tau = float(np.min(tau_mean_arr[valid]))
        max_tau = float(np.max(tau_mean_arr[valid]))
        avg_mse = float(np.mean(mse_arr[valid]))
        avg_final_purity = float(np.mean(final_purities[valid]))
        decay_ratio = float(np.mean(mag_last[valid] / mag_first[valid]))
        avg_err = float(np.mean(tau_std_arr[valid]))

        print(f"- log-log slope(τ vs noise): {slope:.3f}")
        print(f"- τ avg/min/max: {avg_tau:.3f} / {min_tau:.3f} / {max_tau:.3f} cycles")
        print(f"- Bootstrap τ std avg: {avg_err:.3f} cycles")
        print(f"- Fit MSE avg: {avg_mse:.3e}")
        print(f"- Final half-chain purity avg: {avg_final_purity:.4f}")
        print(f"- Envelope decay ratio (last/first): {decay_ratio:.3f}")
        print(f"- Energy conservation: FAILED (Floquet drive + depolarizing noise)")
        print(f"- Quantumness Limit: Purity {avg_final_purity:.4f} indicates "
              f"{'Mixed' if avg_final_purity < 0.2 else 'Coherent'} regime")

        flagged = False
        if abs(slope) < 0.2:
            print("HALUSINASI DETECTED: Lifetime slope ~0 on log-log. Check EPSILON dominance, envelope mode, or fit robustness.")
            flagged = True
        corr = np.corrcoef(p1_values[valid], tau_mean_arr[valid])[0, 1]
        if corr > -0.2:
            print("HALUSINASI DETECTED: Weak anti-correlation between noise and lifetime. Verify noise sync and disorder strength.")
            flagged = True
        if decay_ratio >= 0.95:
            print("HALUSINASI DETECTED: Envelope barely decays across 30 cycles; increase cycles or adjust parameter range.")
            flagged = True
        if avg_err <= 0.0 or not np.isfinite(avg_err):
            print("HALUSINASI DETECTED: Bootstrap variance invalid; check resampling and fit stability.")
            flagged = True
        if not flagged:
            print("- Status: OK (anti-cheat checks passed)")
    else:
        print("- Insufficient valid points for audit.")
        print("- Energy conservation: FAILED (Floquet drive + depolarizing noise)")
        print(f"- Quantumness Limit: Purity {np.nan:.4f} indicates Mixed/Coherent regime (indeterminate)")
        print("HALUSINASI DETECTED: τ invalid or insufficient; bootstrap/fitting failed across sweep.")

if __name__ == "__main__":
    main()
