
# Mark 8: Pure Physics DTC Lifetime vs Noise (EPSILON=0.02, NOISE_P2=2*NOISE_P1)
# Pure Science & Anti-Cheat: no penalty terms, no magic constants; failures are logged explicitly.
# Rigoritas Numerik: float64/complex128 default (NumPy/Qiskit).
# Self-Audit Ledger at end; HALUSINASI DETECTED if trend flat or illogical.

import numpy as np
import matplotlib.pyplot as plt
import warnings
from dataclasses import dataclass, field, asdict

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.quantum_info import DensityMatrix, partial_trace, purity

# =========================
# Konfigurasi Mark 8
# =========================
@dataclass
class Config:
    N_QUBITS: int = 6
    N_CYCLES: int = 30
    SAMPLES: int = 8
    EPSILON: float = 0.02
    H_STRENGTH: float = 4.0
    J_INT: float = 0.25
    SEED_GLOBAL: int = 20250101
    NOISE_P1_VALUES: np.ndarray = field(
        default_factory=lambda: np.logspace(-4, -2, 12, dtype=np.float64)
    )
    OPTIMIZATION_LEVEL: int = 0
    ENVELOPE_MODE: str = "abs"   # "abs" | "demod"
    TAU_MIN_GREEN: float = 100.0
    PLOT: bool = True
    CSV_LEDGER: str = None       # e.g., "mark8_tau_ledger.csv"
    AUDIT_VERBOSE: bool = True

CFG = Config()

# =========================
# Numerik ketat
# =========================
np.set_printoptions(precision=6, suppress=True)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# Utilities
# =========================
def print_parameter_ledger(cfg: Config):
    print("=== Mark 8 Parameter Ledger ===")
    d = asdict(cfg)
    for k, v in d.items():
        if k == "NOISE_P1_VALUES":
            arr = v
            print(f"- {k}: [{arr[0]:.5f} .. {arr[-1]:.5f}] ({len(arr)} pts)")
        else:
            print(f"- {k}: {v}")
    print("================================")

def write_csv_header(path: str):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "NOISE_P1","NOISE_P2","EPSILON","tau_fit","model","beta","C",
            "fit_mse","final_purity","envelope_first","envelope_last"
        ])

def append_csv_row(path: str, p1, p2, epsilon, tau_fit, model, beta, C, mse, final_purity, env_first, env_last):
    import csv
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            f"{p1:.8f}", f"{p2:.8f}", f"{epsilon:.6f}", f"{tau_fit:.6f}", model,
            f"{beta:.6f}", f"{C:.6f}", f"{mse:.6e}", f"{final_purity:.6f}",
            f"{env_first:.6f}", f"{env_last:.6f}"
        ])

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

def build_circuit_with_snapshots(cfg: Config, h_fields: np.ndarray) -> QuantumCircuit:
    qc = build_initial_neel(cfg.N_QUBITS)
    for t in range(cfg.N_CYCLES):
        append_floquet_cycle(qc, cfg.EPSILON, h_fields, cfg.J_INT)
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
def run_sample(cfg: Config, sim: AerSimulator, sample_id: int):
    seed = cfg.SEED_GLOBAL + sample_id
    rng = np.random.default_rng(seed)
    h_rand = rng.uniform(-cfg.H_STRENGTH, cfg.H_STRENGTH, cfg.N_QUBITS).astype(np.float64)

    qc = build_circuit_with_snapshots(cfg, h_rand)
    tqc = transpile(qc, sim, optimization_level=cfg.OPTIMIZATION_LEVEL)
    result = sim.run(tqc).result()
    data = result.data()

    mags = np.empty(cfg.N_CYCLES, dtype=np.float64)
    purities = np.empty(cfg.N_CYCLES, dtype=np.float64)

    for t in range(cfg.N_CYCLES):
        rho = data[f"rho_{t}"]
        rho_dm = rho if isinstance(rho, DensityMatrix) else DensityMatrix(rho)
        mags[t] = staggered_magnetization(rho_dm, cfg.N_QUBITS)
        purities[t] = half_chain_purity(rho_dm, cfg.N_QUBITS)

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
# Model selection (AIC) and fitting
# =========================
def aic_mse(mse: float, n_params: int, n_points: int) -> float:
    if mse <= 0.0:
        return np.inf
    return n_points * np.log(mse) + 2 * n_params

def fit_envelope_models(abs_mags: np.ndarray, t_arr: np.ndarray = None):
    if t_arr is None:
        t_arr = np.arange(len(abs_mags), dtype=np.float64)
    x = t_arr.astype(np.float64)
    y = abs_mags.astype(np.float64)

    y_min = float(np.min(y))
    n_points = len(y)

    best = {
        "tau": np.nan,
        "model": None,
        "beta": np.nan,
        "C": np.nan,
        "mse": np.inf,
        "aic": np.inf
    }

    C_candidates = np.linspace(0.0, 0.9 * y_min, 41, dtype=np.float64)

    # Exponential model
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
        mse = float(np.mean((y_fit - y) ** 2))
        aic = aic_mse(mse, n_params=3, n_points=n_points)
        if aic < best["aic"]:
            best.update({"tau": float(tau), "model": "exp", "beta": 1.0, "C": float(C), "mse": mse, "aic": aic})

    # Stretched exponential model
    beta_grid = np.linspace(0.6, 1.4, 17, dtype=np.float64)
    for beta in beta_grid:
        x_beta = x ** beta
        for C in C_candidates:
            y_shift = y - C
            if np.any(y_shift <= 0.0):
                continue
            ln_y = np.log(y_shift)
            slope, intercept = np.polyfit(x_beta, ln_y, 1)
            if slope >= 0.0 or not np.isfinite(slope):
                continue
            tau_beta = -1.0 / slope
            if not np.isfinite(tau_beta) or tau_beta <= 0.0:
                continue
            tau = tau_beta ** (1.0 / beta)
            y_fit = np.exp(intercept) * np.exp(-(x / tau) ** beta) + C
            mse = float(np.mean((y_fit - y) ** 2))
            aic = aic_mse(mse, n_params=4, n_points=n_points)
            if aic < best["aic"]:
                best.update({"tau": float(tau), "model": "stretched", "beta": float(beta), "C": float(C), "mse": mse, "aic": aic})

    return best

# =========================
# Sweep NOISE_P1 dan plotting
# =========================
def main():
    cfg = CFG
    print_parameter_ledger(cfg)

    p1_values = cfg.NOISE_P1_VALUES.astype(np.float64)
    taus_fit = []
    model_types = []
    betas = []
    Cs = []
    mse_list = []
    final_purities = []
    env_first = []
    env_last = []

    if cfg.CSV_LEDGER:
        write_csv_header(cfg.CSV_LEDGER)

    for p1 in p1_values:
        noise_model, p2 = get_noise_model_synced(float(p1))
        sim = AerSimulator(method="density_matrix", noise_model=noise_model)

        mags_all = np.empty((cfg.SAMPLES, cfg.N_CYCLES), dtype=np.float64)
        pur_all = np.empty((cfg.SAMPLES, cfg.N_CYCLES), dtype=np.float64)
        for s in range(cfg.SAMPLES):
            mags, purities = run_sample(cfg, sim, sample_id=s)
            mags_all[s] = mags
            pur_all[s] = purities

        mean_mags = np.mean(mags_all, axis=0).astype(np.float64)
        mean_pur = np.mean(pur_all, axis=0).astype(np.float64)

        envelope = dtc_envelope(mean_mags, mode=cfg.ENVELOPE_MODE)
        fit = fit_envelope_models(envelope)

        taus_fit.append(fit["tau"])
        model_types.append(fit["model"])
        betas.append(fit["beta"])
        Cs.append(fit["C"])
        mse_list.append(fit["mse"])
        final_purities.append(float(mean_pur[-1]))
        env_first.append(float(envelope[0]))
        env_last.append(float(envelope[-1]))

        status = "GREEN" if (np.isfinite(fit["tau"]) and fit["tau"] >= cfg.TAU_MIN_GREEN) else "RED"
        print(f"NOISE_P1={p1:.5f} | NOISE_P2={2.0*p1:.5f} | τ_fit={fit['tau']:.3f} | model={fit['model']}, beta={fit['beta']:.3f}, C={fit['C']:.4f} | MSE={fit['mse']:.3e} | AIC={fit['aic']:.2f} | status={status}")

        if cfg.CSV_LEDGER:
            append_csv_row(cfg.CSV_LEDGER, p1, p2, cfg.EPSILON, fit["tau"], fit["model"], fit["beta"], fit["C"], fit["mse"], float(mean_pur[-1]), envelope[0], envelope[-1])

    taus_fit = np.array(taus_fit, dtype=np.float64)
    mse_arr = np.array(mse_list, dtype=np.float64)
    final_purities = np.array(final_purities, dtype=np.float64)
    env_first = np.array(env_first, dtype=np.float64)
    env_last = np.array(env_last, dtype=np.float64)

    if cfg.PLOT:
        plt.figure(figsize=(9, 6))
        plt.loglog(p1_values, taus_fit, marker="s", lw=2, label="τ_fit (AIC-selected envelope model)")
        plt.xlabel("NOISE_P1 (depolarizing prob per 1-qubit gate)")
        plt.ylabel("Lifetime τ (cycles)")
        plt.title("Mark 8: DTC lifetime vs noise (EPSILON=0.02, NOISE_P2=2*NOISE_P1)")
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # =========================
    # Audit Ledger (Self-Audit)
    # =========================
    valid = np.isfinite(taus_fit) & (taus_fit > 0.0)
    print("\n=== Audit Ledger ===")
    print(f"- Points used: {int(np.sum(valid))} / {len(p1_values)}")
    if np.sum(valid) >= 4:
        lx = np.log(p1_values[valid])
        ly = np.log(taus_fit[valid])
        slope, intercept = np.polyfit(lx, ly, 1)
        avg_tau = float(np.mean(taus_fit[valid]))
        min_tau = float(np.min(taus_fit[valid]))
        max_tau = float(np.max(taus_fit[valid]))
        avg_mse = float(np.mean(mse_arr[valid]))
        frac_stretched = float(np.mean([1.0 if mt == "stretched" else 0.0 for mt in np.array(model_types)[valid]]))
        avg_beta = float(np.nanmean(np.array(betas)[valid]))
        avg_C = float(np.mean(np.array(Cs)[valid]))
        avg_final_purity = float(np.mean(final_purities[valid]))
        decay_ratio = float(np.mean(env_last[valid] / env_first[valid]))
        tau_monotonic = float(np.mean(np.diff(taus_fit[valid]) < 0.0))

        print(f"- log-log slope(τ vs noise): {slope:.3f}")
        print(f"- τ_fit avg/min/max: {avg_tau:.3f} / {min_tau:.3f} / {max_tau:.3f} cycles")
        print(f"- Envelope model: stretched fraction={frac_stretched:.2f}, avg beta={avg_beta:.3f}, avg C={avg_C:.5f}")
        print(f"- Fit MSE avg: {avg_mse:.3e}")
        print(f"- Final half-chain purity avg: {avg_final_purity:.4f}")
        print(f"- Envelope decay ratio (last/first): {decay_ratio:.3f}")
        print(f"- Energy conservation: FAILED (Floquet drive + depolarizing noise)")
        print(f"- Quantumness Limit: Purity {avg_final_purity:.4f} indicates "
              f"{'Mixed' if avg_final_purity < 0.2 else 'Coherent'} regime")
        print(f"- τ monotonic decrease fraction vs noise: {tau_monotonic:.2f}")

        flagged = False
        if abs(slope) < 0.2:
            print("HALUSINASI DETECTED: Lifetime slope ~0 on log-log. Check EPSILON dominance, envelope mode, or fit robustness.")
            flagged = True
        corr = np.corrcoef(p1_values[valid], taus_fit[valid])[0, 1]
        if corr > -0.2:
            print("HALUSINASI DETECTED: Weak anti-correlation between noise and lifetime. Verify noise sync and disorder strength.")
            flagged = True
        if decay_ratio >= 0.95:
            print("HALUSINASI DETECTED: Envelope barely decays across 30 cycles; increase cycles or adjust parameter range.")
            flagged = True
        if tau_monotonic < 0.6:
            print("HALUSINASI DETECTED: τ_fit does not decrease monotonically with noise; investigate fit selection and envelope definition.")
            flagged = True
        if not flagged:
            print("- Status: OK (anti-cheat checks passed)")
    else:
        print("- Insufficient valid points for audit.")
        print("- Energy conservation: FAILED (Floquet drive + depolarizing noise)")
        print(f"- Quantumness Limit: Purity {np.nan:.4f} indicates Mixed/Coherent regime (indeterminate)")
        print("HALUSINASI DETECTED: τ_fit invalid or insufficient; fitting failed across sweep.")

if __name__ == "__main__":
    main()
