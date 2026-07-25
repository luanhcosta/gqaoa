# GQAOA — Quantum Portfolio Optimization

A GPT2 transformer used as a learned autoregressive sampler over QAOA circuit
angle-parameters ("Generative QAOA" / GQAOA), trained via a custom
log-probability-matching loss, compared against two classical baselines
(PennyLane gradient descent + SPSA, and scipy.optimize).

This is a from-scratch reorganization of `local-quantum-portfolio-optimization`
into an installable, modular, tested Python package. See `pyproject.toml` for
the package layout.

## Setup

```bash
./setup.sh                         # creates .venv, installs CUDA torch + all extras
source .venv/bin/activate
```

Or manually, without a GPU (e.g. to run tests only):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Start the MLflow UI (tracking DB lives at `artifacts/mlflow.db`, created on first run):

```bash
mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --allowed-hosts '*' --cors-allowed-origins '*'
```

| Experiment | MLflow experiment name |
|---|---|
| Stability check (main experiment) | `gqaoa-stability` |
| Bracket strategy | `gqaoa-bracket` |
| HPO | `gqaoa-hpo` |
| Benchmark (classical GD baseline) | `gqaoa-benchmark-gd` |

---

## Problem

Minimize `energy_min` — the ground state energy of a QUBO/Ising Hamiltonian for a 10-asset portfolio optimization problem.

- **Best known value:** -0.795 (GQAOA + annealing, 1000 QPU calls)
- **QPU calls** are the real cost metric — each circuit evaluation on the simulator counts as one call
- **Graph:** 10-node ring — `gqaoa.config.RING_TOPOLOGY_EDGES`
- **Fixed params:** `q=0.3, B=5, lamb=0, sdp=True, initial_state=dicke_state, mixture_layer=xy`

## Best known configuration

Found via Optuna TPE (`gqaoa-hpo run`), stored in `gqaoa.config.BEST_KNOWN_CONFIG`:

```python
optimizer_lr = 3.86e-4
vocab_size   = 20
n_embd       = 768
n_layer      = 12
n_head       = 12
beta_temp    = 0.7817   # final value after annealing
depth        = 5
```

---

## Package layout

```
src/gqaoa/
├── config.py       # RING_TOPOLOGY_EDGES, ProblemConfig/ModelConfig/TrainingConfig, BEST_KNOWN_CONFIG
├── paths.py        # artifacts/ (mlflow.db, optuna.db, checkpoints/) resolution
├── domain/         # pure quantum logic: QAOA Hamiltonian, data, SDP compression, injectable device
├── models/         # GPT2_QAOA model + epoch_train training loop
├── strategies/      # the 3 optimizer strategies behind a common run_job() interface
├── experiments/     # bracket (warm-restart), stability (repeated-run stats), hpo
├── tracking/        # MLflow init helper
├── reporting/        # report_stats()
└── cli/             # thin argparse entry points, one per experiment below
```

The quantum backend (`lightning.gpu` in production, `default.qubit` in tests)
is injected via `device_name` on every `run_job()` — see `domain/device.py`.

---

## Experiments

### 1. GQAOA dev-run (single training run)

```bash
gqaoa-run --limit-epochs 100 --limit-qpu-call 100 --run-name gqaoa-dev
```

### 2. Hyperparameter Optimization (HPO)

Searches for the best config using Optuna's TPE sampler. Results go to experiment `gqaoa-hpo`.

```bash
gqaoa-hpo --n-trials 40
```

200 QPU calls per trial, search space over `arch` (small/medium/full), `depth`, `vocab_size`, `beta_temp`, `optimizer_lr`.
Resumes safely — reads remaining trials from `artifacts/optuna.db`.

### 3. Stability Check (main experiment)

Runs the gqaoa strategy N times and reports the statistical distribution of `energy_min`. Results go to `gqaoa-stability`.

```bash
gqaoa-stability-check --n-runs 10 --limit-qpu-call 1000
```

Uses `BEST_KNOWN_CONFIG` (annealing: `beta_temp_max=4.0`, `beta_temp_anneal_frac=0.8`, `init_scale=0.1`).

**Statistical results by configuration** (from the original project's runs):

| Config | N | mean | std | min |
|---|---|---|---|---|
| Baseline (no annealing), 200 calls | 10 | -0.491 | 0.108 | -0.750 |
| Annealing, 200 calls | 10 | -0.491 | 0.030 | -0.549 |
| Annealing, 500 calls | 5 | -0.564 | 0.022 | -0.586 |
| Annealing, 1000 calls | 20 | -0.629 | 0.107 | **-0.795** |

### 4. Bracket Strategy (single run)

Multi-phase exploration: diverse Phase 1 → warm restart top-K into Phase 2 → top-1 into Phase 3. Results go to `gqaoa-bracket`.

```bash
gqaoa-bracket
```

Budget: 10×80 + 3×50 + 50 = **1000 QPU calls**. Checkpoints are written to
`artifacts/checkpoints/bracket/` and deleted after each run by default
(`--no-cleanup-checkpoints` to keep them).

### 5. Bracket Stability (bracket repeated N times)

Runs the full bracket strategy N times to measure its statistical stability.

```bash
gqaoa-stability-bracket --n-repetitions 3
```

### 6. Benchmark — Gradient Descent (classical baseline)

PennyLane `GradientDescentOptimizer` + SPSA. Results go to its own experiment, `gqaoa-benchmark-gd`
(the original project's `benchmark_gd.py` mistakenly shared `gqaoa-stability` — fixed here).

```bash
gqaoa-benchmark-gd --n-runs 10
```

---

## Tests

```bash
pytest                    # unit + integration (fast, no GPU needed) — default
pytest -m slow            # includes slow smoke tests (a real HPO trial, a real bracket run)
pytest -m gpu             # includes the real lightning.gpu test (requires an NVIDIA GPU)
```

Unit tests cover pure logic (Hamiltonian coefficients, config/data shapes,
`report_stats`, bracket QPU-call budgets). Integration tests run each of the 3
strategies end-to-end on the CPU `default.qubit` simulator with tiny
`depth`/`limit_epochs`/`limit_qpu_call`, so the whole pipeline is exercised
without needing a GPU.

---

## Key files

| File | Role |
|---|---|
| `src/gqaoa/strategies/gqaoa_strategy.py` | Neural-sampler strategy — `run_job()` used by stability check, bracket, HPO |
| `src/gqaoa/strategies/gradient_descent_strategy.py` | Classical baseline #1 (PennyLane GD + SPSA) |
| `src/gqaoa/strategies/scipy_strategy.py` | Classical baseline #2 (scipy.optimize) |
| `src/gqaoa/models/gpt_qaoa.py` | GPT2-based QAOA parameter sampler |
| `src/gqaoa/models/training.py` | `epoch_train()` — one training epoch |
| `src/gqaoa/domain/qaoa.py` | QAOA circuit definition, QPU call counter |
| `src/gqaoa/domain/data.py` | Expected returns + covariance matrix for the 10-asset problem |
| `src/gqaoa/domain/compression.py` | SDP compression for the problem graph |
| `src/gqaoa/experiments/bracket.py` | Unified bracket strategy (single run or repeated) |
| `src/gqaoa/experiments/stability.py` | Unified repeated-run stability analysis (any strategy) |
| `src/gqaoa/experiments/hpo.py` | Optuna HPO search |
| `artifacts/mlflow.db`, `artifacts/optuna.db` | Tracking databases (gitignored, created on first run) |
| `artifacts/checkpoints/` | Model checkpoints for bracket warm restarts (gitignored) |

## Key parameters (`TrainingConfig`, `src/gqaoa/config.py`)

| Parameter | Description |
|---|---|
| `limit_qpu_call` | Hard stop on QPU evaluations (main cost knob) |
| `beta_temp` | Final sampling temperature (lower = more greedy) |
| `beta_temp_max` | Starting temperature for annealing (`None` = no annealing) |
| `beta_temp_anneal_frac` | Fraction of QPU budget used for annealing (0.0–1.0) |
| `init_scale` | Multiplier on initial model weights (`1.0` = PyTorch default, `0.1` = near-uniform sampling) |
| `lr_T0` | Cosine warm restart period in epochs |
| `lr_T_mult` | Period multiplier after each restart |
