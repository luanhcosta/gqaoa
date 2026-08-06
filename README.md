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

> **Don't delete `artifacts/mlflow.db`/`artifacts/optuna.db` while `mlflow ui` is running.**
> The server keeps the file open; deleting it out from under a running server leaves the
> server holding a handle to an orphaned file, and a training run started afterward can
> collide with it — this is exactly what caused an `attempt to write a readonly database`
> crash partway through a real run. If you need to reset the tracking DB, stop the server
> first (`pkill -f "mlflow ui"`), delete the file, then restart the server.

| Experiment | MLflow experiment name |
|---|---|
| Stability check (main experiment) | `gqaoa-stability` |
| Bracket strategy | `gqaoa-bracket` |
| HPO | `gqaoa-hpo` |
| Benchmark (classical GD baseline) | `gqaoa-benchmark-gd` |
| Benchmark (classical scipy baseline) | `gqaoa-benchmark-scipy` |

---

## Running without a GPU (CPU-only)

The quantum backend is injected via `--device-name` on every CLI and defaults to
`"lightning.gpu"` (GPU simulator, requires an NVIDIA GPU + CUDA + the `gpu` extra).
Passing `--device-name default.qubit` switches to a pure-Python/NumPy statevector
simulator that ships with PennyLane itself — no GPU or extra required:

```bash
pip install -e ".[dev]"          # no need for the gpu extra at all
```

**What is genuinely fast on CPU** (verified: seconds, not minutes):

- **The full test suite** (`pytest`) — `tests/conftest.py` already fixes
  `device_name="default.qubit"` and uses tiny models/budgets for every
  unit/integration test, so `pytest` runs green on a machine with no GPU at all
  (see "Tests" below).
- **`gqaoa-run`**, if you also shrink the model via its CLI flags (`--vocab-size`,
  `--n-layer`, `--depth` are all exposed) — e.g. `gqaoa-run --device-name
  default.qubit --vocab-size 4 --n-layer 1 --depth 2 --limit-epochs 5
  --limit-qpu-call 5` finishes in ~15s. Using its *defaults* (`--n-layer 12
  --depth 20`) on CPU is not fast — expect minutes, not seconds.
- **`gqaoa-benchmark-gd`, `gqaoa-benchmark-scipy`**, at any budget — neither classical
  baseline has a neural net at all, so `--limit-qpu-call` directly controls the cost and
  CPU is fine even at the full 1000-call budget (e.g. `gqaoa-benchmark-gd --device-name
  default.qubit --n-runs 3` or `gqaoa-benchmark-scipy --device-name default.qubit
  --n-runs 3` both run in seconds per run).
- **`gqaoa-stability-check`, `gqaoa-bracket`, `gqaoa-stability-bracket`, `gqaoa-hpo`**,
  *if you also shrink the model* — all four use `BEST_KNOWN_CONFIG`'s "full" GPT2
  architecture (`n_embd=768, n_layer=12, n_head=12`) by default, and on CPU the
  transformer forward/backward pass dominates cost, not the QAOA circuit, so
  lowering `--limit-qpu-call` alone does **not** help (confirmed:
  `gqaoa-stability-check --limit-qpu-call 5` alone didn't finish within a
  minute). Each of the four now also accepts `--vocab-size`/`--n-embd`/
  `--n-layer`/`--n-head` overrides (default: unchanged, so full-budget GPU runs
  are unaffected) — e.g. `gqaoa-stability-check --device-name default.qubit
  --n-layer 1 --vocab-size 4 --n-runs 1 --limit-qpu-call 5` completes in ~8s.
  `gqaoa-hpo` additionally exposes `--limit-epochs`/`--limit-qpu-call` (default
  900/200) so a smoke run can use e.g. `--n-trials 2 --limit-epochs 5
  --limit-qpu-call 10`.

**What CPU is still not good for:**

- `default.qubit` has no CUDA acceleration, so reproducing the documented
  results tables below (`limit_qpu_call=1000`, `depth=5`, full architecture,
  many repeated runs) is only realistic with a GPU. The overrides above are for
  validating that a pipeline runs end-to-end without errors, not for
  reproducing real numbers.

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
optimizer_lr           = 3.86e-4
vocab_size             = 20
n_embd                 = 768
n_layer                = 12
n_head                 = 12
beta_temp              = 0.7817   # final value after annealing
beta_temp_max          = 4.0      # starting value (None = no annealing, see NO_ANNEAL_CONFIG)
beta_temp_anneal_frac  = 0.8      # fraction of QPU budget spent annealing
init_scale             = 0.1
depth                  = 5
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

## Optimizer strategies

All three strategies in `src/gqaoa/strategies/` solve the same problem —
minimize `energy_min`, the energy of the QAOA cost Hamiltonian — but choose the
circuit's angle parameters (γ, β) in different ways. They share a common
`run_job(problem, training, ..., device_name, run_name, checkpoint_in,
checkpoint_out)` signature (`strategies/base.py::OptimizerStrategy`), which is
what lets `experiments/bracket.py` and `experiments/stability.py` dispatch to
any of them through a simple registry instead of duplicating logic per strategy.

- **`gqaoa_strategy.py` — the neural-sampler strategy (the project's main idea).**
  Instead of optimizing the QAOA angles directly, trains a GPT2 model
  (`models/gpt_qaoa.py`) to *generate* token sequences that decode into γ/β
  angles. Training (`models/training.py::epoch_train`) uses a custom
  "log-probability matching" loss: the model learns to assign high probability
  to low-energy sequences and low probability to high-energy ones, compared
  against the best/worst energies seen so far. Each epoch samples at 5
  different "temperatures" (`beta_temp`, `-beta_temp`, near-random, replay of
  the best minimum, replay of the best maximum) to explore the search space.
  It's the only strategy that supports checkpointing (saving/loading model
  weights), which is what the bracket strategy relies on for warm restarts.

- **`gradient_descent_strategy.py` — classical baseline #1.** PennyLane's
  `GradientDescentOptimizer` with `diff_method="spsa"` (a stochastic
  gradient approximation, useful when differentiating the circuit exactly is
  expensive) acting directly on the γ/β parameters — no neural net at all.
  Each optimizer step is exactly one QPU call regardless of depth (verified:
  10 steps ⇒ 10 circuit evaluations at both depth=1 and depth=5), so
  `--limit-qpu-call` bounds the real cost precisely.

- **`scipy_strategy.py` — classical baseline #2.** `scipy.optimize.minimize`,
  parametrized over `minimize_method` (default `COBYLA`, but any scipy method
  works — see item 7), treating the QAOA cost function as a black box to
  minimize. A counter inside the `objective()` closure aborts the solver once
  `limit_qpu_call` is spent, since scipy has no native way to cap the number
  of function evaluations. Derivative-free methods (`COBYLA`, `Nelder-Mead`,
  `Powell`) keep their own `maxiter`/`maxfev` close enough to the true
  eval count that this rarely triggers; gradient-based methods (`BFGS`, `CG`,
  `L-BFGS-B`, `SLSQP`, ...) estimate gradients by finite differences, spend
  several evaluations per solver iteration, and reliably hit this abort
  before `maxiter` (an iteration count, not an eval count) would — in that
  case the best point/energy seen so far is returned instead of crashing,
  and a `qpu_budget_exceeded` MLflow param is logged so it's visible which
  runs took that path.

---

## Experiments

Each experiment below is a thin CLI (`src/gqaoa/cli/`) built on top of
`src/gqaoa/experiments/`, combining or repeating the strategies above to
answer a different question.

### 1. GQAOA dev-run (single training run)

A single training run with the gqaoa strategy, meant for quick debugging/development iteration rather than a real result.

```bash
gqaoa-run --limit-epochs 100 --limit-qpu-call 100 --run-name gqaoa-dev
```

### 2. Hyperparameter Optimization (HPO)

Searches for the best config for the gqaoa strategy using Optuna's TPE sampler, varying architecture
(small/medium/full), `depth`, `vocab_size`, `beta_temp`, `optimizer_lr` — this is how `BEST_KNOWN_CONFIG`
was originally found. Results go to experiment `gqaoa-hpo`.

```bash
gqaoa-hpo --n-trials 40
```

200 QPU calls per trial. Resumes safely — reads remaining trials from `artifacts/optuna.db`.
`--limit-epochs`/`--limit-qpu-call` override the per-trial budget (e.g. for a CPU smoke run — see
"Running without a GPU" above).

### 3. Stability Check (main experiment)

Runs the gqaoa strategy N times with `BEST_KNOWN_CONFIG` and reports the statistical distribution of
`energy_min` (min/percentiles/mean/std) — measures how consistent the best known config actually is
across independent runs. Results go to `gqaoa-stability`.

```bash
gqaoa-stability-check --n-runs 10 --limit-qpu-call 1000
```

Uses `BEST_KNOWN_CONFIG` by default (annealing: `beta_temp_max=4.0`, `beta_temp_anneal_frac=0.8`,
`init_scale=0.1`). `--vocab-size`/`--n-embd`/`--n-layer`/`--n-head` override the model architecture
(e.g. for a CPU smoke run — see "Running without a GPU" above); omit them to keep the full
architecture unchanged.

Pass `--no-annealing` to run the same config with `beta_temp` held constant instead (via
`gqaoa.config.NO_ANNEAL_CONFIG` — identical to `BEST_KNOWN_CONFIG` except `beta_temp_max=None`),
reproducing the "Baseline (no annealing)" row below and letting you measure annealing's actual
contribution. A single isolated run is just `--n-runs 1`:

```bash
gqaoa-stability-check --no-annealing --n-runs 10 --limit-qpu-call 1000   # ablation, N runs
gqaoa-stability-check --no-annealing --n-runs 1  --limit-qpu-call 1000   # ablation, 1 isolated run
```

Both modes log to the same `gqaoa-stability` experiment — the `stability_anneal_*`/
`stability_no_anneal_*` run-name prefix and the `stability_summary` run's `beta_temp_max` param
(`None` vs `4.0`) tell them apart for side-by-side comparison in the MLflow UI.

**Statistical results by configuration** (from the original project's runs):

| Config | N | mean | std | min |
|---|---|---|---|---|
| Baseline (no annealing), 200 calls | 10 | -0.491 | 0.108 | -0.750 |
| Annealing, 200 calls | 10 | -0.491 | 0.030 | -0.549 |
| Annealing, 500 calls | 5 | -0.564 | 0.022 | -0.586 |
| Annealing, 1000 calls | 20 | -0.629 | 0.107 | **-0.795** |

### 4. Bracket Strategy (single run)

Multi-phase warm-restart exploration on top of the gqaoa strategy: 10 diverse runs of 80 calls each
(Phase 1) → the 3 best continue for 50 more calls each, resuming from checkpoint (Phase 2) → the best
of those continues for 50 more calls (Phase 3). Same total budget as the stability check (1000 QPU
calls), but spends it exploring broadly before committing to refining the most promising candidates.
Results go to `gqaoa-bracket`.

```bash
gqaoa-bracket
```

Budget: 10×80 + 3×50 + 50 = **1000 QPU calls**. Checkpoints are written to
`artifacts/checkpoints/bracket/` and deleted after each run by default
(`--no-cleanup-checkpoints` to keep them). Same `--vocab-size`/`--n-embd`/`--n-layer`/`--n-head`
overrides as the stability check are available here too.

### 5. Bracket Stability (bracket repeated N times)

Runs the full bracket strategy (item 4) N times back-to-back, to check whether the bracket approach
itself is consistent or has high variance across independent executions — the bracket-strategy
equivalent of item 3's stability check.

```bash
gqaoa-stability-bracket --n-repetitions 3
```

### 6. Benchmark — Gradient Descent (classical baseline)

Runs the classical `gradient_descent_strategy` (PennyLane `GradientDescentOptimizer` + SPSA) N times
with the same QPU-call budget as the stability check, as the "does the neural sampler actually help?"
comparison point. Results go to its own experiment, `gqaoa-benchmark-gd`
(the original project's `benchmark_gd.py` mistakenly shared `gqaoa-stability` — fixed here).

```bash
gqaoa-benchmark-gd --n-runs 10 --limit-qpu-call 1000
```

`--limit-qpu-call` is exact here — each SPSA optimizer step is one QPU call (see the
`gradient_descent_strategy.py` note above), so it directly bounds the real cost with no risk of
overrun.

### 7. Benchmark — scipy (classical baseline)

Runs the classical `scipy_strategy` (`scipy.optimize.minimize`) N times, the same "does the
neural sampler actually help?" comparison as item 6, for whichever `scipy.optimize.minimize`
algorithm you pass via `--minimize-method` (e.g. `COBYLA`, `Nelder-Mead`, `Powell`, `CG`, `BFGS`,
`L-BFGS-B`, `TNC`, `SLSQP`, `trust-constr` — see
[scipy's docs](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html) for
the full list and which accept bounds/constraints). Results go to their own experiment,
`gqaoa-benchmark-scipy`, with the method name in the run-name prefix so different algorithms are
easy to tell apart.

```bash
gqaoa-benchmark-scipy --n-runs 10 --minimize-method COBYLA --limit-qpu-call 1000
gqaoa-benchmark-scipy --n-runs 10 --minimize-method Nelder-Mead --limit-qpu-call 1000
gqaoa-benchmark-scipy --n-runs 10 --minimize-method BFGS --limit-qpu-call 1000
```

`--limit-qpu-call` behaves differently depending on `--minimize-method` — see the
`scipy_strategy.py` note above: exact for derivative-free methods (`COBYLA`, `Nelder-Mead`,
`Powell`), a soft cap for gradient-based ones (falls back to the best point seen so far instead of
crashing, flagged via the `qpu_budget_exceeded` MLflow param).

`experiments/stability.py::run_stability()` takes an optional `strategy_kwargs` dict, forwarded
verbatim to the strategy's `run_job()` on every run — this is what threads `--minimize-method`
through, since it's specific to the scipy strategy and isn't a `TrainingConfig` field. Any future
strategy-specific CLI flag (for a new strategy, or a new scipy option) can reuse the same
mechanism instead of growing `TrainingConfig`.

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
