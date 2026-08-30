"""Hyperparameter optimization for GQAOA using Optuna + MLflow.

Search phase  : 200 QPU calls / trial (TPE sampler)
Confirmation  : run gqaoa_strategy.run_job manually with the best params at full budget (900+)

Resumable: re-running main() picks up remaining trials from optuna.db.
"""
import dataclasses
import logging

import mlflow
import optuna

from gqaoa.config import ModelConfig, ProblemConfig, RING_TOPOLOGY_EDGES, TrainingConfig
from gqaoa.paths import ARTIFACTS_DIR, MLFLOW_TRACKING_URI, OPTUNA_STORAGE
from gqaoa.strategies import gqaoa_strategy

logging.basicConfig(level=logging.WARNING)
optuna.logging.set_verbosity(optuna.logging.WARNING)

FIXED_PROBLEM = ProblemConfig(
    q=0.3, B=5, lamb=0,
    initial_state="dicke_state", mixture_layer="xy",
    edges_hc=RING_TOPOLOGY_EDGES, edges_hb=RING_TOPOLOGY_EDGES,
    sdp=True,
)

ARCH_PRESETS = {
    "small": {"n_embd": 128, "n_layer": 4, "n_head": 4},
    "medium": {"n_embd": 256, "n_layer": 6, "n_head": 8},
    "full": {"n_embd": 768, "n_layer": 12, "n_head": 12},
}


def objective(
    trial: optuna.Trial,
    device_name: str = "lightning.gpu",
    limit_epochs: int = 900,
    limit_qpu_call: int = 200,
    sdp: bool = True,
) -> float:
    arch = trial.suggest_categorical("arch", list(ARCH_PRESETS.keys()))
    depth = trial.suggest_categorical("depth", [5, 10, 15, 20])
    vocab_size = trial.suggest_categorical("vocab_size", [5, 10, 20])
    beta_temp = trial.suggest_float("beta_temp", 0.5, 3.0, log=True)
    optimizer_lr = trial.suggest_float("optimizer_lr", 1e-5, 5e-3, log=True)

    model = ModelConfig(vocab_size=vocab_size, **ARCH_PRESETS[arch])
    training = TrainingConfig(
        depth=depth, limit_epochs=limit_epochs, limit_qpu_call=limit_qpu_call,
        optimizer_lr=optimizer_lr, beta_temp=beta_temp,
    )
    problem = dataclasses.replace(FIXED_PROBLEM, sdp=sdp)

    result = gqaoa_strategy.run_job(
        problem, training, model,
        device_name=device_name, run_name=f"hpo_trial_{trial.number}",
    )
    return result["final_exp_val"]


def main(
    n_trials: int = 40,
    device_name: str = "lightning.gpu",
    limit_epochs: int = 900,
    limit_qpu_call: int = 200,
    sdp: bool = True,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("gqaoa-hpo")

    study = optuna.create_study(
        study_name="gqaoa-hpo",
        direction="minimize",
        storage=OPTUNA_STORAGE,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    remaining = max(0, n_trials - completed)
    print(f"Trials completed: {completed} / {n_trials}  —  running {remaining} more")

    study.optimize(
        lambda trial: objective(
            trial, device_name=device_name,
            limit_epochs=limit_epochs, limit_qpu_call=limit_qpu_call, sdp=sdp,
        ),
        n_trials=remaining, show_progress_bar=True, catch=(Exception,),
    )

    best = study.best_trial
    arch_cfg = ARCH_PRESETS[best.params["arch"]]

    print(f"\n{'='*40}")
    print(f"Best trial  : #{best.number}")
    print(f"energy_min  : {best.value:.6f}")
    print(f"\nBest hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k:15s}: {v}")

    print(f"\nTo run final confirmation (full budget):")
    print(f"  gqaoa_strategy.run_job(")
    print(f"    problem=..., model=ModelConfig(vocab_size={best.params['vocab_size']}, "
          f"n_embd={arch_cfg['n_embd']}, n_layer={arch_cfg['n_layer']}, n_head={arch_cfg['n_head']}),")
    print(f"    training=TrainingConfig(depth={best.params['depth']}, "
          f"optimizer_lr={best.params['optimizer_lr']:.2e}, beta_temp={best.params['beta_temp']:.4f}, "
          f"limit_qpu_call=900),")
    print(f"    run_name='hpo_best_final')")
