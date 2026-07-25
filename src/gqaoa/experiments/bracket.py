"""Bracket warm-restart strategy for the gqaoa neural-sampler.

Phase 1: N_PHASE1 runs x QPU_PHASE1 calls  -> select top TOP_K
Phase 2: TOP_K  runs x QPU_PHASE2 calls    -> select top 1
Phase 3: 1      run  x QPU_PHASE3 calls

Total QPU calls per bracket: N_PHASE1*QPU_PHASE1 + TOP_K*QPU_PHASE2 + QPU_PHASE3
                                  10*80          +    3*50           +    50      = 1000

Unifies the old bracket.py (n_repetitions=1) and stability_bracket.py
(n_repetitions=3) into one function, and fixes the inconsistency where only
stability_bracket.py cleaned up its checkpoints: cleanup_checkpoints now
defaults to True for every repetition count.
"""
import os
import shutil

from gqaoa.config import BestKnownConfig, BEST_KNOWN_CONFIG, replace_training
from gqaoa.paths import CHECKPOINTS_DIR
from gqaoa.reporting.stats import report_stats
from gqaoa.strategies import gqaoa_strategy
from gqaoa.tracking.mlflow_utils import init_mlflow

N_PHASE1 = 10
QPU_PHASE1 = 80
TOP_K = 3
QPU_PHASE2 = 50
QPU_PHASE3 = 50
TOTAL_QPU_CALLS_PER_BRACKET = N_PHASE1 * QPU_PHASE1 + TOP_K * QPU_PHASE2 + QPU_PHASE3


def _run_phase(
    phase_label, run_id, qpu_calls, ckpt_dir, base_config: BestKnownConfig,
    device_name, run_name_prefix, checkpoint_in=None, lr_T0=8, lr_T_mult=1,
):
    ckpt_out = os.path.join(ckpt_dir, f"{phase_label}_run{run_id}.pt")
    config = replace_training(base_config, limit_qpu_call=qpu_calls, lr_T0=lr_T0, lr_T_mult=lr_T_mult)
    result = gqaoa_strategy.run_job(
        config.problem, config.training, config.model,
        device_name=device_name,
        run_name=f"{run_name_prefix}{phase_label}_r{run_id}",
        checkpoint_in=checkpoint_in,
        checkpoint_out=ckpt_out,
    )
    return result["final_exp_val"], ckpt_out


def _run_one_bracket(bracket_id, ckpt_dir, base_config, device_name, run_name_prefix):
    os.makedirs(ckpt_dir, exist_ok=True)

    p1_results = []
    for i in range(N_PHASE1):
        e, c = _run_phase("p1", i + 1, QPU_PHASE1, ckpt_dir, base_config, device_name,
                           run_name_prefix, lr_T0=8, lr_T_mult=1)
        p1_results.append((e, c))
        print(f"  [bracket {bracket_id}] P1 run {i+1}/{N_PHASE1}: energy_min = {e:.6f}")
    p1_results.sort(key=lambda x: x[0])
    top_k = p1_results[:TOP_K]

    p2_results = []
    for i, (_, ckpt_in) in enumerate(top_k):
        e, c = _run_phase("p2", i + 1, QPU_PHASE2, ckpt_dir, base_config, device_name,
                           run_name_prefix, checkpoint_in=ckpt_in, lr_T0=5, lr_T_mult=2)
        p2_results.append((e, c))
        print(f"  [bracket {bracket_id}] P2 run {i+1}/{TOP_K}: energy_min = {e:.6f}")
    p2_results.sort(key=lambda x: x[0])

    final_energy, _ = _run_phase("p3", 1, QPU_PHASE3, ckpt_dir, base_config, device_name,
                                  run_name_prefix, checkpoint_in=p2_results[0][1], lr_T0=5, lr_T_mult=2)
    print(f"  [bracket {bracket_id}] FINAL: energy_min = {final_energy:.6f}")

    return final_energy, p1_results[0][0], p2_results[0][0]


def run_bracket(
    n_repetitions: int = 1,
    cleanup_checkpoints: bool = True,
    device_name: str = "lightning.gpu",
    base_config: BestKnownConfig = BEST_KNOWN_CONFIG,
    experiment_name: str = "gqaoa-bracket",
) -> list:
    """Run the bracket warm-restart strategy `n_repetitions` times.

    n_repetitions=1 reproduces the old bracket.py (single run, prints final
    energy + phase bests + improvement); n_repetitions>1 reproduces the old
    stability_bracket.py (reports percentile statistics across brackets).
    """
    init_mlflow(experiment_name)

    print(f"Bracket strategy — total QPU calls per run: {TOTAL_QPU_CALLS_PER_BRACKET}")
    print(f"Phase 1: {N_PHASE1} runs x {QPU_PHASE1} QPU calls")
    print(f"Phase 2: top {TOP_K} x {QPU_PHASE2} QPU calls")
    print(f"Phase 3: top 1 x {QPU_PHASE3} QPU calls\n")

    finals = []
    for b in range(1, n_repetitions + 1):
        print(f"\n{'='*50}")
        print(f"BRACKET RUN {b}/{n_repetitions}")
        print(f"{'='*50}")
        ckpt_dir = os.path.join(CHECKPOINTS_DIR, "bracket", f"run{b}")
        run_name_prefix = f"bracket{b}_" if n_repetitions > 1 else "bracket_"
        final_energy, p1_best, p2_best = _run_one_bracket(b, ckpt_dir, base_config, device_name, run_name_prefix)
        finals.append(final_energy)

        if n_repetitions == 1:
            print(f"\n{'='*50}")
            print(f"FINAL energy_min : {final_energy:.6f}")
            print(f"Phase 1 best     : {p1_best:.6f}")
            print(f"Phase 2 best     : {p2_best:.6f}")
            print(f"Improvement P1->P3: {p1_best - final_energy:+.6f}")

        if cleanup_checkpoints:
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    if n_repetitions > 1:
        report_stats(finals, f"Bracket stability — {n_repetitions} bracket runs, {TOTAL_QPU_CALLS_PER_BRACKET} QPU calls each")

    return finals
