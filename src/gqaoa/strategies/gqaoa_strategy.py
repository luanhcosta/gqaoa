import os
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import pennylane as qml
import mlflow

from gqaoa.config import ModelConfig, ProblemConfig, TrainingConfig
from gqaoa.domain.device import get_device
from gqaoa.domain.qaoa import QAOA
from gqaoa.models.gpt_qaoa import GPT2_QAOA
from gqaoa.models.training import epoch_train
from gqaoa.strategies.base import StrategyResult
from gqaoa.strategies.common import beta_temp_schedule, build_problem

logging.basicConfig(level=logging.INFO)


def run_job(
    problem: ProblemConfig,
    training: TrainingConfig,
    model: ModelConfig,
    *,
    device_name: str = "lightning.gpu",
    run_name: str = "gqaoa",
    checkpoint_in: Optional[str] = None,
    checkpoint_out: Optional[str] = None,
) -> StrategyResult:
    """Neural-sampler (GPT2_QAOA) optimization strategy.

    Trains a GPT2-based autoregressive model to sample QAOA circuit angle
    parameters via a log-probability-matching loss, tracking the best/worst
    energy seen so far, until `training.limit_qpu_call` circuit evaluations
    are spent.
    """

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "optimizer_lr": training.optimizer_lr,
            "vocab_size": model.vocab_size,
            "n_embd": model.n_embd,
            "n_layer": model.n_layer,
            "n_head": model.n_head,
            "beta_temp": training.beta_temp,
            "q": problem.q,
            "B": problem.B,
            "lamb": problem.lamb,
            "initial_state": problem.initial_state,
            "mixture_layer": problem.mixture_layer,
            "sdp": problem.sdp,
            "depth": training.depth,
            "limit_epochs": training.limit_epochs,
            "limit_qpu_call": training.limit_qpu_call,
            "lr_T0": training.lr_T0,
            "lr_T_mult": training.lr_T_mult,
            "checkpoint_in": str(checkpoint_in),
            "beta_temp_max": training.beta_temp_max if training.beta_temp_max is not None else training.beta_temp,
            "beta_temp_anneal_frac": training.beta_temp_anneal_frac,
            "init_scale": training.init_scale,
        })

        expected_value, cov_matrix, lam = build_problem(problem)
        logging.info('Success: Load data')

        qaoa = QAOA(
            expected_value,
            cov_matrix,
            problem.q,
            problem.B,
            problem.lamb,
            initial_state=problem.initial_state,
            mixture_layer=problem.mixture_layer,
            edges_hc=problem.edges_hc,
            edges_hb=problem.edges_hb,
        )

        nn_qaoa = GPT2_QAOA(
            vocab_size=model.vocab_size,
            max_depth=training.depth,
            n_embd=model.n_embd,
            n_layer=model.n_layer,
            n_head=model.n_head,
        )
        logging.info(f'GPT2_QAOA running on: {nn_qaoa.device}')

        if training.init_scale != 1.0:
            with torch.no_grad():
                for param in nn_qaoa.parameters():
                    param.mul_(training.init_scale)
            logging.info(f'Applied init_scale={training.init_scale} to all weights')

        optimizer = optim.Adam(nn_qaoa.parameters(), lr=training.optimizer_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=training.lr_T0, T_mult=training.lr_T_mult, eta_min=training.optimizer_lr * 0.01
        )
        device = get_device(device_name, wires=qaoa.n_assets)
        qnode_cost_function = qml.QNode(qaoa.cost_function, device)

        energy_min = np.inf
        energy_max = -np.inf
        full_input_ids_energy_min = None
        full_input_ids_energy_max = None
        df_hist = pd.DataFrame(columns=['full_input_ids', 'energy'])

        if checkpoint_in is not None:
            ckpt = torch.load(checkpoint_in, map_location=nn_qaoa.device, weights_only=False)
            nn_qaoa.load_state_dict(ckpt['model_state'])
            logging.info(f'Loaded checkpoint from {checkpoint_in} (prev energy_min={ckpt.get("energy_min", "?")})')

        for i in range(training.limit_epochs):
            beta_temp_current = beta_temp_schedule(training, qaoa.count_qpu_call)

            (
             loss0,
             loss_log_pr_beta_temp_item,
             loss_log_pr_beta_temp_neg_item,
             loss_log_pr_beta_temp_random_item,
             loss_log_pr_energy_max_item,
             log_pr_ids_energy_min_item,
             log_pr_ids_energy_max_item,
             log_pr_ids_beta_temp_item,
             log_pr_ids_beta_temp_neg_item,
             log_pr_ids_beta_temp_random_item,
             energy_min0,
             full_input_ids_min0,
             energy_max0,
             full_input_ids_max0,
             df_hist,
             energy_beta_temp,
             energy_beta_temp_neg,
             energy_beta_temp_random,
             sum_w_beta_temp,
             sum_w_beta_temp_neg,
             sum_w_energy_min,
             sum_w_energy_max,
             w_less_full_input_ids
            ) = epoch_train(
                            nn_qaoa,
                            qnode_cost_function,
                            optimizer,
                            beta_temp_current,
                            training.depth,
                            df_hist=df_hist,
                            full_input_ids_energy_min=full_input_ids_energy_min,
                            full_input_ids_energy_max=full_input_ids_energy_max,
                            energy_min=energy_min,
                            energy_max=energy_max
                           )

            if energy_min0 < energy_min:
                energy_min = energy_min0
                full_input_ids_energy_min = full_input_ids_min0
            if energy_max0 > energy_max:
                energy_max = energy_max0
                full_input_ids_energy_max = full_input_ids_max0

            try:
                mlflow.log_metric("energy_min", energy_min, step=i)
                mlflow.log_metric("energy_beta_temp", energy_beta_temp, step=i)
                mlflow.log_metric("energy_max", energy_max, step=i)
                mlflow.log_metric("energy_beta_temp_neg", energy_beta_temp_neg, step=i)
                mlflow.log_metric("energy_beta_temp_random", energy_beta_temp_random, step=i)
                mlflow.log_metric("sum_w_energy_min", sum_w_energy_min, step=i)
                mlflow.log_metric("sum_w_beta_temp", sum_w_beta_temp, step=i)
                mlflow.log_metric("sum_w_energy_max", sum_w_energy_max, step=i)
                mlflow.log_metric("sum_w_beta_temp_neg", sum_w_beta_temp_neg, step=i)
                mlflow.log_metric("w_less_sum_energy_min", w_less_full_input_ids, step=i)
                mlflow.log_metric("loss0", loss0, step=i)
                mlflow.log_metric("loss_log_pr_beta_temp", loss_log_pr_beta_temp_item, step=i)
                mlflow.log_metric("loss_log_pr_beta_temp_neg", loss_log_pr_beta_temp_neg_item, step=i)
                mlflow.log_metric("loss_log_pr_beta_temp_random", loss_log_pr_beta_temp_random_item, step=i)
                mlflow.log_metric("loss_log_pr_energy_max", loss_log_pr_energy_max_item, step=i)
                mlflow.log_metric("log_pr_ids_energy_min", log_pr_ids_energy_min_item, step=i)
                mlflow.log_metric("log_pr_ids_energy_max", log_pr_ids_energy_max_item, step=i)
                mlflow.log_metric("log_pr_ids_beta_temp", log_pr_ids_beta_temp_item, step=i)
                mlflow.log_metric("log_pr_ids_beta_temp_neg", log_pr_ids_beta_temp_neg_item, step=i)
                mlflow.log_metric("log_pr_ids_beta_temp_random", log_pr_ids_beta_temp_random_item, step=i)
                mlflow.log_metric("count_qpu_call", qaoa.count_qpu_call, step=i)
                mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=i)
                mlflow.log_metric("beta_temp_current", beta_temp_current, step=i)
            except Exception as exc:
                # A transient MLflow/SQLite hiccup (e.g. lock contention with a
                # concurrently running `mlflow ui`) shouldn't discard an entire
                # training run's worth of compute over a missed metrics log.
                logging.warning(f"epoch {i}: failed to log metrics to MLflow ({exc}); continuing training")

            scheduler.step(i)

            if qaoa.count_qpu_call >= training.limit_qpu_call:
                break

        device_probs = get_device(device_name, wires=qaoa.n_assets, shots=training.shots_probs_result)
        gamma_array = []
        beta_array = []
        for i in range(training.depth):
            gamma = nn_qaoa.vocab_gamma[full_input_ids_energy_min[2*i]]
            beta = nn_qaoa.vocab_beta[full_input_ids_energy_min[2*i+1]]
            gamma_array.append(gamma)
            beta_array.append(beta)
        qnode_probability = qml.QNode(qaoa.probability_circuit, device_probs)
        probs = qnode_probability(np.array([gamma_array, beta_array]))

        result: StrategyResult = {
                "params": np.array([gamma_array, beta_array]),
                "probs": probs,
                "final_exp_val": energy_min,
                "cov_matrix": cov_matrix,
                "expected_value": expected_value,
                "lambda_sdp": lam,
               }
        mlflow.log_metric("final_exp_val", energy_min)

        if checkpoint_out is not None:
            ckpt_dir = os.path.dirname(checkpoint_out)
            if ckpt_dir:
                os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({'model_state': nn_qaoa.state_dict(), 'energy_min': energy_min}, checkpoint_out)
            logging.info(f'Saved checkpoint to {checkpoint_out}')

        return result
