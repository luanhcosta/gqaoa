import pandas as pd
import numpy as np
import torch


def f_loss(energies, sum_w):
    response = torch.mean((sum_w-energies).pow(2))
    return response


def f_loss2(log_pr1, energy1, log_pr2, energy2, energy_scale=1):
    response = ((log_pr1-log_pr2)-energy_scale*(energy1-energy2)).pow(2)
    return response


def f_log_pr_ids(logits, full_input_ids):
    response = 0
    for i in range(len(logits)):
        softmax = torch.softmax(-logits[i].view(1, -1), dim=1)
        response -= torch.log(softmax[0][full_input_ids[i]])
    return response


def epoch_train(
                model,
                qnode,
                optimizer,
                beta_temp,
                depth,
                df_hist=pd.DataFrame(),
                full_input_ids_energy_min=None,
                full_input_ids_energy_max=None,
                energy_min=None,
                energy_max=None
               ):

    device = model.device

    def qnode2(qnode, gamma, beta):
        response = qnode(np.array([gamma, beta]))
        if isinstance(response, np.ndarray):
            response = response.item()
        return response

    ## beta_temp
    sum_w, _, logits, full_input_ids_, gamma, beta = model.forward_qc(beta_temp, depth)
    sum_w_beta_temp = sum_w.item()
    log_pr_ids_beta_temp = f_log_pr_ids(logits, full_input_ids_)
    log_pr_ids_beta_temp_item = log_pr_ids_beta_temp.item()
    index_filter = df_hist['full_input_ids'].apply(lambda x: x==full_input_ids_)
    if index_filter.sum()>0:
        energy_beta_temp = df_hist[index_filter]['energy'].values[0]
        energies = torch.tensor([energy_beta_temp], device=device)
    else:
        energy_beta_temp = qnode2(qnode, gamma, beta)
        energies = torch.tensor([energy_beta_temp], device=device)
        df_ = pd.DataFrame([{'full_input_ids': full_input_ids_, 'energy': energy_beta_temp}])
        df_hist = pd.concat([df_hist, df_])
    energy_min_temp0 = energy_beta_temp
    energy_max_temp0 = energy_beta_temp
    full_input_ids_min_temp0 = full_input_ids_
    full_input_ids_max_temp0 = full_input_ids_

    ## -beta_temp
    sum_w_, _, logits, full_input_ids_, gamma, beta = model.forward_qc(-beta_temp, depth)
    sum_w_beta_temp_neg = sum_w_.item()
    log_pr_ids_beta_temp_neg = f_log_pr_ids(logits, full_input_ids_)
    log_pr_ids_beta_temp_neg_item = log_pr_ids_beta_temp_neg.item()
    index_filter = df_hist['full_input_ids'].apply(lambda x: x==full_input_ids_)
    if index_filter.sum()>0:
        energy_beta_temp_neg = df_hist[index_filter]['energy'].values[0]
        energy_ = torch.tensor([energy_beta_temp_neg], device=device)
    else:
        energy_beta_temp_neg = qnode2(qnode, gamma, beta)
        energy_ = torch.tensor([energy_beta_temp_neg], device=device)
        df_ = pd.DataFrame([{'full_input_ids': full_input_ids_, 'energy': energy_.item()}])
        df_hist = pd.concat([df_hist, df_]).reset_index(drop=True)
    if energy_.item() < energy_min_temp0:
        energy_min_temp0 = energy_beta_temp_neg
        full_input_ids_min_temp0 = full_input_ids_
    else:
        energy_max_temp0 = energy_beta_temp_neg
        full_input_ids_max_temp0 = full_input_ids_
    sum_w = torch.cat((sum_w, sum_w_), dim=0)
    energies = torch.cat((energies, energy_), dim=0)

    ## random
    sum_w_, _, logits, full_input_ids_, gamma, beta = model.forward_qc(1e-3, depth)
    sum_w_beta_temp_random = sum_w_.item()
    log_pr_ids_beta_temp_random = f_log_pr_ids(logits, full_input_ids_)
    log_pr_ids_beta_temp_random_item = log_pr_ids_beta_temp_random.item()
    index_filter = df_hist['full_input_ids'].apply(lambda x: x==full_input_ids_)
    if index_filter.sum()>0:
        energy_beta_temp_random = df_hist[index_filter]['energy'].values[0]
        energy_ = torch.tensor([energy_beta_temp_random], device=device)
    else:
        energy_beta_temp_random = qnode2(qnode, gamma, beta)
        energy_ = torch.tensor([energy_beta_temp_random], device=device)
        df_ = pd.DataFrame([{'full_input_ids': full_input_ids_, 'energy': energy_.item()}])
        df_hist = pd.concat([df_hist, df_]).reset_index(drop=True)
    if energy_.item() < energy_min_temp0:
        energy_min_temp0 = energy_beta_temp_random
        full_input_ids_min_temp0 = full_input_ids_
    elif energy_.item() > energy_max_temp0:
        energy_max_temp0 = energy_beta_temp_random
        full_input_ids_max_temp0 = full_input_ids_
    sum_w = torch.cat((sum_w, sum_w_), dim=0)
    energies = torch.cat((energies, energy_), dim=0)

    ## min_energy
    sum_w_, w_less_full_input_ids, logits, _, gamma, beta = model.forward_qc(1, depth, full_input_ids_energy_min)
    sum_w_energy_min = sum_w_.item()
    if full_input_ids_energy_min is not None:
        log_pr_ids_energy_min = f_log_pr_ids(logits, full_input_ids_energy_min)
        log_pr_ids_energy_min_item = log_pr_ids_energy_min.item()
    else:
        log_pr_ids_energy_min = log_pr_ids_energy_min_item = 0

    ## max_energy
    sum_w_, _, logits, _, gamma, beta = model.forward_qc(1, depth, full_input_ids_energy_max)
    sum_w_energy_max = sum_w_.item()
    if full_input_ids_energy_max is not None:
        log_pr_ids_energy_max = f_log_pr_ids(logits, full_input_ids_energy_max)
        log_pr_ids_energy_max_item = log_pr_ids_energy_max.item()
    else:
        log_pr_ids_energy_max = log_pr_ids_energy_max_item = 0

    loss0 = 0*f_loss(energies, sum_w)
    loss_log_pr_beta_temp = 0
    loss_log_pr_beta_temp_item = 0
    if full_input_ids_energy_min is not None:
        loss_log_pr_beta_temp = f_loss2(log_pr_ids_energy_min, energy_min, log_pr_ids_beta_temp, energy_beta_temp, energy_scale=1e4)
        loss_log_pr_beta_temp_item = loss_log_pr_beta_temp.item()
    loss_log_pr_beta_temp_neg = 0
    loss_log_pr_beta_temp_neg_item = 0
    if full_input_ids_energy_min is not None:
        loss_log_pr_beta_temp_neg = f_loss2(log_pr_ids_energy_min, energy_min, log_pr_ids_beta_temp_neg, energy_beta_temp_neg, energy_scale=1e4)
        loss_log_pr_beta_temp_neg_item = loss_log_pr_beta_temp_neg.item()
    loss_log_pr_beta_temp_random = 0
    loss_log_pr_beta_temp_random_item = 0
    if full_input_ids_energy_min is not None:
        loss_log_pr_beta_temp_random = f_loss2(log_pr_ids_energy_min, energy_min, log_pr_ids_beta_temp_random, energy_beta_temp_random, energy_scale=1e4)
        loss_log_pr_beta_temp_random_item = loss_log_pr_beta_temp_random.item()
    loss_log_pr_energy_max = 0
    loss_log_pr_energy_max_item = 0
    if full_input_ids_energy_min is not None:
        loss_log_pr_energy_max = f_loss2(log_pr_ids_energy_min, energy_min, log_pr_ids_energy_max, energy_max, energy_scale=1e4)
        loss_log_pr_energy_max_item = loss_log_pr_energy_max.item()

    optimizer.zero_grad()
    loss = loss0+loss_log_pr_beta_temp+loss_log_pr_beta_temp_neg+loss_log_pr_beta_temp_random+loss_log_pr_energy_max
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return (
            loss0.item(),
            loss_log_pr_beta_temp_item,
            loss_log_pr_beta_temp_neg_item,
            loss_log_pr_beta_temp_random_item,
            loss_log_pr_energy_max_item,
            log_pr_ids_energy_min_item,
            log_pr_ids_energy_max_item,
            log_pr_ids_beta_temp_item,
            log_pr_ids_beta_temp_neg_item,
            log_pr_ids_beta_temp_random_item,
            energy_min_temp0,
            full_input_ids_min_temp0,
            energy_max_temp0,
            full_input_ids_max_temp0,
            df_hist,
            energy_beta_temp,
            energy_beta_temp_neg,
            energy_beta_temp_random,
            sum_w_beta_temp,
            sum_w_beta_temp_neg,
            sum_w_energy_min,
            sum_w_energy_max,
            w_less_full_input_ids
           )
