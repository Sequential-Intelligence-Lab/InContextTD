import datetime
import json
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from experiment.model import HardLinearTransformer, Transformer, MambaSSM
from experiment.prompt import MRPPromptGenerator
from MRP.mrp import MRP
from utils import (compare_sensitivity, compute_msve, implicit_weight_sim,
                   set_seed)


def _init_log() -> dict:
    log = {'xs': [],
           'alpha': [],
           'v_tf v_td msve': [],
           'implicit_weight_sim': [],
           'sensitivity cos sim': [],
           'P': [],
           'Q': []}
    return log


def _init_save_dir(save_dir: str) -> None:
    if save_dir is None:
        startTime = datetime.datetime.now()
        save_dir = os.path.join('./logs',
                                "train",
                                startTime.strftime("%Y-%m-%d-%H-%M-%S"))

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


def _save_log(log: dict, save_dir: str) -> None:
    for key, value in log.items():
        log[key] = np.array(value)
    np.savez(os.path.join(save_dir, 'data.npz'), **log)


def train(d: int,
          s: int,
          n: int,
          l: int,
          gamma: float = 0.9,
          activation: str = 'identity',
          sample_weight: bool = False,
          mode: str = 'auto',
          lr: float = 0.001,
          weight_decay=1e-6,
          n_mrps: int = 1000,
          mini_batch_size: int = 64,
          n_batch_per_mrp: int = 5,
          log_interval: int = 10,
          save_dir: str = None,
          random_seed: int = 2,
          use_mamba: bool = False):
    '''
    d: feature dimension
    s: number of states
    n: context length
    l: number of layers
    gamma: discount factor
    activation: activation function (e.g. softmax, identity, relu)
    sample_weight: sample a random true weight vector
    mode: 'auto' or 'sequential'
    lr: learning rate
    weight_decay: regularization
    log_interval: logging interval
    save_dir: directory to save logs
    mini_batch_size: mini batch size
    n_batch_per_mrp: number of batches per MRP
    n_mrps: number of MRPs
    random_seed: random seed
    '''

    _init_save_dir(save_dir)

    set_seed(random_seed)

    if use_mamba:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MambaSSM(d, device).to(device)
    else:
        device = torch.device('cpu')
        model = Transformer(d, n, l, activation=activation, mode=mode)

    # this is the hardcoded transformer that implements Batch TD with fixed weights
    batch_td = HardLinearTransformer(d, n, l)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    opt_hard = optim.Adam(batch_td.parameters(),
                          lr=lr, weight_decay=weight_decay)
    log = _init_log()

    pro_gen = MRPPromptGenerator(s, d, n, gamma)

    ### Training Loop ###
    for i in tqdm(range(1, n_mrps+1)):
        pro_gen.reset_feat()  # reset feature
        pro_gen.reset_mrp(sample_weight=sample_weight)  # reset MRP
        prompt = pro_gen.get_prompt()  # get prompt object
        for _ in range(n_batch_per_mrp):
            mstde = 0.0
            mstde_hard = 0.0
            Z_0 = prompt.reset()
            v_current = model.pred_v(Z_0.to(device)).cpu()
            v_hard_current = batch_td.pred_v(Z_0)
            for _ in range(mini_batch_size):
                Z_next, reward = prompt.step()  # slide window
                v_next = model.pred_v(Z_next.to(device)).cpu()
                v_hard_next = batch_td.pred_v(Z_next)
                tde = reward + gamma*v_next.detach() - v_current
                tde_hard = reward + gamma*v_hard_next.detach() - v_hard_current
                mstde += tde**2
                mstde_hard += tde_hard**2
                v_current = v_next
                v_hard_current = v_hard_next
            mstde /= mini_batch_size  # MSTDE for the trainable transformer
            mstde_hard /= mini_batch_size  # MSTDE for the hardcoded transformer
            opt.zero_grad()
            mstde.backward()
            opt.step()
            # the learning rate for batch td (alpha) is still trainable so we need to backpropagate
            opt_hard.zero_grad()
            mstde_hard.backward()
            opt_hard.step()

        if i % log_interval == 0:
            prompt.reset()  # reset prompt for fair testing
            mrp: MRP = prompt.mrp
            phi: np.ndarray = prompt.get_feature_mat().numpy()
            steady_d: np.ndarray = mrp.steady_d

            v_model: np.ndarray = model.fit_value_func(
                prompt.context().to(device), torch.from_numpy(phi).to(device)).detach().cpu().numpy()
            v_td: np.ndarray = batch_td.fit_value_func(
                prompt.context(), torch.from_numpy(phi)).detach().numpy()

            log['xs'].append(i)

            log['alpha'].append(batch_td.attn.alpha.item())

            # Value Difference (VD)
            log['v_tf v_td msve'].append(compute_msve(v_model, v_td, steady_d))     # TODO: fix logging labels

            # Sensitivity Similarity (SS)
            sens_cos_sim = compare_sensitivity(model, batch_td, prompt)
            log['sensitivity cos sim'].append(sens_cos_sim)

            # Implicit Weight Similarity (IWS)
            iws = implicit_weight_sim(v_model, batch_td, prompt)
            log['implicit_weight_sim'].append(iws)

            if use_mamba:
                # TODO: implement logging for mamba parameters
                pass
            else:
                if mode == 'auto':
                    log['P'].append([model.attn.P.detach().numpy().copy()])
                    log['Q'].append([model.attn.Q.detach().numpy().copy()])
                else:
                    log['P'].append(
                        np.stack([layer.P.detach().numpy().copy() for layer in model.layers]))
                    log['Q'].append(
                        np.stack([layer.Q.detach().numpy().copy() for layer in model.layers]))

    _save_log(log, save_dir)

    hyperparameters = {
        'd': d,
        's': s,
        'n': n,
        'l': l,
        'gamma': gamma,
        'activation': activation,
        'sample_weight': sample_weight,
        'mode': mode,
        'n_mrps': n_mrps,
        'mini_batch_size': mini_batch_size,
        'n_batch_per_mrp': n_batch_per_mrp,
        'lr': lr,
        'weight_decay': weight_decay,
        'log_interval': log_interval,
        'random_seed': random_seed,
        'linear': True if activation == 'identity' else False
    }

    # Save hyperparameters as JSON
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(hyperparameters, f)
