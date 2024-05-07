import datetime
import json
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from experiment.model import HardLinearTransformer, Transformer
from experiment.plotter import (generate_attention_params_gif, load_data,
                                plot_attention_params, plot_mean_attn_params,
                                plot_multiple_runs)
from experiment.prompt import MDPPrompt, MDPPromptGenerator
from experiment.utils import compute_msve, cos_sim, set_seed, solve_msve_weight
from MRP.mrp import MRP


def _init_log() -> dict:
    log = {'xs': [],
           'alpha': [],
           'mstde': [],
           'mstde hard': [],
           'transformer msve': [],
           'zero order cos sim': [],
           'zero order l2 dist': [],
           'first order cos sim': [],
           'first order l2 dist': [],
           'value dist': [],
           'P': [],
           'Q': []
           }
    return log


def _init_save_dir(save_dir: str) -> None:
    if save_dir is None:
        startTime = datetime.datetime.now()
        save_dir = os.path.join('./logs',
                                "nonlinear_discounted_train",
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
          lmbd: float = 0.0,
          sample_weight: bool = False,
          mode: str = 'auto',
          lr: float = 0.001,
          weight_decay=1e-6,
          n_mdps: int = 1000,
          mini_batch_size: int = 64,
          n_batch_per_mdp: int = 5,
          log_interval: int = 10,
          save_dir: str = None,
          random_seed: int = 2):
    '''
    d: feature dimension
    s: number of states
    n: context length
    l: number of layers
    gamma: discount factor
    lmbd: eligibility trace decay
    sample_weight: sample a random true weight vector
    mode: 'auto' or 'sequential'
    lr: learning rate
    weight_decay: regularization
    log_interval: logging interval
    save_dir: directory to save logs
    mini_batch_size: mini batch size
    '''

    _init_save_dir(save_dir)

    set_seed(random_seed)

    tf = Transformer(d, n, l, lmbd, mode=mode)
    tf_hard = HardLinearTransformer(d, n, l, lmbd)

    opt = optim.Adam(tf.parameters(), lr=lr, weight_decay=weight_decay)
    opt_hard = optim.Adam(tf_hard.parameters(), lr=lr,
                          weight_decay=weight_decay)

    log = _init_log()

    pro_gen = MDPPromptGenerator(s, d, n, gamma)

    ### Training Loop ###
    for i in tqdm(range(1, n_mdps+1)):
        pro_gen.reset_feat()  # reset feature
        pro_gen.reset_mdp(sample_weight=sample_weight)  # reset MDP
        prompt = pro_gen.get_prompt()  # get prompt object
        for _ in range(n_batch_per_mdp):
            mstde = 0.0
            mstde_hard = 0.0
            Z_0 = prompt.reset()
            v_current = tf.pred_v(Z_0)
            v_current_hard = tf_hard.pred_v(Z_0)
            for _ in range(mini_batch_size):
                Z_next, reward = prompt.step()  # slide window
                v_next = tf.pred_v(Z_next)
                v_next_hard = tf_hard.pred_v(Z_next)
                tde = reward + gamma*v_next.detach() - v_current
                tde_hard = reward + gamma*v_next_hard.detach() - v_current_hard
                mstde += tde**2
                mstde_hard += tde_hard**2
                v_current = v_next
                v_current_hard = v_next_hard
            mstde /= mini_batch_size
            mstde_hard /= mini_batch_size
            opt.zero_grad()
            mstde.backward()
            opt.step()
            opt_hard.zero_grad()
            mstde_hard.backward()
            opt_hard.step()

        if i % log_interval == 0:
            prompt.reset()  # reset prompt for fair testing
            mdp: MRP = prompt.mdp
            Phi: np.ndarray = prompt.get_feature_mat().numpy()
            steady_d: np.ndarray = mdp.steady_d
            true_v: np.ndarray = mdp.v
            v_tf: np.ndarray = tf.fit_value_func(
                prompt.context(), prompt.get_feature_mat()).detach().numpy()
            w_td: np.ndarray = tf_hard.manual_weight_extraction(
                prompt.context(), d).detach().numpy()

            log['xs'].append(i)
            log['alpha'].append(tf_hard.attn.alpha.item())
            log['mstde'].append(mstde.item())
            log['mstde hard'].append(mstde_hard.item())

            tf_msve = compute_msve(v_tf, true_v, steady_d)
            log['transformer msve'].append(tf_msve)

            zo_cos_sim, zo_l2_dist = zero_order_comparison(v_tf, w_td,
                                                           Phi, steady_d)
            log['zero order cos sim'].append(zo_cos_sim)
            log['zero order l2 dist'].append(zo_l2_dist)

            fo_cos_sim, fo_l2_dist = first_order_comparison(tf, v_tf,
                                                            w_td, prompt)
            log['first order cos sim'].append(fo_cos_sim)
            log['first order l2 dist'].append(fo_l2_dist)

            if mode == 'auto':
                log['P'].append([tf.attn.P.detach().numpy().copy()])
                log['Q'].append([tf.attn.Q.detach().numpy().copy()])
            else:
                log['P'].append(
                    np.stack([layer.P.detach().numpy().copy() for layer in tf.layers]))
                log['Q'].append(
                    np.stack([layer.Q.detach().numpy().copy() for layer in tf.layers]))

    _save_log(log, save_dir)

    hyperparameters = {
        'd': d,
        's': s,
        'n': n,
        'l': l,
        'lmbd': lmbd,
        'gamma': gamma,
        'sample_weight': sample_weight,
        'n_mdps': n_mdps,
        'mini_batch_size': mini_batch_size,
        'n_batch_per_mdp': n_batch_per_mdp,
        'lr': lr,
        'weight_decay': weight_decay,
        'log_interval': log_interval,
        'random_seed': random_seed,
        'mode': mode,
        'linear': False
    }

    # Save hyperparameters as JSON
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(hyperparameters, f)


def zero_order_comparison(v_tf: np.ndarray,
                          w_td: np.ndarray,
                          Phi: np.ndarray,
                          steady_dist: np.ndarray):
    '''
    computes the cosine similarity and l2 distance
    between the batch TD weight (with the fitted learning rate) 
    and the weight of the best linear model that explaines v_tf
    '''
    w_tf = solve_msve_weight(steady_dist, Phi, v_tf)
    return cos_sim(w_tf, w_td), np.linalg.norm(w_tf - w_td)


def first_order_comparison(tf: Transformer,
                           v_tf: np.ndarray,
                           w_td: np.ndarray,
                           prompt: MDPPrompt):
    return 0.0, 0.0


if __name__ == '__main__':
    from plotter import (compute_weight_metrics, plot_error_data,
                         plot_weight_metrics, process_log)
    from utils import get_hardcoded_P, get_hardcoded_Q
    d = 4
    n = 50
    l = 3
    s = 10
    gamma = 0.9
    mode = 'auto'
    sample_weight = False
    startTime = datetime.datetime.now()
    save_dir = os.path.join(
        './logs', "nonlinear_discounted_train", startTime.strftime("%Y-%m-%d-%H-%M-%S"))
    data_dirs = []
    for seed in [38, 42, 99, 128, 256]:
        data_dir = os.path.join(save_dir, f'seed_{seed}')
        data_dirs.append(data_dir)
        train(d, s, n, l, lmbd=0.0, mode=mode,
              n_mdps=2_000, log_interval=10,
              random_seed=seed, save_dir=data_dir,
              gamma=gamma, sample_weight=sample_weight)
        log, hyperparams = load_data(data_dir)
        xs, error_log, attn_params = process_log(log)
        l_tf = l if mode == 'sequential' else 1
        plot_error_data(xs, error_log, save_dir=data_dir, params=hyperparams)
        plot_attention_params(xs, attn_params, save_dir=data_dir)
        # generate_attention_params_gif(xs, l_tf, attn_params, data_dir)
        P_true = get_hardcoded_P(d)
        Q_true = get_hardcoded_Q(d)
        P_metrics, Q_metrics = compute_weight_metrics(
            attn_params, P_true, Q_true, d)
        plot_weight_metrics(xs, l_tf, P_metrics, Q_metrics,
                            data_dir, params=hyperparams)
    plot_multiple_runs(data_dirs, save_dir=save_dir)
    plot_mean_attn_params(data_dirs, save_dir=save_dir)
