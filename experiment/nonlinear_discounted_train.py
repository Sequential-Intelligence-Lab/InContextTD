import datetime
import json
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from experiment.model import Transformer
from experiment.plotter import (generate_attention_params_gif, load_data,
                                plot_attention_params, plot_mean_attn_params,
                                plot_multiple_runs)
from experiment.prompt import MDPPrompt, MDPPromptGenerator
from experiment.utils import (compute_msve, solve_msve_weight, set_seed)


def _init_log() -> dict:
    log = {'xs': [],
           'mstde': [],
           'true msve': [],
           'transformer msve': [],
           'implicit w_tf and w_td cos sim': [],
           'fo cos dist': [],
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

    opt = optim.Adam(tf.parameters(), lr=lr, weight_decay=weight_decay)

    log = _init_log()

    pro_gen = MDPPromptGenerator(s, d, n, gamma)

    ### Training Loop ###
    for i in tqdm(range(1, n_mdps+1)):
        pro_gen.reset_feat()  # reset feature
        pro_gen.reset_mdp(sample_weight=sample_weight)  # reset MDP
        prompt = pro_gen.get_prompt()  # get prompt object
        for _ in range(n_batch_per_mdp):
            mstde = 0.0
            Z_0 = prompt.reset()
            v_current = tf.pred_v(Z_0)
            for _ in range(mini_batch_size):
                Z_next, reward = prompt.step()  # slide window
                v_next = tf.pred_v(Z_next)
                tde = reward + gamma*v_next.detach() - v_current
                mstde += tde**2
                v_current = v_next
            mstde /= mini_batch_size
            opt.zero_grad()
            mstde.backward()
            opt.step()

        if i % log_interval == 0:
            prompt.reset()  # reset prompt for fair testing
            mdp = prompt.mdp
            phi = prompt.feature_fun.phi
            steady_d = mdp.steady_d
            true_v = mdp.v
            v_tf = tf.fit_value_func(prompt.context(), torch.from_numpy(phi)).detach().numpy()

            log['xs'].append(i)
            log['mstde'].append(mstde.item())

            tf_msve = compute_msve(v_tf, true_v, steady_d)
            log['transformer msve'].append(tf_msve)

            cos_sim = linear_model_cos_similarity(tf, prompt)
            log['implicit w_tf and w_td cos sim'].append(cos_sim) 

            fo_cos_dist, value_dist = first_order_dist(tf, prompt)
            log['fo cos dist'].append(fo_cos_dist)
            log['value dist'].append(value_dist)


            if mode=='auto':
                log['P'].append([tf.attn.P.detach().numpy().copy()])
                log['Q'].append([tf.attn.Q.detach().numpy().copy()])
            else:
                log['P'].append(np.stack([layer.P.detach().numpy().copy() for layer in tf.layers]))
                log['Q'].append(np.stack([layer.Q.detach().numpy().copy() for layer in tf.layers]))

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
            
            
# computes the cosine similarity between the tf forward pass and TD
def linear_model_cos_similarity( tf:Transformer,  prompt: MDPPrompt):
    '''
    computes the cosine similarity between the transformer forward pass implicit weight and the TD update weight 
    tf: an instance of LinearTransformer
    prompt: an instance of MDPPrompt
    '''
    # make another transformer with 1 layer and copy over the attention weights
    new_tf = Transformer(tf.d, tf.n, 1, tf.lmbd, mode='auto')
    new_tf.attn.load_state_dict(tf.attn.state_dict())
    new_tf.eval()

    phi = prompt.get_feature_mat()
    v_tf = new_tf.fit_value_func(prompt.context(), phi).detach().numpy().reshape((-1, 1))
    w_tf = solve_msve_weight(prompt.mdp.steady_d, phi.numpy(), v_tf)
    

    w_td = torch.zeros((tf.d, 1))
    w_td = prompt.td_update(w_td)[0].numpy()
    cos_sim: np.ndarray = w_tf.T @ w_td / (np.linalg.norm(w_tf) * np.linalg.norm(w_td))
    return cos_sim.item()


def first_order_dist(tf: Transformer, prompt: MDPPrompt):
    new_tf = Transformer(tf.d, tf.n, 1, tf.lmbd, mode='auto')
    new_tf.attn.load_state_dict(tf.attn.state_dict())
    new_tf.eval()

    phi = prompt.get_feature_mat()
    ns, d = phi.shape
    v_tf = new_tf.fit_value_func(prompt.context(), phi).detach().reshape((-1, 1))

    w_td = torch.zeros((tf.d, 1))
    w_td = prompt.td_update(w_td)[0]

    cos_dist = 0.0
    for i in range(ns - 1):
        for j in range(i + 1, ns):
            phi_i = phi[i].reshape((-1, 1))
            phi_j = phi[j].reshape((-1, 1))
            phi_mid = (phi_i + phi_j) / 2
            prompt.set_query(phi_mid)
            prompt.enable_query_grad()
            v_mid = new_tf.pred_v(prompt.z())
            v_mid.backward()
            grad_mid = prompt.query_grad()
            cos_dist += (1.0 - grad_mid.T @ w_td / (torch.norm(grad_mid) * torch.norm(w_td))).item()
            prompt.zero_query_grad()
    
    return cos_dist / (ns * (ns - 1) / 2), torch.norm(phi @ w_td - v_tf).item()

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
    startTime = datetime.datetime.now()
    save_dir = os.path.join('./logs', "nonlinear_discounted_train", startTime.strftime("%Y-%m-%d-%H-%M-%S"))
    data_dirs = []
    for seed in [38, 42, 99, 128, 256]:
        data_dir = os.path.join(save_dir, f'seed_{seed}')
        data_dirs.append(data_dir)
        train(d, s, n, l, lmbd=0.0, mode=mode,
              n_mdps=500, log_interval=20, 
              random_seed=seed, save_dir=data_dir,
              gamma=gamma)
        log, hyperparams = load_data(data_dir)
        xs, error_log, attn_params = process_log(log)
        l_tf = l if mode == 'sequential' else 1
        plot_error_data(xs, error_log, save_dir=data_dir, params=hyperparams)
        plot_attention_params(xs, attn_params, save_dir=data_dir)
        # generate_attention_params_gif(xs, l_tf, attn_params, data_dir)
        P_true = get_hardcoded_P(d)
        Q_true = get_hardcoded_Q(d)
        P_metrics, Q_metrics = compute_weight_metrics(attn_params, P_true, Q_true, d)
        plot_weight_metrics(xs, l_tf, P_metrics, Q_metrics, data_dir, params=hyperparams)
    plot_multiple_runs(data_dirs, save_dir=save_dir)
    plot_mean_attn_params(data_dirs, save_dir=save_dir)