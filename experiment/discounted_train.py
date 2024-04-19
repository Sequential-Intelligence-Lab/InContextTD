import datetime
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from experiment.loss import weight_error_norm
from experiment.model import LinearTransformer
from experiment.prompt import MDPPromptGenerator
from experiment.utils import (compute_msve, solve_mspbe_weight,
                              solve_msve_weight, set_seed)

from experiment.plotter import plot_data, plot_multiple_runs, print_final_weights


def compute_tf_msve(v_tf: np.ndarray,
                    v_true: np.ndarray,
                    steady_d: np.ndarray) -> float:
    error = v_tf - v_true
    msve = steady_d.dot(error**2)
    return msve.item()


def compute_tf_mspbe(v_tf: np.ndarray,
                     X: np.ndarray,
                     P: np.ndarray,
                     r: np.ndarray,
                     gamma: float,
                     steady_dist: np.ndarray) -> float:

    D = np.diag(steady_dist)
    projection = X @ np.linalg.inv(X.T @ D @ X) @ X.T @ D

    pbe = projection @ (r + gamma * P @ v_tf - v_tf)
    mspbe = steady_dist.dot(pbe**2)
    return mspbe.item()


def train(d: int,
          s: int,
          n: int,
          l: int,
          gamma: float = 0.9,
          lmbd: float = 0.0,
          sample_weight: bool = False,
          manual: bool = False,
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
    manual: whether to use manual weight extraction or not
    mode: 'auto' or 'sequential'
    lr: learning rate
    weight_decay: regularization
    log_interval: logging interval
    save_dir: directory to save logs
    mini_batch_size: mini batch size
    '''

    if save_dir is None:
        startTime = datetime.datetime.now()
        save_dir = os.path.join(
            './logs', "discounted_train", startTime.strftime("%Y-%m-%d-%H-%M-%S"))

    set_seed(random_seed)
    tf = LinearTransformer(d, n, l, lmbd, mode='auto')
    opt = optim.Adam(tf.parameters(), lr=lr, weight_decay=weight_decay)

    log = {'xs': [],
           'mstde': [],
           'msve weight error norm': [],
           'mspbe weight error norm': [],
           'true msve': [],
           'transformer msve': [],
           'transformer mspbe': [],
            'P': [],
            'Q': []
           }

    pro_gen = MDPPromptGenerator(s, d, n, gamma)
    ### Training Loop ###
    for i in range(n_mdps):
        pro_gen.reset_feat()  # reset feature
        pro_gen.reset_mdp(sample_weight=sample_weight)  # reset MDP
        prompt = pro_gen.get_prompt()  # get prompt object
        for _ in range(n_batch_per_mdp):
            mstde = 0.0
            Z_0 = prompt.reset()
            v_current = tf.pred_v(Z_0, manual=manual)
            for _ in range(mini_batch_size):
                Z_next, reward = prompt.step()  # slide window
                v_next = tf.pred_v(Z_next, manual=manual)
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
            w_tf = tf.manual_weight_extraction(
                prompt.context(), d).detach().numpy()
            v_tf = tf.fit_value_func(
                prompt.context(), torch.from_numpy(phi)).detach().numpy()

            log['xs'].append(i)
            log['mstde'].append(mstde.item())

            w_msve = solve_msve_weight(mdp.steady_d, phi, mdp.v)
            log['msve weight error norm'].append(
                weight_error_norm(w_tf, w_msve).item())

            w_mspbe = solve_mspbe_weight(
                mdp.steady_d, mdp.P, phi, mdp.r, gamma)
            log['mspbe weight error norm'].append(
                weight_error_norm(w_tf, w_mspbe).item())

            true_msve = compute_msve(w_msve, mdp.steady_d, phi, mdp.v)
            log['true msve'].append(true_msve)
            tf_msve = compute_tf_msve(v_tf, mdp.v, mdp.steady_d)
            log['transformer msve'].append(tf_msve)

            tf_mspbe = compute_tf_mspbe(
                v_tf, phi, mdp.P, mdp.r, gamma, mdp.steady_d)
            log['transformer mspbe'].append(tf_mspbe)

            log['P'].append(tf.attn.P.detach().numpy().copy())
            log['Q'].append(tf.attn.Q.detach().numpy().copy())

            print('Step:', i)
            #print('Transformer Learned Weight:\n', w_tf)
            #print('MSVE Weight:\n', w_msve)
            #print('MSPBE Weight:\n', w_mspbe)

    prompt.reset()  # reset prompt for fair testing
    mdp = prompt.mdp
    phi = prompt.feature_fun.phi
    w_tf = tf.manual_weight_extraction(prompt.context(), d).detach().numpy()
    v_tf = tf.fit_value_func(
        prompt.context(), torch.from_numpy(phi)).detach().numpy()

    log['xs'].append(n_mdps)
    log['mstde'].append(mstde.item())

    w_msve = solve_msve_weight(mdp.steady_d, phi, mdp.v)
    log['msve weight error norm'].append(
        weight_error_norm(w_tf, w_msve).item())

    w_mspbe = solve_mspbe_weight(mdp.steady_d, mdp.P, phi, mdp.r, gamma)
    log['mspbe weight error norm'].append(
        weight_error_norm(w_tf, w_mspbe).item())

    true_msve = compute_msve(w_msve, mdp.steady_d, phi, mdp.v)
    log['true msve'].append(true_msve)
    tf_msve = compute_tf_msve(v_tf, mdp.v, mdp.steady_d)
    log['transformer msve'].append(tf_msve)

    tf_mspbe = compute_tf_mspbe(v_tf, phi, mdp.P, mdp.r, gamma, mdp.steady_d)
    log['transformer mspbe'].append(tf_mspbe)

    log['P'].append(tf.attn.P.detach().numpy().copy())
    log['Q'].append(tf.attn.Q.detach().numpy().copy())

    print('Step:', n_mdps)
    print('Transformer Learned Weight:\n', w_tf)
    print('MSVE Weight:\n', w_msve)
    print('MSPBE Weight:\n', w_mspbe)

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save log dictionary as JSON
    with open(os.path.join(save_dir, 'discounted_train.pkl'), 'wb') as f:
        pickle.dump(log, f)

    hyperparameters = {
        'd': d,
        's': s,
        'n': n,
        'l': l,
        'gamma': gamma,
        'lmbd': lmbd,
        'sample_weight': sample_weight,
        'manual': manual,
        'n_mdps': n_mdps,
        'mini_batch_size': mini_batch_size,
        'n_batch_per_mdp': n_batch_per_mdp,
        'lr': lr,
        'weight_decay': weight_decay,
        'log_interval': log_interval,
        'random_seed': random_seed,
    }

    # Save hyperparameters as JSON
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(hyperparameters, f)

    plot_data(log, save_dir)
    evaluate_weights(tf, save_dir)


def plot_data(log, save_dir):

    # Loss Plot
    plt.figure()
    plt.plot(log['xs'], log['mstde'], label='MSTDE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_mstde.png'), dpi=300)

    # Weight norm plot
    plt.figure()
    plt.plot(log['xs'], log['msve weight error norm'],
             label='MSVE Weight Error Norm')
    plt.plot(log['xs'], log['mspbe weight error norm'],
             label='MSPBE Weight Error Norm')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Error L2 Norm')
    plt.title('Weight Error Norm vs Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'weight_error_norm.png'), dpi=300)

    # Value Error Plot
    plt.figure()
    plt.plot(log['xs'], log['true msve'], label='True MSVE')
    plt.plot(log['xs'], log['transformer msve'], label='Transformer MSVE')
    plt.xlabel('Epochs')
    plt.ylabel('MSVE')
    plt.title('MSVE vs Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'msve.png'), dpi=300)

    # MSPBE Plot
    plt.figure()
    plt.plot(log['xs'], log['transformer mspbe'], label='Transformer MSPBE')
    plt.xlabel('Epochs')
    plt.ylabel('MSPBE')
    plt.title('MSPBE vs Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'mspbe.png'), dpi=300)


def evaluate_weights(tf, save_dir):
    if tf.mode == 'auto':
        # Save the final P and Q matrices
        final_P = tf.attn.P.detach().numpy()
        final_P /= np.max(np.abs(final_P))
        final_Q = tf.attn.Q.detach().numpy()
        final_Q /= np.max(np.abs(final_Q))

        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
        cax1 = axs[0].matshow(final_P, vmin=-1, vmax=1, cmap='gray')
        axs[0].set_title('Final P Matrix')
        axs[1].matshow(final_Q, vmin=-1, vmax=1, cmap='gray')
        axs[1].set_title('Final Q Matrix')
        fig.colorbar(cax1, ax=axs, orientation='vertical')
        plt.savefig(os.path.join(save_dir, 'final_PQ.png'), dpi=300)
        plt.close(fig)
    else: # sequential
        for l, layer in enumerate(tf.layers):
            final_P = layer.P.detach().numpy()
            final_P /= np.max(np.abs(final_P))
            final_Q = layer.Q.detach().numpy()
            final_Q /= np.max(np.abs(final_Q))

            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
            cax1 = axs[0].matshow(final_P, vmin=-1, vmax=1, cmap='gray')
            axs[0].set_title(f'Final P Matrix Layer {l+1}')
            axs[1].matshow(final_Q, vmin=-1, vmax=1, cmap='gray')
            axs[1].set_title(f'Final Q Matrix Layer {l+1}')
            fig.colorbar(cax1, ax=axs, orientation='vertical')
            plt.savefig(os.path.join(save_dir, f'final_PQ_{l+1}.png'), dpi=300)
            plt.close(fig)


def run_hyperparam_search():
    torch.manual_seed(2)
    np.random.seed(2)
    d = 5
    n = 200
    # l = 4
    # s = int(n/10)  # number of states equal to the context length
    s_frac = 10
    for l in [1, 2, 4, 6]:
        for sw in [True, False]:
            s = int(n/s_frac)
            train(d, s, n, l, lmbd=0.0, sample_weight=sw, epochs=25_000,
                  log_interval=250, save_dir='l{layer}_s{s_}_sw{samp_w}'.format(layer=l, s_=s, samp_w=sw))

    print_final_weights(tf, save_dir)

if __name__ == '__main__':
    d = 4
    n = 100
    l = 4
    s = int(n/10)
    startTime = datetime.datetime.now()
    save_dir = os.path.join(
            './logs', "discounted_train", startTime.strftime("%Y-%m-%d-%H-%M-%S"))
    for seed in [1, 2, 3, 42, 456]:
        train(d, s, n, l, lmbd=0.0,  n_mdps=5000, log_interval=25, random_seed=seed, save_dir= os.path.join(save_dir, f'seed_{seed}'))
    


