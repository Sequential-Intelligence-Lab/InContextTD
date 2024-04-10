import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import datetime
import json

from experiment.loss import mean_squared_td_error, weight_error_norm
from experiment.model import LinearTransformer
from experiment.prompt import Feature, MDP_Prompt, Prompt
from experiment.utils import (compute_mspbe, compute_msve,
                              manual_weight_extraction, solve_mspbe_weight,
                              solve_msve_weight)
from MRP.boyan import BoyanChain
import os



def tf_pred_v(tf: LinearTransformer,
              context: torch.tensor,
              X: np.ndarray) -> torch.tensor:
    d = X.shape[1]
    X = torch.from_numpy(X)
    tf_v = []
    for feature in X:
        feature_col = torch.zeros((2*d+1, 1))
        feature_col[:d, 0] = feature
        Z_p = torch.cat([context, feature_col], dim=1)
        Z_tf = tf(Z_p)
        tf_v.append(-Z_tf[-1, -1])
    tf_v = torch.stack(tf_v, dim=0).reshape(-1, 1)
    return tf_v

def compute_tf_msve(tf: LinearTransformer,
                    context: torch.tensor,
                    X: np.ndarray,
                    true_v: np.ndarray,
                    steady_d: np.ndarray) -> float:
    tf_v = tf_pred_v(tf, context, X)
    tf_v = tf_v.detach().numpy()
    error = tf_v - true_v
    msve = steady_d.dot(error**2)
    return msve.item()


def compute_tf_mspbe(tf: LinearTransformer,
                     context: torch.tensor,
                     X: np.ndarray,
                     P: np.ndarray,
                     r: np.ndarray,
                     gamma: float,
                     steady_dist: np.ndarray) -> float:
    tf_v = tf_pred_v(tf, context, X)
    tf_v = tf_v.detach().numpy()
    
    D = np.diag(steady_dist)
    projection = X @ np.linalg.inv(X.T @ D @ X) @ X.T @ D

    pbe = projection @ (r + gamma * P @ tf_v - tf_v)
    mspbe = steady_dist.dot(pbe**2)
    return mspbe.item()

def train(d: int,
          s: int,
          n: int,
          l: int,
          gamma: float = 0.9,
          lmbd: float = 0.0,
          sample_weight: bool = True,
          lr: float = 0.001,
          weight_decay=1e-6,
          steps: int = 50_000,
          log_interval: int = 100,
          save_dir: str = None):
    '''
    d: feature dimension
    s: number of states
    n: context length
    l: number of layers
    gamma: discount factor
    lmbd: eligibility trace decay
    sample_weight: sample a random true weight vector
    lr: learning rate
    weight_decay: regularization
    steps: number of training steps
    log_interval: logging interval
    '''

    if save_dir is None:
        startTime = datetime.datetime.now()
        save_dir = os.path.join('./logs', "discounted_train", startTime.strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        save_dir = os.path.join('./logs', "discounted_train", save_dir)
    
    print(save_dir)
    tf = LinearTransformer(d, n, l, lmbd, mode='auto')
    opt = optim.Adam(tf.parameters(), lr=lr, weight_decay=weight_decay)
    features = Feature(d, s)

    log = {'xs': [],
           'mstde': [],
           'msve weight error norm': [],
           'mspbe weight error norm': [],
           'true msve': [],
           'transformer msve': [],
           'transformer mspbe': []
           }
    for i in range(steps):
        # generate a new prompt
        if sample_weight:
            w_true = np.random.randn(d, 1).astype(np.float32)
            boyan_mdp = BoyanChain(
                n_states=s, gamma=gamma, weight=w_true, X=features.phi)
        else:
            boyan_mdp = BoyanChain(n_states=s, gamma=gamma)

        # Markovian prompt based prompt from Boyan Chain
        pro = MDP_Prompt(boyan_mdp, features, n, gamma)

        Z_0 = pro.z()

        # extract the learned weights from the transformer
        w_tf = manual_weight_extraction(tf, Z_0, d)
        mstde = mean_squared_td_error(w_tf, Z_0, d, n)
        opt.zero_grad()
        mstde.backward()
        opt.step()

        if i % log_interval == 0:
            log['xs'].append(i)
            log['mstde'].append(mstde.item())

            w_msve = solve_msve_weight(boyan_mdp.steady_d, features.phi, boyan_mdp.v)
            w_msve_tensor = torch.from_numpy(w_msve)
            log['msve weight error norm'].append(weight_error_norm(w_tf.detach(), w_msve_tensor).item())

            w_mspbe = solve_mspbe_weight(boyan_mdp.steady_d, boyan_mdp.P, features.phi, boyan_mdp.r, gamma)
            w_mspbe_tensor = torch.from_numpy(w_mspbe)
            log['mspbe weight error norm'].append(weight_error_norm(w_tf.detach(), w_mspbe_tensor).item())

            true_msve = compute_msve(w_msve, boyan_mdp.steady_d, features.phi, boyan_mdp.v)
            log['true msve'].append(true_msve)
            tf_msve = compute_tf_msve(tf, pro.context(), features.phi, boyan_mdp.v, boyan_mdp.steady_d)
            log['transformer msve'].append(tf_msve)

            tf_mspbe = compute_tf_mspbe(tf, pro.context(), features.phi, boyan_mdp.P, boyan_mdp.r, gamma, boyan_mdp.steady_d)
            log['transformer mspbe'].append(tf_mspbe)

            print('Step:', i)
            print('Transformer Learned Weight:\n', w_tf.detach().numpy())
            print('MSVE Weight:\n', w_msve)
            print('MSPBE Weight:\n', w_mspbe)

    log['xs'].append(steps)
    log['mstde'].append(mstde.item())

    w_msve = solve_msve_weight(boyan_mdp.steady_d, features.phi, boyan_mdp.v)
    w_msve_tensor = torch.from_numpy(w_msve)
    log['msve weight error norm'].append(weight_error_norm(w_tf.detach(), w_msve_tensor).item())

    w_mspbe = solve_mspbe_weight(boyan_mdp.steady_d, boyan_mdp.P, features.phi, boyan_mdp.r, gamma)
    w_mspbe_tensor = torch.from_numpy(w_mspbe)
    log['mspbe weight error norm'].append(weight_error_norm(w_tf.detach(), w_mspbe_tensor).item())

    true_msve = compute_msve(w_msve, boyan_mdp.steady_d, features.phi, boyan_mdp.v)
    log['true msve'].append(true_msve)
    tf_msve = compute_tf_msve(tf, pro.context(), features.phi, boyan_mdp.v, boyan_mdp.steady_d)
    log['transformer msve'].append(tf_msve)

    tf_mspbe = compute_tf_mspbe(tf, pro.context(), features.phi, boyan_mdp.P, boyan_mdp.r, gamma, boyan_mdp.steady_d)
    log['transformer mspbe'].append(tf_mspbe)

    print('Step:', steps)
    print('Transformer Learned Weight:\n', w_tf.detach().numpy())
    print('MSVE Weight:\n', w_msve)
    print('MSPBE Weight:\n', w_mspbe)

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(os.path.join(save_dir,'discounted_train.pkl'), 'wb') as f:
        pickle.dump(log, f)

    hyperparameters = {
        'd': d,
        's': s,
        'n': n,
        'l': l,
        'gamma': gamma,
        'lmbd': lmbd,
        'sample_weight': sample_weight,
        'lr': lr,
        'weight_decay': weight_decay,
        'steps': steps,
        'log_interval': log_interval
    }

    # Save log dictionary as JSON
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(hyperparameters, f)

    plot_data(log, save_dir)

def plot_data(log,save_dir):

    # Loss Plot
    plt.figure()
    plt.plot(log['xs'], log['mstde'], label='MSTDE')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss vs Steps')
    plt.legend()
    plt.savefig(os.path.join(save_dir,'loss_mstde.png'), dpi= 300)

    # Weight norm plot
    plt.figure()
    plt.plot(log['xs'], log['msve weight error norm'], label='MSVE Weight Error Norm')
    plt.plot(log['xs'], log['mspbe weight error norm'], label='MSPBE Weight Error Norm')
    plt.xlabel('Steps')
    plt.ylabel('Weight Error L2 Norm')
    plt.title('Weight Error Norm vs Steps')
    plt.legend()
    plt.savefig(os.path.join(save_dir,'weight_error_norm.png'),dpi=300)

    # Value Error Plot
    plt.figure()
    plt.plot(log['xs'], log['true msve'], label='True MSVE')
    plt.plot(log['xs'], log['transformer msve'], label='Transformer MSVE')
    plt.xlabel('Steps')
    plt.ylabel('MSVE')
    plt.title('MSVE vs Steps')
    plt.legend()
    plt.savefig(os.path.join(save_dir,'msve.png'),dpi=300)

    # MSPBE Plot
    plt.figure()
    plt.plot(log['xs'], log['transformer mspbe'], label='Transformer MSPBE')
    plt.xlabel('Steps')
    plt.ylabel('MSPBE')
    plt.title('MSPBE vs Steps')
    plt.legend()
    plt.savefig(os.path.join(save_dir,'mspbe.png'), dpi=300)

if __name__ == '__main__':
    torch.manual_seed(2)
    np.random.seed(2)
    d = 5
    n = 200
    #l = 4
    #s = int(n/10)  # number of states equal to the context length
    for l in [1,2,4,6]:
        for s_frac in [10, 15, 20]:
            for sw in [True, False]:
                s = int(n/s_frac)
                train(d, s, n, l, lmbd=0.0, sample_weight=sw, steps=25_000, 
                      log_interval=250,save_dir='l{layer}_s{s_}_sw{samp_w}'.format(layer=l, s_=s, samp_w=sw))
