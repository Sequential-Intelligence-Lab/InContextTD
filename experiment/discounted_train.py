import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import datetime
import json

from experiment.loss import mean_squared_td_error, weight_error_norm
from experiment.model import LinearTransformer
from experiment.prompt import Feature, MDP_Prompt_Generator, Prompt
from experiment.utils import (compute_mspbe, compute_msve,
                              solve_mspbe_weight,
                              solve_msve_weight)
from MRP.boyan import BoyanChain
import os



def compute_tf_msve(tf: LinearTransformer,
                    context: torch.tensor,
                    X: np.ndarray,
                    true_v: np.ndarray,
                    steady_d: np.ndarray) -> float:
    tf_v = tf.pred_v_array(context, X)
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
    tf_v = tf.pred_v_array(context, X)
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
          epochs: int = 10_000,
          log_interval: int = 100,
          save_dir: str = None,
          mini_batch_size: int = None,
          mdp_eval_samples: int = None):
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
    save_dir: directory to save logs
    mini_batch_size: mini batch size
    '''

    if save_dir is None:
        startTime = datetime.datetime.now()
        save_dir = os.path.join('./logs', "discounted_train", startTime.strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        save_dir = os.path.join('./logs', "discounted_train", save_dir)

    if mdp_eval_samples is None:
        mdp_eval_samples = n

    if mini_batch_size is None:
        mini_batch_size = n
        
    tf = LinearTransformer(d, n, l, lmbd, mode='auto')
    opt = optim.Adam(tf.parameters(), lr=lr, weight_decay=weight_decay)

    log = {'xs': [],
           'mstde': [],
           'msve weight error norm': [],
           'mspbe weight error norm': [],
           'true msve': [],
           'transformer msve': [],
           'transformer mspbe': []
           }
    
    eval_samples_used = 1
    # generate a new feature set and prompt generator
    features = Feature(d, s)
    # generate a new prompt
    if sample_weight:
        w_true = np.random.randn(d, 1).astype(np.float32)
        boyan_mdp = BoyanChain(
            n_states=s, gamma=gamma, weight=w_true, X=features.phi)
    else:
        boyan_mdp = BoyanChain(n_states=s, gamma=gamma)
    pro = MDP_Prompt_Generator(boyan_mdp, features, n, mdp_eval_samples, gamma)

    ### Training Loop ###
    for i in range(epochs):
        vf_predictions = []
        vf_targets = []
        rewards = []
        for _ in range(mini_batch_size):
            if eval_samples_used == mdp_eval_samples:
                # generate a new feature set and prompt generator
                features = Feature(d, s)
                if sample_weight:
                    w_true = np.random.randn(d, 1).astype(np.float32)
                    boyan_mdp = BoyanChain(
                        n_states=s, gamma=gamma, weight=w_true, X=features.phi)
                else:
                    boyan_mdp = BoyanChain(n_states=s, gamma=gamma)
                pro = MDP_Prompt_Generator(boyan_mdp, features, n, mdp_eval_samples, gamma)
                eval_samples_used = 1
            Z_0 = pro.z()
            tf_pred_vf= tf.pred_v(Z_0)
            reward = pro.query_state_reward() # get the observed reward for the query state
            pro.next_prompt()   # generate a new prompt by sliding over the mdp samples
            eval_samples_used += 1
            with torch.no_grad(): # no gradient computation for the target value function
                tf_target_vf = tf.pred_v(pro.z()) # now the query is the successor state

            vf_predictions.append(tf_pred_vf)
            vf_targets.append(tf_target_vf)
            rewards.append(reward)
            
        vf_pred_tensor = torch.stack(vf_predictions)
        vf_targets_tensor = torch.stack(vf_targets)
        rewards_tensor = torch.tensor(rewards)
        mstde = mean_squared_td_error(rewards_tensor, vf_pred_tensor, vf_targets_tensor, gamma)
        # extract the learned weights from the transformer
        opt.zero_grad()
        mstde.backward()
        opt.step()

        if i % log_interval == 0:
            w_tf = tf.manual_weight_extraction(Z_0, d)
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

    log['xs'].append(epochs)
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

    print('Step:', epochs)
    print('Transformer Learned Weight:\n', w_tf.detach().numpy())
    print('MSVE Weight:\n', w_msve)
    print('MSPBE Weight:\n', w_mspbe)

    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Save log dictionary as JSON
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
        'steps': epochs,
        'log_interval': log_interval
    }

    # Save hyperparameters as JSON
    with open(os.path.join(save_dir, 'params.json'), 'w') as f:
        json.dump(hyperparameters, f)

    plot_data(log, save_dir)
    evaluate_weights(tf, save_dir)

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


def evaluate_weights(tf, save_dir):
    # Save the final P and Q matrices
    final_P = tf.attn.P.detach().numpy()
    final_Q = tf.attn.Q.detach().numpy()

    plt.figure()
    plt.matshow(final_P)
    plt.colorbar()
    plt.title('Final P Matrix')
    plt.savefig(os.path.join(save_dir, 'final_P.png'), dpi=300)

    plt.figure()
    plt.matshow(final_Q)
    plt.colorbar()
    plt.title('Final Q Matrix')
    plt.savefig(os.path.join(save_dir, 'final_Q.png'), dpi=300)

def run_hyperparam_search():
    torch.manual_seed(2)
    np.random.seed(2)
    d = 5
    n = 200
    #l = 4
    #s = int(n/10)  # number of states equal to the context length
    s_frac = 10
    for l in [1,2,4,6]:
        for sw in [True, False]:
            s = int(n/s_frac)
            train(d, s, n, l, lmbd=0.0, sample_weight=sw, epochs=25_000, 
                    log_interval=250,save_dir='l{layer}_s{s_}_sw{samp_w}'.format(layer=l, s_=s, samp_w=sw))

if __name__ == '__main__':
    torch.manual_seed(2)
    np.random.seed(2)
    d = 5
    n = 200
    l = 3
    s = int(n/10) 
    train(d, s, n, l, lmbd=0.0, sample_weight=False, epochs=20_000, mdp_eval_samples= n,mini_batch_size=n, log_interval=200)
