import json
import os

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from experiment.prompt import MRPPromptGenerator
from MRP.mrp import MRP
from rnn.model import RNN
from utils import compute_msve, set_seed


def _init_log() -> dict:
    log = {'xs': [],
           'msve': []}
    return log


def _init_save_dir(save_dir: str) -> None:
    if save_dir is None:
        startTime = datetime.datetime.now()
        save_dir = os.path.join('./logs',
                                startTime.strftime("%Y-%m-%d-%H-%M-%S"))


def _save_log(log: dict, save_dir: str) -> None:
    for key, value in log.items():
        log[key] = np.array(value)
    np.savez(os.path.join(save_dir, 'data.npz'), **log)


def train_rnn(d: int,
              s: int,
              n: int,
              l: int,
              gamma: float = 0.9,
              activation: str = 'tanh',
              sample_weight: bool = False,
              lr: float = 0.001,
              weight_decay=1e-6,
              n_mrps: int = 1000,
              mini_batch_size: int = 64,
              n_batch_per_mrp: int = 5,
              log_interval: int = 10,
              save_dir: str = None,
              random_seed: int = 2):
    '''
    d: feature dimension
    s: number of states
    n: context length
    l: number of layers
    gamma: discount factor
    activation: activation function (e.g. softmax, identity, relu)
    sample_weight: sample a random true weight vector
    lr: learning rate
    weight_decay: regularization
    log_interval: logging interval
    save_dir: directory to save logs
    mini_batch_size: mini batch size
    n_batch_per_mrp: number of batches per MRP
    n_mrps: number of MRPs
    random_seed: random seed
    '''
    ckpt_dir = os.path.join(save_dir, 'ckpt')

    _init_save_dir(save_dir)

    set_seed(random_seed)

    rnn = RNN(d, l, activation)

    opt = optim.Adam(rnn.parameters(), lr=lr, weight_decay=weight_decay)

    log = _init_log()

    pro_gen = MRPPromptGenerator(s, d, n, gamma)

    flag_vec = torch.zeros((1, n+1))
    flag_vec[0, -1] = 1
    ### Training Loop ###
    for i in tqdm(range(1, n_mrps+1)):
        pro_gen.reset_feat()  # reset feature
        pro_gen.reset_mrp(sample_weight=sample_weight)  # reset MRP
        prompt = pro_gen.get_prompt()  # get prompt object
        for _ in range(n_batch_per_mrp):
            mstde = 0.0
            Z_0 = prompt.reset()
            Z_0 = torch.cat((Z_0, flag_vec), dim=0)
            v_current = rnn(Z_0.t())
            for _ in range(mini_batch_size):
                Z_next, reward = prompt.step()  # slide window
                Z_next = torch.cat((Z_next, flag_vec), dim=0)
                v_next = rnn(Z_next.t())
                tde = reward + gamma*v_next.detach() - v_current
                mstde += tde**2
                v_current = v_next
            mstde /= mini_batch_size
            opt.zero_grad()
            mstde.backward()
            opt.step()

        if i % log_interval == 0:
            prompt.reset()  # reset prompt for fair testing
            mrp: MRP = prompt.mrp
            phi = prompt.get_feature_mat()
            steady_d: np.ndarray = mrp.steady_d
            ctxt = prompt.context()
            v_tf: np.ndarray = rnn.fit_value_func(ctxt, phi).detach().numpy()
            msve = compute_msve(v_tf, mrp.v, steady_d)

            log['xs'].append(i)
            log['msve'].append(msve)

    

    _save_log(log, save_dir)

    torch.save(rnn.state_dict(), os.path.join(ckpt_dir, f'params.pt'))

    hyperparameters = {
        'd': d,
        's': s,
        'n': n,
        'l': l,
        'gamma': gamma,
        'activation': activation,
        'sample_weight': sample_weight,
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
