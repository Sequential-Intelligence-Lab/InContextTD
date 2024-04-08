import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from experiment.loss import (mean_squared_td_error, self_consistency_loss,
                             value_error, weight_error_norm)
from experiment.model import LinearTransformer
from experiment.prompt import Feature, MDP_Prompt, Prompt
from experiment.utils import manual_weight_extraction, solve_mspbe, solve_msve
# from torch_in_context_td import HC_Transformer
from MRP.boyan import BoyanChain


def train(d: int,
          s: int,
          n: int,
          l: int,
          gamma: float = 0.9,
          lmbd: float = 0.0,
          sample_weight: bool = True,
          lr: float = 0.001,
          weight_decay = 1e-6,
          steps: int = 50_000,
          log_interval: int = 100):
    
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

    tf = LinearTransformer(d, n, l, lmbd, mode='auto')
    opt = optim.Adam(tf.parameters(), lr=lr, weight_decay=weight_decay)
    features = Feature(d, s)

    writer = SummaryWriter(log_dir='./logs')

    for i in range(steps): 
        #generate a new prompt
        if sample_weight:
            w_true = np.random.randn(d, 1).astype(np.float32)
            boyan_mdp = BoyanChain(n_states=s, gamma=gamma, weight=w_true, X=features.phi)
        else:
            boyan_mdp = BoyanChain(n_states=s, gamma=gamma)

        pro =  MDP_Prompt(boyan_mdp, features, n, gamma)   # Markovian prompt based prompt from Boyan Chain

        Z_0 = pro.z()
        phi_query = Z_0[:d, [n]]

        # extract the learned weights from the transformer
        w_tf = manual_weight_extraction(tf, Z_0, d)

        mstde = mean_squared_td_error(w_tf, Z_0, d, n)
        sc_loss = self_consistency_loss(w_tf, phi_query, Z_0)

        opt.zero_grad()
        total_loss = mstde + sc_loss
        total_loss.backward()
        opt.step()

        if i % log_interval == 0:
            writer.add_scalar('Loss/Mean Square TD Error', mstde.item(), i)
            writer.add_scalar('Loss/Self-Consistency Loss', sc_loss.item(), i)
            w_msve, _ = solve_msve(boyan_mdp.P, features.phi, boyan_mdp.v)
            w_msve_tensor = torch.from_numpy(w_msve)
            writer.add_scalar('Loss/MSVE Weight Error Norm', weight_error_norm(w_tf.detach(), w_msve_tensor).item(), i)
            w_mspbe, _ = solve_mspbe(boyan_mdp.P, features.phi, boyan_mdp.r, gamma)
            w_mspbe_tensor = torch.from_numpy(w_mspbe)
            writer.add_scalar('Loss/MSPBE Weight Error Norm', weight_error_norm(w_tf.detach(), w_mspbe_tensor).item(), i)

            print('Step:', i)
            print('Transformer Learned Weight:\n', w_tf.detach().numpy())
            print('MSVE Weight:\n', w_msve)
            print('MSPBE Weight:\n', w_mspbe)

    writer.add_scalar('Loss/Mean Square TD Error', mstde.item(), steps)
    writer.add_scalar('Loss/Self-Consistency Loss', sc_loss.item(), steps)
    w_msve, _ = solve_msve(boyan_mdp.P, features.phi, boyan_mdp.v)
    w_msve_tensor = torch.from_numpy(w_msve)
    writer.add_scalar('Loss/MSVE Weight Error Norm', weight_error_norm(w_tf, w_msve_tensor).item(), steps)
    w_mspbe, _ = solve_mspbe(boyan_mdp.P, features.phi, boyan_mdp.r, gamma)
    w_mspbe_tensor = torch.from_numpy(w_mspbe)
    writer.add_scalar('Loss/MSPBE Weight Error Norm', weight_error_norm(w_tf, w_mspbe_tensor).item(), steps)
    print('Step:', steps)
    print('Transformer Learned Weight:\n', w_tf.detach().numpy())
    print('MSVE Weight:\n', w_msve)
    print('MSPBE Weight:\n', w_mspbe)
    writer.flush()
    writer.close()

if __name__ == '__main__':
    torch.manual_seed(2)
    np.random.seed(2)
    d = 4
    n = 200
    l = 4
    s= int(n/10) # number of states equal to the context length
    train(d, s, n, l, lmbd=0.0, sample_weight=False, steps=30_000)
