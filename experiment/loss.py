import torch
import numpy as np
from experiment.model import LinearTransformer


def mean_squared_td_error(reward_vec: torch.tensor,
                          v_vec: torch.tensor,
                          v_prime_vec: torch.tensor,
                          gamma: float):
    '''
    reward_vec: rewards
    v_vec: value function prediction
    v_prime_vec: target value function
    gamma: discount factor
    '''
    tde_vec = reward_vec + gamma*v_prime_vec - v_vec
    mstde = torch.mean(tde_vec**2)
    return mstde


def weight_error_norm(w1: np.ndarray,
                      w2: np.ndarray):
    '''
    w1: weight vector (d, 1)
    w2: weight vector (d, 1)
    '''
    return np.linalg.norm(w1 - w2)


def value_error(v1: torch.tensor,
                v2: torch.tensor):
    '''
    v1: value vector (1, 1)
    v2: value vector (1, 1)
    '''
    return torch.abs(v1 - v2)
