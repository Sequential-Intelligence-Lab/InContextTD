import torch
import numpy as np
from model import LinearTransformer


def mean_squared_td_error(v_vec: torch.tensor,
                          v_prime_vec: torch.tensor,
                          gamma: float,
                          Z: torch.tensor,
                          n: int):
    '''
    v_vec: value function prediction
    v_prime_vec: target value function
    Z: prompt (2d+1, n)
    n: context length
    '''
    reward_vec = Z[-1, :n].reshape(1, n)

    tde_vec = reward_vec + gamma*v_prime_vec - v_vec
    mstde = torch.mean(tde_vec**2, dim=1)
    return mstde


def self_consistency_loss(tf: LinearTransformer,
                          w_tf: torch.tensor,
                          context: torch.tensor,
                          X: np.array,
                          steady_dist: np.array):
    d = X.shape[1]
    w_tf = w_tf.detach()
    X = torch.from_numpy(X).float()
    steady_dist = torch.from_numpy(steady_dist).float()
    v_tfs = []
    for feature in X:
        feature_col = torch.zeros((2*d+1, 1))
        feature_col[:d, 0] = feature
        Z_p = torch.cat([context, feature_col], dim=1)
        Z_tf = tf(Z_p)
        v_tfs.append(-Z_tf[-1, -1])
    
    v_tfs = torch.stack(v_tfs, dim=0).reshape(-1, 1)
    squared_error = torch.square(v_tfs - X @ w_tf)
    
    return steady_dist @ squared_error


def weight_error_norm(w1: torch.tensor,
                      w2: torch.tensor):
    '''
    w1: weight vector (d, 1)
    w2: weight vector (d, 1)
    '''
    return torch.norm(w1 - w2)


def value_error(v1: torch.tensor,
                v2: torch.tensor):
    '''
    v1: value vector (1, 1)
    v2: value vector (1, 1)
    '''
    return torch.abs(v1 - v2)
