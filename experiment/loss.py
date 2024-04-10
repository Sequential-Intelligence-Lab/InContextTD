import torch
import numpy as np
from model import LinearTransformer


def mean_squared_td_error(w: torch.tensor,
                          Z: torch.tensor,
                          d: int,
                          n: int):
    '''
    w: weight vector (d, 1)
    Z: prompt (2d+1, n)
    d: feature dimension
    n: context length
    '''

    Phi = Z[:d, :n]
    Phi_prime = Z[d:2*d, :n]
    reward_vec = Z[-1, :n].reshape(1, n)

    v_vec = w.t() @ Phi
    # use detach() to prevent backpropagation through w here
    v_prime_vec = w.t().detach() @ Phi_prime
    tde_vec = reward_vec + v_prime_vec - v_vec
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
