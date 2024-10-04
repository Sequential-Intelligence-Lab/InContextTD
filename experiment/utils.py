from typing import Union

import numpy as np
import scipy as sp
import torch


def stack_four(A: torch.Tensor, B: torch.Tensor,
               C: torch.Tensor, D: torch.Tensor):
    top = torch.cat([A, B], dim=1)
    bottom = torch.cat([C, D], dim=1)
    return torch.cat([top, bottom], dim=0)


def stack_four_np(A: np.ndarray, B: np.ndarray,
                  C: np.ndarray, D: np.ndarray):
    top = np.concatenate([A, B], axis=1)
    bottom = np.concatenate([C, D], axis=1)
    return np.concatenate([top, bottom], axis=0)


def scale(matrix: np.ndarray):
    return matrix / np.max(np.abs(matrix))


def analytical_weight_update(w_tf: torch.Tensor,
                             Z: torch.Tensor,
                             d: int,
                             n: int,
                             C: torch.Tensor = None):
    '''
    w_tf: current transformer weight
    Z: context matrix
    d: feature dimension
    n: context length
    C: preconditioning matrix
    '''
    Phi = Z[:d, :n]
    Y = Z[-1, :n].reshape(n, 1)
    prod = Phi @ Y
    if C:
        prod = C @ prod

    return w_tf + 1/n * prod


def compute_steady_dist(P: np.array) -> np.ndarray:
    '''
    P: transition probability matrix
    '''

    n = P.shape[0]
    null_vec = sp.linalg.null_space(np.eye(n) - P.T)

    return (null_vec / np.sum(null_vec)).flatten()


def solve_msve_weight(steady_dist: np.ndarray,
                      X: np.ndarray,
                      v: np.ndarray) -> np.ndarray:
    '''
    P: transition probability matrix
    X: feature matrix
    v: true value
    returns weight minimizing MSVE
    '''
    D = np.diag(steady_dist)
    return np.linalg.inv(X.T @ D @ X) @ X.T @ D @ v


def compute_msve(v_hat: np.ndarray,
                 v: np.ndarray,
                 steady_dist: np.ndarray) -> float:
    '''
    v_hat: predicted value
    v: true value
    steady_dist: steady state distribution
    returns MSVE
    '''
    error = v - v_hat
    msve = steady_dist.dot(error**2)
    return msve.item()


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)


def compare_P(P_tf: np.ndarray, d: int):
    '''
    P_tf: P matrix from transformer
    P_true: hardcoded P matrix that implements TD
    '''
    bottom_right = P_tf[-1, -1]
    avg_abs_all_others = 1/((2*d+1)**2 - 1) * \
        (np.sum(np.abs(P_tf)) - np.abs(P_tf[-1, -1]))
    return bottom_right, avg_abs_all_others


def compare_Q(Q_tf: np.ndarray, d: int):
    '''
    Q_tf: Q matrix from transformer
    Q_true: hardcoded Q matrix that implements TD
    d: feature dimension
    '''
    upper_left_block_trace = np.trace(Q_tf[:d, :d])
    upper_right_block_trace = np.trace(Q_tf[:d, d:2*d])
    # average of absolute values of all other elements
    # (we have 2d+1 x 2d+1 matrix and we are excluding the diagonal entries of the two upper dxd blocks)
    avg_abs_all_others = 1/((2*d+1)**2 - 2*d)*(np.sum(np.abs(Q_tf)) -
                                               upper_right_block_trace - upper_left_block_trace)
    return upper_left_block_trace, upper_right_block_trace, avg_abs_all_others


# Ensures that the hyperparameters are the same across 2 runs
def check_params(params, params_0):
    for key in [k for k in params.keys() if k != 'random_seed']:
        if params[key] != params_0[key]:
            raise ValueError(f'Parameter {key} is not the same across runs.')


def compute_scaling_factor(M1: np.ndarray, M2: np.ndarray) -> float:
    '''
    M1 approximately C*M2
    '''
    m1 = M1.flatten()
    m2 = M2.flatten()
    return m1.dot(m2) / m2.dot(m2)


def in_context_learning_rate(P: np.ndarray,
                             Q: np.ndarray,
                             d: int) -> float:
    c_P = P[-1, -1]

    diag_true = np.ones(2 * d)
    diag_true[:d] = -1
    diag_Q_first = np.array([Q[i, i] for i in range(d)])
    diag_Q_second = np.array([Q[i, i+d] for i in range(d)])
    diag_Q = np.concatenate((diag_Q_first, diag_Q_second))
    c_Q = compute_scaling_factor(diag_Q, diag_true)

    rate = c_P * c_Q
    return rate


def cos_sim(v1: Union[torch.Tensor, np.ndarray],
            v2: Union[torch.Tensor, np.ndarray]) -> float:
    '''
    v1: vector 1
    v2: vector 2
    returns cosine distance between v1 and v2
    '''
    if isinstance(v1, torch.Tensor):
        v1 = v1.detach().numpy()
    if isinstance(v2, torch.Tensor):
        v2 = v2.detach().numpy()

    v1 = v1.flatten()
    v2 = v2.flatten()
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def compare_sensitivity(tf,
                        tf_hard,
                        prompt):
    '''
    computes the expected cosine similarity and l2 norm 
    between the transformers' gradients w.r.t query
    '''
    prompt = prompt.copy()
    Phi: torch.Tensor = prompt.get_feature_mat()
    steady_d: np.ndarray = prompt.mrp.steady_d
    mean_cos_sim = 0.0
    mean_l2_dist = 0.0
    for s, feature in enumerate(Phi):
        prompt.set_query(feature)
        prompt.enable_query_grad()

        tf_v = tf.pred_v(prompt.z())
        tf_v.backward()
        tf_grad = prompt.query_grad().numpy()
        prompt.zero_query_grad()

        tf_v_hard = tf_hard.pred_v(prompt.z())
        tf_v_hard.backward()
        tf_grad_hard = prompt.query_grad().numpy()
        prompt.disable_query_grad()

        mean_cos_sim += steady_d[s]*cos_sim(tf_grad, tf_grad_hard)
    return mean_cos_sim


def implicit_weight_sim(v_tf: np.ndarray,
                          tf_hard,
                          prompt):
    '''
    computes the cosine similarity and l2 distance
    between the batch TD weight (with the fitted learning rate) 
    and the weight of the best linear model that explaines v_tf
    '''
    prompt = prompt.copy()
    steady_d = prompt.mrp.steady_d
    Phi = prompt.get_feature_mat().numpy()
    w_tf = solve_msve_weight(steady_d, Phi, v_tf).flatten()
    prompt.enable_query_grad()
    v_td = tf_hard.pred_v(prompt.z())
    v_td.backward()
    w_td = prompt.query_grad().numpy().flatten()
    prompt.zero_query_grad()
    prompt.disable_query_grad()

    return cos_sim(w_tf, w_td)
