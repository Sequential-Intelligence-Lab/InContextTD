from typing import Tuple

import numpy as np
import scipy as sp
import torch

import experiment.model as model


def stack_four(A: torch.Tensor, B: torch.Tensor,
               C: torch.Tensor, D: torch.Tensor):
    top = torch.cat([A, B], dim=1)
    bottom = torch.cat([C, D], dim=1)
    return torch.cat([top, bottom], dim=0)


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


def manual_weight_extraction(tf: model.LinearTransformer,
                             Z: torch.Tensor,
                             d: int):
    '''
    tf: transformer model
    Z: prompt
    d: feature dimension
    '''

    context = Z[:, :-1]
    weight = []
    for i in range(d):
        query = torch.zeros((2*d+1, 1))
        query[i, 0] = -1
        Z_p = torch.concat([context, query], dim=1)
        Z_tf = tf(Z_p)
        weight.append(Z_tf[-1, -1])
    weight = torch.stack(weight, dim=0)
    return weight.reshape((d, 1))


def compute_steady_dist(P: np.array) -> np.ndarray:
    '''
    P: transition probability matrix
    '''

    n = P.shape[0]
    null_vec = sp.linalg.null_space(np.eye(n) - P.T)

    return (null_vec / np.sum(null_vec)).flatten()


def solve_msve(P: np.ndarray,
               X: np.ndarray,
               v: np.ndarray) -> Tuple[np.ndarray, int]:
    '''
    P: transition probability matrix
    X: feature matrix
    v: true value
    returns weight minimizing MSVE and the corresponding MSVE
    '''
    steady_dist = compute_steady_dist(P)
    D = np.diag(steady_dist)
    w = np.linalg.inv(X.T @ D @ X) @ X.T @ D @ v
    error = X @ w - v
    msve = steady_dist.dot(error**2)
    return w, msve.item()


def solve_mspbe(P: np.ndarray,
                X: np.ndarray,
                r: np.ndarray,
                gamma: float) -> Tuple[np.ndarray, int]:
    '''
    P: transition probability matrix
    X: feature matrix
    r: reward vector
    gamma: discount factor
    returns weight minimizing MSPBE and the corresponding MSPBE
    '''
    n = P.shape[0]
    steady_dist = compute_steady_dist(P)
    D = np.diag(steady_dist)

    A = X.T @ D @ (gamma*P - np.eye(n)) @ X
    b = - X.T @ D @ r
    w = np.linalg.inv(A) @ b

    projection = X @ np.linalg.inv(X.T @ D @ X) @ X.T @ D
    v_hat = X @ w
    pbe = projection @ (r + gamma * P @ v_hat) - v_hat
    mspde =  steady_dist.dot(pbe**2) # this quantity should be very close to zero because td fixed point brings MSPBE to zero

    return w, mspde.item()



if __name__ == '__main__':
    from MRP.boyan import BoyanChain
    np.random.seed(0)
    X = np.random.randn(10, 3)
    bc = BoyanChain(n_states=10)
    w_msve, msve = solve_msve(bc.P, X, bc.v)
    print(w_msve, msve)
    w_mspbe, mspbe = solve_mspbe(bc.P, X, bc.r, bc.gamma)
    print(w_mspbe, mspbe)
