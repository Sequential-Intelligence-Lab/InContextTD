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


def compute_msve(w: np.ndarray,
                 steady_dist: np.ndarray,
                 X: np.ndarray,
                 v: np.ndarray) -> float:
    '''
    w: weight vector
    steady_dist: steady state distribution
    X: feature matrix
    v: true value
    returns MSVE
    '''
    error = X @ w - v
    msve = steady_dist.dot(error**2)
    return msve.item()


def solve_mspbe_weight(steady_dist: np.ndarray,
                       P: np.ndarray,
                       X: np.ndarray,
                       r: np.ndarray,
                       gamma: float) -> np.ndarray:
    '''
    steady_dist: steady state distribution
    P: transition probability matrix
    X: feature matrix
    r: reward vector
    gamma: discount factor
    returns weight minimizing MSPBE
    '''
    n = P.shape[0]
    D = np.diag(steady_dist)

    A = X.T @ D @ (gamma*P - np.eye(n)) @ X
    b = - X.T @ D @ r
    w = np.linalg.inv(A) @ b

    return w


def compute_mspbe(w: np.ndarray,
                  steady_dist: np.ndarray,
                  P: np.ndarray,
                  X: np.ndarray,
                  r: np.ndarray,
                  gamma: float):
    '''
    w: weight vector
    steady_dist: steady state distribution
    P: transition probability matrix
    X: feature matrix
    r: reward vector
    gamma: discount factor
    returns MSPBE
    '''
    D = np.diag(steady_dist)
    projection = X @ np.linalg.inv(X.T @ D @ X) @ X.T @ D
    v_hat = X @ w
    pbe = projection @ (r + gamma * P @ v_hat) - v_hat
    # this quantity should be very close to zero because td fixed point brings MSPBE to zero
    mspbe = steady_dist.dot(pbe**2)
    return mspbe.item()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    from MRP.boyan import BoyanChain
    set_seed(0)
    X = np.random.randn(10, 3)
    bc = BoyanChain(n_states=10)
    w_msve = solve_msve_weight(bc.steady_d, X, bc.v)
    print('MSVE Weight\n', w_msve)
    msve = compute_msve(w_msve, bc.steady_d, X, bc.v)
    print('MSVE\n', msve)
    w_mspbe = solve_mspbe_weight(bc.steady_d, bc.P, X, bc.r, bc.gamma)
    print('MSPBE Weight\n', w_mspbe)
    mspbe = compute_mspbe(w_mspbe, bc.steady_d, bc.P, X, bc.r, bc.gamma)
    print('MSPBE\n', mspbe)
