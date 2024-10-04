from typing import Tuple

import numpy as np

from MRP.mrp import MRP
from utils import compute_steady_dist


class BoyanChain(MRP):
    def __init__(self,
                 n_states: int,
                 gamma: float = 0.9,
                 weight: np.ndarray = None,
                 X: np.ndarray = None) -> None:
        '''
        n_states: number of states of the Boyan Chain
        gamma: discount factor
        noise: Gaussian noise added to the reward
        X: feature matrix of shape (n, d)
        '''
        super().__init__(n_states)
        # common attributes for all variants
        self.gamma = gamma

        # initialze transition matrix
        self.P = np.zeros((n_states, n_states))
        for i in range(n_states - 2):
            trans_sample = np.random.uniform(0.01, 0.99)
            self.P[i, i + 1] = trans_sample
            self.P[i, i + 2] = 1-trans_sample
        self.P[-2, -1] = 1.0
        self.P[-1, :] = np.random.uniform(0.01, 0.99, size=n_states)
        self.P[-1, :] /= self.P[-1, :].sum()
        assert np.allclose(self.P.sum(axis=1), 1)
        self.steady_d = compute_steady_dist(self.P)

        self.mu = np.random.uniform(0.01, 0.99, size=n_states)
        self.mu /= self.mu.sum()

        if weight is not None:
            assert X is not None, 'feature matrix X must be provided if weight is given'
            self.w = weight
            self.v = X.dot(self.w)
            self.r = (np.eye(n_states) - gamma * self.P).dot(self.v)
        else:
            self.r = np.random.uniform(low=-1.0, high=1.0, size=(n_states, 1))
            self.v = np.linalg.inv(np.eye(n_states) - gamma * self.P).dot(self.r)

    def reset(self) -> int:
        s = np.random.choice(self.n_states, p=self.mu)
        return s

    def step(self, state: int) -> Tuple[int, float]:
        assert 0 <= state < self.n_states
        next_state = np.random.choice(self.n_states, p=self.P[state])
        reward = self.r[state, 0]
        return next_state, reward
    
    def sample_stationary(self) -> int:
        return np.random.choice(self.n_states, p=self.steady_d)
    
    def copy(self) -> 'BoyanChain':
        bc = BoyanChain(self.n_states, self.gamma)
        bc.P = self.P.copy()
        bc.mu = self.mu.copy()
        bc.r = self.r.copy()
        bc.v = self.v.copy()
        bc.steady_d = self.steady_d.copy()
        if hasattr(self, 'w'):
            bc.w = self.w.copy()
        return bc


if __name__ == '__main__':
    w = np.random.randn(2, 1)
    bc = BoyanChain(n_states=5, weight=w, X=np.random.randn(5, 2))
    print('stochastic matrix\n', bc.P)
    print('weight\n', bc.w)
    print('reward\n', bc.r)
    print('value\n', bc.v)
    print('stationary distribution\n', bc.steady_d)
    print('initial distribution\n', bc.mu)
