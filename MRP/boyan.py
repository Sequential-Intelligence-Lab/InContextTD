from typing import Tuple

import numpy as np

from experiment.utils import compute_steady_dist
from MRP.mrp import MRP


class BoyanChain(MRP):
    def __init__(self,
                 n_states: int,
                 gamma: float = 0.9,
                 noise: float = 0.1,
                 initial_dist: np.array = None,
                 weight: np.array = None,
                 X: np.array = None) -> None:
        '''
        n_states: number of states of the Boyan Chain
        gamma: discount factor
        noise: Gaussian noise added to the reward
        X: feature matrix of shape (n, d)
        '''
        super().__init__(n_states)
        # common attributes for all variants
        self.gamma = gamma
        self.noise = noise

        # initialze transition matrix
        self.P = np.zeros((n_states, n_states))
        for i in range(n_states - 2):
            self.P[i, i + 1] = 0.5
            self.P[i, i + 2] = 0.5
        self.P[-2, -1] = 1.0
        self.P[-1, :] = 1/n_states
        assert np.allclose(self.P.sum(axis=1), 1)
        self.steady_d = compute_steady_dist(self.P)

        if initial_dist is not None:
            self.mu = initial_dist
        else:
            # uniform intial distribution
            self.mu = np.ones(n_states) / n_states

        if weight is not None:
            assert X is not None, 'feature matrix X must be provided if weight is given'
            self.w = weight
            self.v = X.dot(self.w)
            self.r = (np.eye(n_states) - gamma * self.P).dot(self.v)
        else:
            self.r = np.random.randn(n_states, 1)
            self.v = np.linalg.inv(
                np.eye(n_states) - gamma * self.P).dot(self.r)

    def reset(self) -> int:
        s = np.random.choice(self.n_states, p=self.mu)
        return s

    def step(self, state: int) -> Tuple[int, float]:
        assert 0 <= state < self.n_states
        next_state = np.random.choice(self.n_states, p=self.P[state])
        reward = self.r[state, 0] + self.noise * \
            np.random.randn()  # add Gaussian noise
        return next_state, reward
    
    def sample_stationary(self) -> int:
        return np.random.choice(self.n_states, p=self.steady_d)


if __name__ == '__main__':
    w = np.random.randn(2, 1)
    bc = BoyanChain(n_states=5, weight=w, X=np.random.randn(5, 2))
    print('stochastic matrix\n', bc.P)
    print('weight\n', bc.w)
    print('reward\n', bc.r)
    print('value\n', bc.v)
    print('stationary distribution\n', bc.steady_d)
