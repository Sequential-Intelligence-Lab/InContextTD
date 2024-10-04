from typing import Tuple

import numpy as np

from MRP.mrp import MRP
from utils import compute_steady_dist


class Loop(MRP):
    def __init__(self,
                 n_states: int,
                 gamma: float = 0.9,
                 threshold: float = 0.5,
                 weight: np.ndarray = None,
                 phi: np.ndarray = None) -> None:
        '''
        n_states: number of states of the Loop MRP
        gamma: discount factor
        weight: weight vector
        Phi: feature matrix of shape (n_states, d)
        '''
        super().__init__(n_states)
        # common attributes for all variants
        self.gamma = gamma

        # initialze transition matrix
        connectivity = np.random.uniform(0.01, 0.99,
                                         size=(n_states, n_states)) > threshold
        for i in range(n_states):
            connectivity[i, i] = False  # no self loops
            connectivity[i, (i+1) % n_states] = True  # connect to next state
        P = np.random.uniform(0.01, 0.99,
                              size=(n_states, n_states))*connectivity
        self.P = P / P.sum(axis=1, keepdims=True)  # normalize
        assert np.allclose(self.P.sum(axis=1), 1)

        self.steady_d = compute_steady_dist(self.P)

        self.mu = np.random.uniform(0.01, 0.99, size=n_states)
        self.mu /= self.mu.sum()

        if weight is not None:
            assert phi is not None, 'feature matrix must be provided if weight is given'
            self.w = weight
            self.v = phi.dot(self.w)
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


if __name__ == '__main__':
    np.random.seed(0)
    mrp = Loop(5, threshold=0.6)
    print(mrp.P)
    print(mrp.steady_d)
