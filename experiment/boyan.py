import numpy as np
from typing import Tuple
from experiment.utils import compute_steady_dist

class BoyanChain:
    def __init__(self, 
                 n_states: int,
                 X: np.array, 
                 gamma: float = 0.9, 
                 noise: float = 0) -> None:
        '''
        n_states: number of states of the Boyan Chain
        X: feature matrix of shape (n, d)
        gamma: discount factor
        noise: Gaussian noise added to the reward
        '''
        self.n_states = n_states
        self.gamma = gamma
        self.noise = noise
        self.P = np.zeros((n_states, n_states))
        for i in range(n_states - 2):
            self.P[i, i + 1] = 0.5
            self.P[i, i + 2] = 0.5
        self.P[-2, -1] = 1.0
        self.P[-1, :] = 1/n_states
        d = X.shape[1]
        self.w = np.random.randn(d).reshape(d, 1)
        self.v = X.dot(self.w)
        self.r = (np.eye(n_states) - gamma * self.P).dot(self.v)
        self.mu = np.ones(n_states) / n_states # uniform intial distribution
        assert np.allclose(self.P.sum(axis=1), 1)
        self.stationary_d = compute_steady_dist(self.P)
    
    def reset(self) -> int:
        s = np.random.choice(self.n_states, p=self.mu)
        return s
    
    def step(self, state: int) -> Tuple[int, float]:
        assert 0 <= state < self.n_states
        next_state = np.random.choice(self.n_states, p=self.P[state])
        reward = self.r[state, 0] + self.noise * np.random.randn() # add Gaussian noise
        return next_state, reward



if __name__ == '__main__':
    bc = BoyanChain(n_states=5, X=np.random.randn(5, 2))
    print('stochastic matrix\n', bc.P)
    print('weight\n', bc.w)
    print('reward\n', bc.r)
    print('value\n', bc.v)
    print('stationary distribution\n', bc.stationary_d)
