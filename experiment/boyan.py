import numpy as np
from typing import Tuple

class BoyanChain:
    def __init__(self, n_states: int, gamma: float = 0.9) -> None:
        self.n_states = n_states
        self.gamma = gamma
        self.P = np.zeros((n_states, n_states))
        for i in range(n_states - 2):
            self.P[i, i + 1] = 0.5
            self.P[i, i + 2] = 0.5
        self.P[-2, -1] = 1.0
        self.P[-1, :] = 1/n_states
        self.v = np.random.randn(n_states).reshape(n_states, 1)
        self.r = (np.eye(n_states) - gamma * self.P).dot(self.v)
        self.mu = np.ones(n_states) / n_states # uniform intial distribution
    
    def reset(self) -> int:
        s = np.random.choice(self.n_states, p=self.mu)
        return s
    
    def step(self, state: int) -> Tuple[int, float]:
        assert 0 <= state < self.n_states
        next_state = np.random.choice(self.n_states, p=self.P[state])
        reward = self.r[state, 0]
        return next_state, reward



if __name__ == '__main__':
    bc = BoyanChain(10)
    initial = bc.reset()
    print(bc.step(initial))
