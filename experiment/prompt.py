from collections import deque

import numpy as np
import torch

from MRP.boyan import BoyanChain
from MRP.loop import Loop
from MRP.mrp import MRP
from MRP.cart_pole import CartPoleEnvironment
from typing import Tuple


class Feature:
    def __init__(self, d: int, s: int, mode: str = 'random'):
        '''
        d: dimension of the feature vector
        s: number of states
        mode: feature type
        '''
        self.d = d
        self.s = s
        if np.isinf(s):
            if mode == 'random':
                # self.A = np.random.uniform(low=-1, high=1,
                #                            size=(d, 4)).astype(np.float32)
                # diagonal_entries = np.random.choice([-1, 1], size=4)
                # self.A = np.diag(diagonal_entries).astype(np.float32)
                self.A = np.random.uniform(low=-1, high=1,
                                           size=(2, 2, 2, 2, d)).astype(np.float32)
            elif mode == 'one-hot':
                self.A = np.eye(4).astype(np.float32)
            else:
                raise ValueError("Unknown mode")
        else:
            if mode == 'random':
                self.phi = np.random.uniform(low=-1, high=1,
                                             size=(s, d)).astype(np.float32)
            elif mode == 'one-hot':
                assert s == d, "number of states must be equal to the feature dimension"
                self.phi = np.eye(s, dtype=np.float32)
            else:
                raise ValueError("Unknown mode")

    def __call__(self, s):
        if isinstance(s, int):  # if s is an integer then return the feature vector of the state
            return self.phi[s]
        # if is already a vector, then return its transformation to a feature vector
        elif isinstance(s, np.ndarray):
            # import pdb; pdb.set_trace()
            return self.A[*s]
        else:
            raise ValueError("s must be an int or a numpy array")

    def copy(self) -> 'Feature':
        f = Feature(self.d, self.s)
        f.phi = self.phi.copy()
        return f


class MRPPrompt:
    def __init__(self,
                 d: int,
                 n: int,
                 gamma: float,
                 mrp: MRP,
                 feature_fun: Feature):
        '''
        d: feature dimension
        n: context length
        gamma: discount factor
        mrp: an instance of a Markov Reward Process
        feature_fun: a function that returns the feature vector of a state  
        '''
        self.d = d
        self.n = n
        self.gamma = gamma
        self.mrp = mrp
        self.feature_fun = feature_fun

    def reset(self) -> torch.Tensor:
        self.feature_window = deque(maxlen=self.n+2)
        self.reward_window = deque(maxlen=self.n+1)
        # populates the feature and rewards
        self.s = self.mrp.reset()
        self.feature_window.append(self.feature_fun(np.array(self.s)))
        for _ in range(self.n+1):
            s_prime, r = self.mrp.step(self.s)
            self.feature_window.append(self.feature_fun(np.array(s_prime)))
            self.reward_window.append(r)
            self.s = s_prime

        self._store_data()

        return self.z()

    def step(self) -> Tuple[torch.Tensor, float]:
        # step the MRP
        s_prime, r = self.mrp.step(self.s)
        self.feature_window.append(self.feature_fun(np.array(s_prime)))
        self.reward_window.append(r)
        self.s = s_prime

        self._store_data()
        return self.z(), r

    def _store_data(self):
        features = np.array(self.feature_window, dtype=np.float32)
        rewards = np.array(self.reward_window, dtype=np.float32)
        self.phi = torch.from_numpy(features[:self.n]).T
        self.phi_prime = self.gamma*torch.from_numpy(features[1:self.n+1]).T
        self.r = torch.from_numpy(rewards[:self.n]).unsqueeze(0)
        self._context = torch.concat([self.phi, self.phi_prime, self.r], dim=0)
        self._query = torch.from_numpy(features[self.n+1]).reshape(self.d, 1)

    def context(self) -> torch.Tensor:
        return self._context

    def query(self) -> torch.Tensor:
        return self._query

    def set_query(self, query: torch.Tensor):
        query = query.reshape(self.d, 1)
        self._query = query

    def enable_query_grad(self):
        self._query.requires_grad_(True)

    def disable_query_grad(self):
        self._query.requires_grad_(False)

    def query_grad(self) -> torch.Tensor:
        assert self._query.grad is not None, "no gradient associated with the query"
        return self._query.grad.reshape((self.d, 1))

    def zero_query_grad(self):
        self._query.grad = None

    def get_feature_mat(self) -> torch.Tensor:
        if self.feature_fun.s == np.inf:  # cannot return feature matrix for continuous state space
            return None
        return torch.from_numpy(self.feature_fun.phi)

    def z(self) -> torch.Tensor:
        query_col = torch.concat([self._query, torch.zeros((self.d+1, 1))],
                                 dim=0)
        return torch.concat([self._context, query_col], dim=1)

    def td_update(self,
                  w: torch.Tensor,
                  lr: float = 1.0) -> Tuple[torch.Tensor, float]:
        '''
        w: weight vector
        lr: learning rate
        '''
        u = torch.zeros((self.d, 1))
        for i in range(self.n):
            target = self.r[0, i] + w.t() @ self.phi_prime[:, [i]]
            tde = target - w.t() @ self.phi[:, [i]]
            u += tde * self.phi[:, [i]]

        w += lr/self.n * u
        v = w.t() @ self.phi[:, [-1]]
        return w, v.item()

    def copy(self) -> 'MRPPrompt':
        mrp_prompt = MRPPrompt(self.d, self.n, self.gamma,
                               self.mrp.copy(), self.feature_fun.copy())
        mrp_prompt.feature_window = self.feature_window.copy()
        mrp_prompt.reward_window = self.reward_window.copy()
        mrp_prompt.s = self.s
        mrp_prompt._store_data()
        return mrp_prompt


class MRPPromptGenerator:
    def __init__(self,
                 s: int,
                 d: int,
                 n: int,
                 gamma: float,
                 mrp_class: str = 'boyan'):
        '''
        s: number of states
        d: feature dimension
        n: context length
        gamma: discount factor
        mrp_class: type of MRP
        '''

        self.s = s
        self.d = d
        self.n = n
        self.gamma = gamma
        self.mrp_class = mrp_class

    def reset_mrp(self, sample_weight: bool = False, threshold: float = 0.5):
        w = np.random.randn(self.d, 1) if sample_weight else None
        if self.mrp_class == 'boyan':
            self.mrp = BoyanChain(n_states=self.s, gamma=self.gamma,
                                  weight=w, X=self.feat.phi)
        elif self.mrp_class == 'loop':
            self.mrp = Loop(n_states=self.s, gamma=self.gamma, threshold=threshold,
                            weight=w, Phi=self.feat.phi)
        elif self.mrp_class == 'cartpole':
            self.mrp = CartPoleEnvironment()
        else:
            raise ValueError("Unknown MRP type")

    def reset_feat(self):
        self.feat = Feature(self.d, self.s)

    def get_prompt(self) -> MRPPrompt:
        assert self.mrp is not None, "call reset_mrp first"
        assert self.feat is not None, "call reset_feat first"
        return MRPPrompt(self.d, self.n, self.gamma, self.mrp, self.feat)
