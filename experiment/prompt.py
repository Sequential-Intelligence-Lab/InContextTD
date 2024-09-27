from collections import deque

import numpy as np
import torch

from MRP.boyan import BoyanChain
from MRP.loop import Loop


class Feature:
    def __init__(self, d: int, s: int):
        '''
        d: dimension of the feature vector
        s: number of states

        '''
        self.d = d
        self.s = s
        if s <= d:
            self.phi = np.eye(s, dtype=np.float32)
        else:
            self.phi = np.random.uniform(
                low=-1, high=1, size=(s, d)).astype(np.float32)

    def __call__(self, s: int):
        return self.phi[s]

    def copy(self) -> 'Feature':
        f = Feature(self.d, self.s)
        f.phi = self.phi.copy()
        return f


class Prompt:
    def __init__(self,
                 d: int,
                 n: int,
                 gamma: float,
                 w: torch.Tensor = None,
                 noise: float = 0.1):
        '''
        d: feature dimension
        n: context length
        gamma: discount factor
        w: weight vector (optional)
        noise: reward noise level
        '''
        self.d = d
        self.n = n
        self.gamma = gamma
        self.phi = torch.randn(d, n+1)

        phi_prime = torch.randn(d, n)
        phi_prime = torch.concat([phi_prime, torch.zeros((d, 1))], dim=1)
        self.phi_prime = gamma * phi_prime

        if w:
            self.w = w
        else:
            self.w = torch.randn((d, 1))

        r = self.w.t() @ self.phi - self.w.t() @ self.phi_prime
        r[0, -1] = 0
        self.r = r

    def z(self):
        return torch.cat([self.phi, self.phi_prime, self.r], dim=0)

    def td_update(self,
                  w: torch.Tensor,
                  C: torch.Tensor = None):
        '''
        w: weight vector
        C: preconditioning matrix
        '''
        u = 0
        for j in range(self.n):
            target = self.r[0, j] + w.t() @ self.phi_prime[:, [j]]
            td_error = target - w.t() @ self.phi[:, [j]]
            u += td_error * self.phi[:, [j]]
        u /= self.n
        if C:
            u = C @ u  # apply conditioning matrix
        new_w = w + u
        v = new_w.t() @ self.phi[:, [-1]]
        return new_w, v.item()


class MDPPrompt:
    def __init__(self,
                 d: int,
                 n: int,
                 gamma: float,
                 mdp: BoyanChain,
                 feature_fun: Feature):
        '''
        d: feature dimension
        n: context length
        gamma: discount factor
        mdp: an instance of a BoyanChain MDP
        feature_fun: a function that returns the feature vector of a state  
        '''
        self.d = d
        self.n = n
        self.gamma = gamma
        self.mdp = mdp
        self.feature_fun = feature_fun

    def reset(self):
        self.feature_window = deque(maxlen=self.n+2)
        self.reward_window = deque(maxlen=self.n+1)
        # populates the feature and rewards
        self.s = self.mdp.reset()
        self.feature_window.append(self.feature_fun(self.s))
        for _ in range(self.n+1):
            s_prime, r = self.mdp.step(self.s)
            self.feature_window.append(self.feature_fun(s_prime))
            self.reward_window.append(r)
            self.s = s_prime

        self._store_data()

        return self.z()

    def step(self):
        # step the MDP
        s_prime, r = self.mdp.step(self.s)
        self.feature_window.append(self.feature_fun(s_prime))
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

    def context(self):
        return self._context

    def query(self):
        return self._query

    def set_query(self, query: torch.Tensor):
        query = query.reshape(self.d, 1)
        self._query = query

    def enable_query_grad(self):
        self._query.requires_grad_(True)

    def disable_query_grad(self):
        self._query.requires_grad_(False)

    def query_grad(self):
        assert self._query.grad is not None, "no gradient associated with the query"
        return self._query.grad.reshape((self.d, 1))

    def zero_query_grad(self):
        self._query.grad = None

    def get_feature_mat(self):
        return torch.from_numpy(self.feature_fun.phi)

    def z(self):
        query_col = torch.concat(
            [self._query, torch.zeros((self.d+1, 1))], dim=0)
        return torch.concat([self._context, query_col], dim=1)

    def td_update(self,
                  w: torch.Tensor,
                  C: torch.Tensor = None,
                  residual: bool = False,
                  lr: float = 1):
        '''
        w: weight vector
        C: preconditioning matrix
        '''
        u = 0
        for j in range(self.n):
            target = self.r[0, j] + w.t() @ self.phi_prime[:, [j]]
            td_error = target - w.t() @ self.phi[:, [j]]
            if residual:
                u += td_error * (self.phi[:, [j]] - self.phi_prime[:, [j]])
            else:
                u += td_error * self.phi[:, [j]]
        u *= lr/self.n
        if C:
            u = C @ u  # apply conditioning matrix
        new_w = w + u
        v = new_w.t() @ self.phi[:, [-1]]
        return new_w, v.item()

    def copy(self) -> 'MDPPrompt':
        mdp_prompt = MDPPrompt(self.d, self.n, self.gamma,
                               self.mdp.copy(), self.feature_fun.copy())
        mdp_prompt.feature_window = self.feature_window.copy()
        mdp_prompt.reward_window = self.reward_window.copy()
        mdp_prompt.s = self.s
        mdp_prompt._store_data()
        return mdp_prompt


class MDPPromptGenerator:
    def __init__(self,
                 s: int,
                 d: int,
                 n: int,
                 gamma: float,
                 mdp_class: str = 'boyan'):
        '''
        s: number of states
        d: feature dimension
        n: context length
        gamma: discount factor
        '''

        self.s = s
        self.d = d
        self.n = n
        self.gamma = gamma
        self.mdp_class = mdp_class

    def reset_mdp(self, sample_weight: bool = False, threshold: float = 0.5):
        w = np.random.randn(self.d, 1) if sample_weight else None
        if self.mdp_class == 'boyan':
            self.mdp = BoyanChain(n_states=self.s, gamma=self.gamma,
                                  weight=w, X=self.feat.phi)
        elif self.mdp_class == 'loop':
            self.mdp = Loop(n_states=self.s, gamma=self.gamma, threshold=threshold,
                            weight=w, Phi=self.feat.phi)
        else:
            raise ValueError("Unknown MDP type")

    def reset_feat(self):
        self.feat = Feature(self.d, self.s)

    def get_prompt(self):
        assert self.mdp is not None, "call reset_mdp first"
        assert self.feat is not None, "call reset_feat first"
        return MDPPrompt(self.d, self.n, self.gamma, self.mdp, self.feat)


if __name__ == '__main__':
    d = 3
    s = 10
    n = 6
    eval_len = 3
    gamma = 0.9

    prompt_gen = MDPPromptGenerator(s, d, n, gamma)
    prompt_gen.reset_feat()
    prompt_gen.reset_mdp(sample_weight=False)
    mdp_prompt = prompt_gen.get_prompt()
    Z_0 = mdp_prompt.reset()
    print(Z_0)
    Z_1 = mdp_prompt.step()
    print(Z_1)

    prompt_gen.reset_feat()
    prompt_gen.reset_mdp(sample_weight=True)
    mdp_prompt = prompt_gen.get_prompt()
    Z_0 = mdp_prompt.reset()
    print(Z_0)
    Z_1 = mdp_prompt.step()
    print(Z_1)

    pro = mdp_prompt.copy()
    print(pro.mdp.w)
