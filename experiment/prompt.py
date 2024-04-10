import torch
from MRP.boyan import BoyanChain
import numpy as np
import random

class Feature:
    def __init__(self, d: int, s: int):
        '''
        d: dimension of the feature vector
        s: number of states

        '''
        self.d = d
        self.s = s
        self.phi = np.random.randn(s, d)

    def get_feature(self, s: int):
        return self.phi[s]


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


class MDP_Prompt:
    def __init__(self,
                 mdp: BoyanChain,
                 features: Feature,
                 n: int,
                 gamma: float):
        '''
        mdp: an instance of a BoyanChain MDP
        features: the features
        n: context length
        gamma: discount factor
        '''
        self.mdp = mdp
        self.features = features
        self.n = n
        self.gamma = gamma

        rows = []
        # sample from initial state distribution
        s = mdp.reset()
        for _ in range(self.n):
            s_prime, r = mdp.step(s)
            row = np.concatenate([features.get_feature(s), self.gamma*features.get_feature(s_prime), [r]])
            rows.append(row)
            s = s_prime
        # randomly sample a query state from the stationary distribution
        q_state = mdp.sample_stationary()
        rows.append(np.concatenate([features.get_feature(q_state), np.zeros(features.d), [0]]))

        prompt = np.stack(rows, axis=-1)
        self.z_0 = torch.tensor(prompt, dtype=torch.float32)

    def z(self):
        return self.z_0
    
    def context(self):
        return self.z_0[:, :-1]


if __name__ == '__main__':
    d = 3
    s = 10
    n = 6
    gamma = 0.9
    feat = Feature(d, s)
    bc = BoyanChain(s, gamma)
    mdp_prompt = MDP_Prompt(bc, feat, n, gamma)

    print("Features")
    print(feat.phi)
    print("Z_0")
    print(mdp_prompt.z())
    print("Context")
    print(mdp_prompt.context())
    assert mdp_prompt.z_0.shape == (2*d+1, n+1)
