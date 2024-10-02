import torch
from typing import Tuple


class Prompt:
    def __init__(self, d: int, n: int, gamma: float):
        '''
        d: feature dimension
        n: context length
        gamma: discount factor
        '''
        self.d = d
        self.n = n
        self.gamma = gamma
        self.phi = torch.cat([torch.randn(d, 1) for _ in range(n+1)], dim=1)
        self.trace = torch.zeros((d, n))
        self.phi_prime = [torch.randn(d, 1) for _ in range(n)]
        self.phi_prime.append(torch.zeros((d, 1)))
        self.phi_prime = gamma * torch.cat(self.phi_prime, dim=1)
        self.r = [torch.randn(1).item() for _ in range(self.n)]
        self.r.append(0)
        self.r = torch.tensor(self.r)
        self.r = torch.reshape(self.r, (1, -1))

    def z(self) -> torch.Tensor:
        return torch.cat([self.phi, self.phi_prime, self.r], dim=0)

    def td_update(self, w: torch.Tensor, C: torch.Tensor,
                  lmbd: float, alpha: float = 1.0) -> Tuple[torch.Tensor, float]:
        '''
        w: weight vector
        C: preconditioning matrix
        lmdb: eligibility trace decay
        alpha: learning rate
        '''
        u = torch.zeros((self.d, 1))  # TD update
        e = torch.zeros((self.d, 1))  # eligibility trace
        for i in range(self.n):  # batch TD aggregation
            target = self.r[0, i] + w.t() @ self.phi_prime[:, [i]]
            tde = target - w.t() @ self.phi[:, [i]]
            e = lmbd * e + self.phi[:, [i]]
            u += tde * e
        u *= alpha/self.n
        w += C @ u
        v = w.t() @ self.phi[:, [-1]]
        return w, v.item()

    def rg_update(self, w: torch.Tensor, C: torch.Tensor,
                  alpha: float = 1.0) -> Tuple[torch.Tensor, float]:
        '''
        w: weight vector
        C: preconditioning matrix
        alpha: learning rate
        '''
        u = torch.zeros((self.d, 1))
        for i in range(self.n):
            target = self.r[0, i] + w.t() @ self.phi_prime[:, [i]]
            tde = target - w.t() @ self.phi[:, [i]]
            u += tde * (self.phi[:, [i]] - self.phi_prime[:, [i]])
        u *= alpha / self.n
        w += C @ u
        v = w.t() @ self.phi[:, [-1]]
        return w, v.item()
