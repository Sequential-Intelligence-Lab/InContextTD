import torch
import torch.nn as nn


class LinearAttention(nn.Module):
    def __init__(self, d: int, n: int):
        '''
        d: feature dimension
        n: context length
        '''
        super(LinearAttention, self).__init__()
        self.d = d
        self.n = n
        self.P = nn.Parameter(torch.randn(2 * d + 1, 2 * d + 1))
        self.M = torch.eye(n + 1)
        self.M[-1, -1] = 0
        self.Q = nn.Parameter(torch.randn(2 * d + 1, 2 * d + 1))

    def forward(self, Z):
        return Z + 1.0 / self.n * self.P @ Z @ self.M @ Z.T @ self.Q @ Z


if __name__ == '__main__':
    from prompt import Prompt
    d = 4
    n = 10
    gamma = 0.9
    prompt = Prompt(d, n, gamma)
    attention = LinearAttention(d, n)
    Z = prompt.z()
    Z_prime = attention(Z)
    print(Z_prime)
    print(Z_prime.shape)
