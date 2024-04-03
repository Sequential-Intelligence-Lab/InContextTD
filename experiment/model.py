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

        self.P = nn.Parameter(torch.empty(2 * d + 1, 2 * d + 1))
        nn.init.xavier_normal_(self.P, gain=0.1)

        self.M = torch.eye(n + 1)
        self.M[-1, -1] = 0

        self.Q = nn.Parameter(torch.empty(2 * d + 1, 2 * d + 1))
        nn.init.xavier_normal_(self.Q, gain=0.1)

    def forward(self, Z):
        return Z + 1.0 / self.n * self.P @ Z @ self.M @ Z.T @ self.Q @ Z


class LinearTransformer(nn.Module):
    def __init__(self,
                 d: int,
                 n: int,
                 l: int,
                 mode='auto'):
        '''
        d: feature dimension
        n: context length
        mode: 'auto' or 'sequential'
        '''
        super(LinearTransformer, self).__init__()
        self.d = d
        self.n = n
        self.l = l
        self.mode = mode
        if mode == 'auto':
            self.attn = LinearAttention(d, n)
        elif mode == 'sequential':
            self.layers = nn.ModuleList([LinearAttention(d, n) for _ in range(l)])
        


    def forward(self, Z):
        if self.mode == 'auto':
            for _ in range(self.l):
                Z = self.attn(Z)
        else:
            for attn in self.layers:
                Z = attn(Z)
        
        return -Z[-1, -1].item(), Z


if __name__ == '__main__':
    from experiment.prompt import Prompt
    d = 4
    n = 10
    l = 5
    gamma = 0.9
    prompt = Prompt(d, n, gamma)
    Z_0 = prompt.z()
    ltf = LinearTransformer(d, n, l, mode='auto')
    _, Z_tf = ltf(Z_0)
    print(Z_tf)
    print(Z_tf.shape)
