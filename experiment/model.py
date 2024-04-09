import torch
import torch.nn as nn


class LinearAttention(nn.Module):
    def __init__(self, d: int, n: int, lmbd: float = 0.0):
        '''
        d: feature dimension
        n: context length
        '''
        super(LinearAttention, self).__init__()
        self.d = d
        self.n = n
        self.lmbd = lmbd

        self.P = nn.Parameter(torch.empty(2 * d + 1, 2 * d + 1))
        self.Q = nn.Parameter(torch.empty(2 * d + 1, 2 * d + 1))

    def forward(self, Z):
        h = Z.shape[1]
        # dynamic masking
        M = torch.eye(h)
        for col in range(self.n):
            for row in range(col+1, self.n):
                M[row, col] = self.lmbd*M[row-1, col]
        for i in range(self.n, h):
            M[i, i] = 0
        return Z + 1.0 / self.n * self.P @ Z @ M @ Z.T @ self.Q @ Z


class LinearTransformer(nn.Module):
    def __init__(self,
                 d: int,
                 n: int,
                 l: int,
                 lmbd: float = 0.0,
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
            attn = LinearAttention(d, n, lmbd)
            nn.init.xavier_normal_(attn.P, gain=0.1)
            nn.init.xavier_normal_(attn.Q, gain=0.1)
            self.attn = attn
        elif mode == 'sequential':
            self.layers = nn.ModuleList(
                [LinearAttention(d, n, lmbd) for _ in range(l)])
            for attn in self.layers:
                nn.init.xavier_normal_(attn.P, gain=0.1/l)
                nn.init.xavier_normal_(attn.Q, gain=0.1/l)

    def forward(self, Z):
        if self.mode == 'auto':
            for _ in range(self.l):
                Z = self.attn(Z)
        else:
            for attn in self.layers:
                Z = attn(Z)

        return Z


if __name__ == '__main__':
    from experiment.prompt import Prompt
    d = 3
    n = 6
    l = 10
    torch.random.manual_seed(0)
    gamma = 0.9
    lmbd = 0.9
    prompt = Prompt(d, n, gamma)
    Z_0 = prompt.z()
    ltf = LinearTransformer(d, n, l, lmbd, mode='sequential')
    Z_tf = ltf(Z_0)
    print(Z_tf)
    print(Z_tf.shape)
