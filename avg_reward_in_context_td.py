import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def stack_four(A, B, C, D):
    top = torch.cat([A, B], dim=1)
    bottom = torch.cat([C, D], dim=1)
    return torch.cat([top, bottom], dim=0)

class TFLayer(nn.Module):
    def __init__(self, d, n):
        super(TFLayer, self).__init__()
        self.d = d
        self.n = n
        self.P = torch.zeros((2 * d + 2, 2 * d + 2))
        self.P[-1, -1] = 1
        self.P[-2, -2] = 1
        self.M = torch.eye(n + 1)
        self.M[-1, -1] = 0
        I = torch.eye(d)
        O = torch.zeros((d, d))
        self.M1 = stack_four(-I, I, O, O)
        self.C = torch.randn(d, d)
        self.B = stack_four(self.C.t(), O, O, O)
        self.A = torch.mm(self.B, self.M1)
        self.Q = torch.zeros_like(self.P)
        self.Q[:2*d, :2*d] = self.A

    def forward(self, Z):
        next_Z = Z + 1.0 / self.n * self.P @ Z @ self.M @ Z.T @ self.Q @ Z
        return next_Z

class Transformer(nn.Module):
    def __init__(self, l, d, n):
        super(Transformer, self).__init__()
        self.n = n
        self.layers = nn.ModuleList([TFLayer(d, n) for _ in range(l)])
        self.Cs = [layer.C for layer in self.layers]

    def forward(self, Z):
        v = []
        av = []
        for layer in self.layers:
            Z = layer.forward(Z)
            v.append(Z[-2, -1].item())
            av.append(Z[-1, -1].item())
        return v, av, Z

class Prompt:
    def __init__(self, d, n, gamma):
        self.n = n
        self.gamma = gamma

        # randomly initialize some feature vectors
        self.phi = torch.cat([torch.randn(d, 1) for _ in range(n+1)], dim=1)
        self.phi_prime = [torch.randn(d, 1) for _ in range(n)]
        self.phi_prime.append(torch.zeros((d, 1)))
        self.phi_prime = gamma * torch.cat(self.phi_prime, dim=1)

        # randomly initialize some rewards 
        self.r = [torch.randn(1).item() for _ in range(self.n)]
        # initialize r_bar
        # let r_bar[i] be the sums of the rewards up through element i
        self.r_bar = [1/(i+1)*sum(self.r[:i+1]) for i in range(self.n)]
        self.r.append(0)
        self.r = torch.tensor(self.r)
        self.r = torch.reshape(self.r, (1, -1))

        self.r_bar.append(0)
        print(len(self.r_bar))
        self.r_bar = torch.tensor(self.r_bar)
        self.r_bar = torch.reshape(self.r_bar, (1, -1))

    def z(self):
        return torch.cat([self.phi, self.phi_prime, self.r, self.r_bar], dim=0)

    def td_update(self, w, C):
        u = 0
        for j in range(self.n):
            td_error = self.r[0, j] - self.r_bar[0,j] + torch.mm(w.t(), self.phi_prime[:, [j]]) - torch.mm(w.t(), self.phi[:, [j]])
            u += td_error * self.phi[:, [j]]
        u /= self.n
        u = torch.mm(C, u)
        w += u
        v = torch.mm(w.t(), self.phi[:, [-1]])
        return w, v.item()

def g(pro, tf, phi, phi_prime, r, r_bar):
    pro.phi[:, [-1]] = phi
    pro.phi_prime[:, [-1]] = phi_prime
    pro.r[0, -1] = r
    pro.r_bar[0, -1] = r_bar
    _, Z = tf.forward(pro.z())
    return Z


def verify(d, n, l):
    # no discounting in average reward setting
    gamma = 1
    tf = Transformer(l, d, n)
    pro = Prompt(d, n, gamma)
    tf_value, tf_av_value, _ = tf.forward(pro.z())

    w = torch.zeros((d, 1))
    td_value = []
    for i in range(l):
        w, v = pro.td_update(w, tf.Cs[i])
        td_value.append(v)
    td_value = np.array(td_value).flatten()
    #import pdb; pdb.set_trace()
    print((np.array(tf_value) - np.array(tf_av_value)) + td_value)

if __name__ == '__main__':
    verify(4, 9, 10)



