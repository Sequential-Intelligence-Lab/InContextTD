import torch
import torch.nn as nn
import numpy as np


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
        M = torch.eye(n+1)
        for col in range(n):
            for row in range(col+1, n):
                M[row, col] = self.lmbd*M[row-1, col]
        M[-1, -1] = 0
        self.M = M
        self.P = nn.Parameter(torch.empty(2 * d + 1, 2 * d + 1))
        self.Q = nn.Parameter(torch.empty(2 * d + 1, 2 * d + 1))

    def forward(self, Z):
        return Z + 1.0 / self.n * self.P @ Z @ self.M @ Z.T @ self.Q @ Z


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
    
    def manual_weight_extraction(self,
                             Z: torch.Tensor,
                             d: int):
        '''
        tf: transformer model
        Z: prompt
        d: feature dimension
        '''

        context = Z[:, :-1]
        weight = []
        for i in range(d):
            query = torch.zeros((2*d+1, 1))
            query[i, 0] = -1
            Z_p = torch.concat([context, query], dim=1)
            Z_tf = self.forward(Z_p)
            weight.append(Z_tf[-1, -1])
        weight = torch.stack(weight, dim=0)
        return weight.reshape((d, 1))
    
    def pred_v_manual(
            self,
            context: torch.tensor,
            X: np.ndarray) -> torch.tensor:
        d = X.shape[1]
        X = torch.from_numpy(X)
        w_tf = self.manual_weight_extraction(context, d)
        v_vec = w_tf.t() @ X.T
        return v_vec
    
    # Computes the value function for all the features in X given the context using the forward pass
    # and taking the negative of the bottom right element of the output tensor
    # returns a tensor of shape (len(X), 1)
    def pred_v_array(self,
              context: torch.tensor,
              X: np.ndarray) -> torch.tensor:
        d = X.shape[1]
        X = torch.from_numpy(X)
        tf_v = []
        for feature in X:
            feature_col = torch.zeros((2*d+1, 1))
            feature_col[:d, 0] = feature
            Z_p = torch.cat([context, feature_col], dim=1)
            Z_tf = self.forward(Z_p)
            tf_v.append(-Z_tf[-1, -1])
        tf_v = torch.stack(tf_v, dim=0).reshape(-1, 1)
        return tf_v
    
    # Computes the estimated value function based on the query
    def pred_v(self, Z: torch.Tensor):
        return -self.forward(Z)[-1, -1]
    



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
