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
        '''
        Z: prompt of shape (2*d+1, n+1)
        '''
        if self.mode == 'auto':
            for _ in range(self.l):
                Z = self.attn(Z)
        else:
            for attn in self.layers:
                Z = attn(Z)

        return Z

    def manual_weight_extraction(self,
                                 context: torch.Tensor,
                                 d: int):
        '''
        context: the context of shape (2*d+1, n)
        d: feature dimension
        '''
        weight = []
        for i in range(d):
            query_col = torch.zeros((2*d+1, 1))
            query_col[i, 0] = -1
            Z_p = torch.concat([context, query_col], dim=1)
            Z_tf = self.forward(Z_p)
            weight.append(Z_tf[-1, -1])
        weight = torch.stack(weight, dim=0)
        return weight.reshape((d, 1))


    def fit_value_func(self,
                       context: torch.tensor,
                       X: torch.tensor,
                       manual: bool = False) -> torch.tensor:
        '''
        context: the context of shape (2*d+1, n)
        X: features of shape (s, d)
        manual: whether to use manual weight extraction or not
        returns the fitted value function given the context in shape (s, 1)
        '''
        if manual:
            w_tf = self.manual_weight_extraction(context, self.d)
            return X @ w_tf
        else:
            tf_v = []
            for feature in X:
                feature_col = torch.zeros((2*d+1, 1))
                feature_col[:d, 0] = feature
                Z_p = torch.cat([context, feature_col], dim=1)
                v_tf = self.pred_v(Z_p)
                tf_v.append(v_tf)
            tf_v = torch.stack(tf_v, dim=0).unsqueeze(1)
            return tf_v

    def pred_v(self, Z: torch.Tensor, manual: bool = False):
        '''
        Z: prompt of shape (2*d+1, n+1)
        manual: whether to use manual weight extraction or not
        predict the value of the query feature
        '''
        if manual:
            context = Z[:, :-1]
            query = Z[:self.d, [-1]]
            w_tf = self.manual_weight_extraction(context, self.d)
            return w_tf.t() @ query
        else:
            Z_tf = self.forward(Z)
            return -Z_tf[-1, -1]


if __name__ == '__main__':
    from experiment.prompt import MDPPromptGenerator
    d = 2
    s = 10
    n = 6
    l = 1
    torch.random.manual_seed(0)
    np.random.seed(0)
    gamma = 0.9
    lmbd = 0.0
    prompt_gen = MDPPromptGenerator(s, d, n, gamma)
    prompt_gen.reset_feat()
    prompt_gen.reset_mdp(sample_weight=False)
    mdp_prompt = prompt_gen.get_prompt()
    Z_0 = mdp_prompt.reset()
    ltf = LinearTransformer(d, n, l, lmbd, mode='auto')
    v_func = ltf.fit_value_func(mdp_prompt.context(), mdp_prompt.get_feature_mat(), manual=False)
    print(v_func.shape)
