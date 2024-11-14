import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, d: int, l: int, activation: str = 'tanh'):
        '''
        d: feature dimension
        l: number of layers
        activation: activation function
        '''
        super(RNN, self).__init__()
        self.d = d
        self.rnn = nn.RNN(input_size=2*d+2,
                          hidden_size=d,
                          num_layers=l,
                          nonlinearity=activation,
                          batch_first=True)
        self.linear = nn.Linear(d, 1)

    def forward(self, Z):
        _, hn = self.rnn(Z)
        return self.linear(hn[-1])

    def fit_value_func(self,
                       context: torch.Tensor,
                       phi: torch.Tensor) -> torch.Tensor:
        '''
        context: the context of shape (2*d+1, n)
        phi: features of shape (s, d)
        returns the fitted value function given the context in shape (s, 1)
        '''
        n = context.shape[1]
        input = []
        for feature in phi:
            query_col = torch.zeros((2*self.d+1, 1))
            query_col[:self.d, 0] = feature
            # integrate the query feature
            Z = torch.cat([context, query_col], dim=1)
             # add flags to indicate the query feature
            flags = torch.zeros((1, n+1))
            flags[0, -1] = 1
            Z = torch.cat([Z, flags], dim=0).t().unsqueeze(0)  
            input.append(Z)
        input = torch.vstack(input)
        return self.forward(input)