import torch
import torch.nn as nn

from experiment.utils import stack_four


class DiscoutedTDLayer(nn.Module):
    def __init__(self, d: int, n: int, lmbd: float):
        '''
        d: feature dimension
        n: context length
        lmbd: eligibility trace decay
        '''
        super(DiscoutedTDLayer, self).__init__()
        self.d = d
        self.n = n
        self.P = torch.zeros((2 * d + 1, 2 * d + 1))
        self.P[-1, -1] = 1
        self.M = torch.eye(n + 1)
        for col in range(n):
            for row in range(col+1, n):
                self.M[row, col] = lmbd*self.M[row-1, col]
        self.M[-1, -1] = 0
        I = torch.eye(d)
        O = torch.zeros((d, d))
        self.M1 = stack_four(-I, I, O, O)
        self.C = torch.eye(d)
        self.B = stack_four(self.C.t(), O, O, O)
        self.A = torch.mm(self.B, self.M1)
        self.Q = torch.zeros_like(self.P)
        self.Q[:2*d, :2*d] = self.A

    def forward(self, Z: torch.Tensor, alpha: float = 1.0):
        '''
        Z: prompt matrix
        alpha: learning rate
        '''
        next_Z = Z + alpha / self.n * self.P @ Z @ self.M @ Z.T @ self.Q @ Z
        return next_Z


class DiscountedTDTransformer(nn.Module):
    def __init__(self, l: int, d: int, n: int, lmbd: float):
        '''
        l: number of layers
        d: feature dimension
        n: context length
        lmbd: eligibility trace decay
        '''
        super(DiscountedTDTransformer, self).__init__()
        self.n = n
        self.layers = nn.ModuleList([DiscoutedTDLayer(d, n, lmbd) for _ in range(l)])
        self.Cs = [layer.C for layer in self.layers] # preconditioning matrices

    def forward(self, Z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        '''
        Z: prompt matrix
        alpha: learning rate
        '''
        v = []
        for layer in self.layers:
            Z = layer.forward(Z, alpha)
            # negate it to align with the convention
            v.append(-Z[-1, -1].item())
        v = torch.tensor(v) 
        return v


class RGLayer(nn.Module):
    def __init__(self, d: int, n: int):
        '''
        d: feature dimension
        n: context length
        '''
        super(RGLayer, self).__init__()
        self.d = d
        self.n = n
        self.P = torch.zeros((2 * d + 1, 2 * d + 1))
        self.P[-1, -1] = 1
        self.M = torch.eye(n + 1)
        self.M[-1, -1] = 0
        I = torch.eye(d)
        O = torch.zeros((d, d))
        self.M1 = stack_four(-I, I, O, O)
        self.M2 = -self.M1
        self.C = torch.eye(d)  # torch.randn(d, d)
        self.B = stack_four(self.C.t(), O, O, O)
        self.A = self.M2.t() @ self.B @ self.M1
        self.Q = torch.zeros_like(self.P)
        self.Q[:2*d, :2*d] = self.A

    def forward(self, Z: torch.Tensor, alpha: float = 1.0):
        next_Z = Z + alpha / self.n * self.P @ Z @ self.M @ Z.T @ self.Q @ Z
        return next_Z
    
class RGTransformer(nn.Module):
    def __init__(self, l: int, d: int, n: int):
        '''
        l: number of layers
        d: feature dimension
        n: context length
        '''
        super(RGTransformer, self).__init__()
        self.n = n
        self.layers = nn.ModuleList([RGLayer(d, n)  for _ in range(l)])
        self.Cs = [layer.C for layer in self.layers] # preconditioning matrices

    def forward(self, Z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        '''
        Z: prompt matrix
        alpha: learning rate
        '''
        v = []
        for layer in self.layers:
            Z = layer.forward(Z, alpha)
            # negate it to align with the convention
            v.append(-Z[-1, -1].item())
        v = torch.tensor(v) 
        return v