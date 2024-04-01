import torch


def stack_four(A: torch.Tensor, B: torch.Tensor,
               C: torch.Tensor, D: torch.Tensor):
    top = torch.cat([A, B], dim=1)
    bottom = torch.cat([C, D], dim=1)
    return torch.cat([top, bottom], dim=0)


def attn_weight_update(w_tf: torch.Tensor,
                       Z: torch.Tensor,
                       d: int,
                       n: int,
                       C: torch.Tensor = None):

    assert Z.shape == (2*d+1, n+1)
    Phi = Z[:d, :-1]
    Y = Z[-1, :-1].reshape(n, 1)
    prod = Phi @ Y
    if C:
        prod = C @ prod

    return w_tf + 1/n * prod


if __name__ == '__main__':
    from prompt import Prompt
    d = 4
    n = 10
    pro = Prompt(d, n, 0.9)
    Z = pro.z()
    w = torch.zeros((d, 1))
    print(pro.td_update(w))
    print(attn_weight_update(w, Z, d, n))
