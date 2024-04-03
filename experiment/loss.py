import torch


def mean_squared_td_error(w: torch.tensor,
                          Z: torch.tensor,
                          d: int,
                          n: int):
    '''
    w: weight vector (d, 1)
    Z: prompt (2d+1, n)
    d: feature dimension
    n: context length
    '''

    Phi = Z[:d, :n]
    Phi_prime = Z[d:2*d, :n]
    reward_vec = Z[-1, :n].reshape(1, n)

    v_vec = w.t() @ Phi
    # use detach() to prevent backpropagation through w here
    v_prime_vec = w.t().detach() @ Phi_prime
    tde_vec = reward_vec + v_prime_vec - v_vec
    mstde = torch.mean(tde_vec**2, dim=1)
    return mstde

def weight_error_norm(w1: torch.tensor,
                      w2: torch.tensor):
    '''
    w1: weight vector (d, 1)
    w2: weight vector (d, 1)
    '''
    return torch.norm(w1 - w2)

def value_error(v1: torch.tensor,
                v2: torch.tensor):
    '''
    v1: value vector (1, 1)
    v2: value vector (1, 1)
    '''
    return torch.abs(v1 - v2)