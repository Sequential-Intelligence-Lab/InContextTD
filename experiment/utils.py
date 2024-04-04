import torch
import experiment.model as model

def stack_four(A: torch.Tensor, B: torch.Tensor,
               C: torch.Tensor, D: torch.Tensor):
    top = torch.cat([A, B], dim=1)
    bottom = torch.cat([C, D], dim=1)
    return torch.cat([top, bottom], dim=0)


def analytical_weight_update(w_tf: torch.Tensor,
                             Z: torch.Tensor,
                             d: int,
                             n: int,
                             C: torch.Tensor = None):
    '''
    w_tf: current transformer weight
    Z: context matrix
    d: feature dimension
    n: context length
    C: preconditioning matrix
    '''
    Phi = Z[:d, :n]
    Y = Z[-1, :n].reshape(n, 1)
    prod = Phi @ Y
    if C:
        prod = C @ prod

    return w_tf + 1/n * prod


def manual_weight_extraction(tf: model.LinearTransformer,
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
        Z_tf = tf(Z_p)
        weight.append(Z_tf[-1, -1])
    weight = torch.stack(weight, dim=0)
    return weight.reshape((d, 1))


if __name__ == '__main__':
    from experiment.prompt import Prompt
    from experiment.model import LinearTransformer
    d = 3
    n = 40
    l = 5
    pro = Prompt(d, n, 0.9)
    Z_0 = pro.z()
    tf = LinearTransformer(d, n, l, mode='auto')
    weight = manual_weight_extraction(tf, Z_0, d)
    print(weight)
