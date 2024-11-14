from experiment.model import RNN
import torch
if __name__ == '__main__':
    d = 3
    n = 10
    l = 4
    s = 5
    context = torch.randn(2*d+1, n)
    phi = torch.randn(s, d)
    model = RNN(d, l)
    v = model.fit_value_func(context, phi)
    print(v)

