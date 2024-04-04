import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from experiment.model import LinearTransformer
from experiment.prompt import Prompt
from experiment.utils import manual_weight_extraction
from experiment.loss import mean_squared_td_error, weight_error_norm, value_error


def train(d: int,
          n: int,
          l: int,
          gamma: float = 0.9,
          lmbd: float = 0.0,
          lr: float = 0.0003,
          epochs: int = 10_000,
          log_interval: int = 100):

    tf = LinearTransformer(d, n, l, lmbd, mode='sequential')
    opt = optim.Adam(tf.parameters(), lr=lr, weight_decay=1e-5)

    xs = []
    mstdes = []
    wes = []
    ves = []
    for i in range(epochs):
        pro = Prompt(d, n, gamma, noise=0.1)
        Z_0 = pro.z()
        phi_query = Z_0[:d, [n]]

        true_w = pro.w
        true_v = true_w.t() @ phi_query

        w_tf = manual_weight_extraction(tf, Z_0, d)
        v_tf = w_tf.t() @ phi_query

        mstde = mean_squared_td_error(w_tf, Z_0, d, n)

        opt.zero_grad()
        total_loss = mstde
        total_loss.backward(retain_graph=True)
        opt.step()

        if i % log_interval == 0:
            xs.append(i)
            mstdes.append(mstde.item())
            wes.append(weight_error_norm(w_tf, true_w).item())
            ves.append(value_error(v_tf, true_v).item())

            print('Epoch:', i)
            print('Transformer Learned Weight:\n', w_tf.detach().numpy())
            print('True Weight:\n', true_w.numpy())

    xs.append(epochs)
    mstdes.append(mstde.item())
    wes.append(weight_error_norm(w_tf, true_w).item())
    ves.append(value_error(v_tf, true_v).item())

    print('Transformer Learned Weight:\n', w_tf.detach().numpy())
    print('True Weight:\n', true_w.numpy())

    plt.plot(xs, np.log(mstdes), label='Mean Squared TD Error')
    plt.plot(xs, np.log(wes), label='Weight Error Norm')
    plt.plot(xs, np.log(ves), label='Absolute Value Error')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(2)
    np.random.seed(2)
    d = 5
    n = 400
    l = 8
    train(d, n, l, lmbd=0.0, epochs=10000)
