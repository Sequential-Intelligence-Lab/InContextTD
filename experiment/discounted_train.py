import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from experiment.model import LinearTransformer
from experiment.prompt import Prompt
from experiment.utils import manual_weight_extraction
from experiment.loss import mean_squared_td_error, weight_error_norm, value_error
from torch_in_context_td import HC_Transformer


def train(d: int,
          n: int,
          l: int,
          gamma: float = 0.9,
          lr: float = 0.001,
          epochs: int = 50_000,
          log_interval: int = 200):

    tf = LinearTransformer(d, n, l, mode='auto')
    opt = optim.Adam(tf.parameters(), lr=lr, weight_decay=1e-5)

    # Transformer with hardcoded weights according to our analytical TD update
    # whats the TD learning rate for our case?
    hc_tf = HC_Transformer(l, d, n)

    xs = [] # epochs
    mstdes = [] # mean squared td errors
    wes = [] # weight error norms
    ves = [] # value errors (absolute difference between true and learned tf predicted value)
    hc_ves = [] # value errors (absolute difference between learned and hardcoded tf predicted value)
    for i in range(epochs): 
        #generate a new prompt
        pro = Prompt(d, n, gamma, noise=0.0)
        Z_0 = pro.z()
        phi_query = Z_0[:d, [n]]

        #get the true value
        true_w = pro.w
        true_v = true_w.t() @ phi_query

        # extract the learned weights from the transformer
        w_tf = manual_weight_extraction(tf, Z_0, d)
        v_tf = w_tf.t() @ phi_query

        mstde = mean_squared_td_error(w_tf, Z_0, d, n)

        opt.zero_grad()
        total_loss = mstde
        total_loss.backward(retain_graph=True) # how does this backward work here if we extract w_tf manually? 
        opt.step()

        # compare the learned weight prediction with the hardcoded TD weight prediction
        w_tf_hc = manual_weight_extraction(hc_tf, Z_0, d)
        v_tf_hc = w_tf_hc.t() @ phi_query
        v_tf_hc2, _ = hc_tf.forward(Z_0)
        #import pdb; pdb.set_trace()
        #assert round(v_tf_hc.item(),2) == round(v_tf_hc2[-1],2) # sanity check on the hardcoded transformer

        if i % log_interval == 0:
            xs.append(i)
            mstdes.append(mstde.item())
            wes.append(weight_error_norm(w_tf, true_w).item())
            ves.append(value_error(v_tf, true_v).item())
            hc_ves.append(value_error(v_tf, v_tf_hc).item()) # compare prediction error between the learned tf with the hardcoded TD tf

            print('Epoch:', i)
            print('Transformer Learned Weight:\n', w_tf.detach().numpy())
            print('True Weight:\n', true_w.numpy())

    xs.append(epochs)
    mstdes.append(mstde.item())
    wes.append(weight_error_norm(w_tf, true_w).item())
    ves.append(value_error(v_tf, true_v).item())
    hc_ves.append(value_error(v_tf, v_tf_hc).item())

    print('Transformer Learned Weight:\n', w_tf.detach().numpy())
    plt.title('Learned Transformer vs Ground Truth')
    plt.yscale('log')
    plt.plot(xs, mstdes, label='Mean Squared TD Error')
    plt.plot(xs, wes, label='Weight(w) Error Norm')
    plt.plot(xs, ves, label='Absolute Value Error (vs True Value)')
    plt.plot(xs, hc_ves, label='Absolute Value Error (vs HC TF)')
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(5)
    np.random.seed(5)
    d = 4
    n = 250
    l = 6
    train(d, n, l)

    # after this we need to evaluate the performance of our meta trained transformer

    # Weight Error


# so some things to check are the metrics for the weight error with the TD learned weights not the ground truth
# we also need to check P and Q for the transformer using the implicit linear weight difference or cosine similarity of the sensitvities 