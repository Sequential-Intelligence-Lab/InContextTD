import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from experiment.model import LinearTransformer
from experiment.prompt import Prompt, Feature, MDP_Prompt
from experiment.utils import manual_weight_extraction, solve_mspbe, solve_msve
from experiment.loss import mean_squared_td_error, weight_error_norm, value_error
from torch_in_context_td import HC_Transformer
from experiment.boyan import BoyanChain

def train(d: int,
          s: int,
          n: int,
          l: int,
          gamma: float = 0.9,
          lmbd: float = 0.0,
          lr: float = 0.001,
          epochs: int = 50_000,
          log_interval: int = 200,
          mdp_train: bool = True):

    tf = LinearTransformer(d, n, l, lmbd, mode='sequential')
    opt = optim.Adam(tf.parameters(), lr=lr, weight_decay=1e-5)
    features = Feature(d,s)

    # Transformer with hardcoded weights according to our analytical TD update
    # whats the TD learning rate for our case?
    hc_tf = HC_Transformer(l, d, n)

    xs = [] # epochs
    mstdes = [] # mean squared td errors
    wes = [] # weight error norms
    ves = [] # value errors (absolute difference between true and learned tf predicted value)
    hc_ves = [] # value errors (absolute difference between true and hardcoded tf predicted value)
    hc_train_ves = [] # value errors (absolute difference between learned and hardcoded tf predicted value)
    for i in range(epochs): 
        #generate a new prompt
        if mdp_train:
            boyan_mdp = BoyanChain(s, gamma)
            pro =  MDP_Prompt(boyan_mdp, features, n, gamma)   # Markovian prompt based prompt from Boyan Chain
        else:
            pro = Prompt(d, n, gamma, noise=0.0) # IID promopt

        Z_0 = pro.z()
        phi_query = Z_0[:d, [n]]

        # extract the learned weights from the transformer
        w_tf = manual_weight_extraction(tf, Z_0, d)
        v_tf = w_tf.t() @ phi_query

        mstde = mean_squared_td_error(w_tf, Z_0, d, n)

        opt.zero_grad()
        total_loss = mstde
        total_loss.backward(retain_graph=True) # how does this backward work here if we extract w_tf manually? 
        opt.step()

        if i % log_interval == 0:
            # Compare the learned weight with true weight value predictions
            if mdp_train:
                w_msve, msve = solve_msve(boyan_mdp.P, features.phi, boyan_mdp.v)
                w_mspbe, mspbe = solve_mspbe(boyan_mdp.P, features.phi, boyan_mdp.r, boyan_mdp.gamma)

                true_w = torch.tensor(w_msve, dtype=torch.float32)
                true_v = true_w.t() @ phi_query
            else:
                #get the true value
                true_w = pro.w
                true_v = true_w.t() @ phi_query
            
            # 1. compute the hc_tf predicted valule function
            #v_out, _ = hc_tf.forward(Z_0)
            #v_tf_hc = v_out[-1] # TODO: we have some numerial instability issue here???

            # 2. compute the value function using l batch TD updates
            # TODO: implement td_update for the MDP_Prompt class
            #w_manual = torch.zeros((d, 1))
            #for _ in range(l):
            #    w_manual, v_manual = pro.td_update(w_manual) #no preconditioning

            xs.append(i)
            mstdes.append(mstde.item())
            wes.append(weight_error_norm(w_tf, true_w).item())
            ves.append(value_error(v_tf, true_v).item())
            #hc_ves.append(value_error(v_manual,true_v).item()) # compare VE btw hc_tf and the ground truth
            #hc_train_ves.append(value_error(v_manual, v_tf).item()) # compare prediction error between the learned tf with the hc_TD tf

            print('Epoch:', i)
            print('Transformer Learned Weight:\n', w_tf.detach().numpy())
            print('True Weight:\n', true_w.numpy())

    xs.append(epochs)
    mstdes.append(mstde.item())
    wes.append(weight_error_norm(w_tf, true_w).item())
    ves.append(value_error(v_tf, true_v).item())
    #hc_ves.append(value_error(true_v, v_manual).item())
    #hc_train_ves.append(value_error(v_tf, v_manual).item())

    print('Transformer Learned Weight:\n', w_tf.detach().numpy())
    plt.figure()
    plt.title('Learned Transformer vs Ground Truth')
    plt.yscale('log')
    plt.plot(xs, mstdes, label='Mean Squared TD Error')
    plt.plot(xs, wes, label='Weight(w) Error Norm')
    plt.plot(xs, ves, label='Absolute Value Error (vs True Value)')
    plt.plot(xs, hc_train_ves, label='AVE (Learned TF vs HC)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Learned Transformer vs Hardcoded Transformer')
    plt.yscale('log')
    plt.plot(xs, hc_ves, label='AVE (HC vs Ground Truth)')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(2)
    np.random.seed(2)
    d = 5
    n = 200
    l = 6
    s= n # number of states equal to the context length
    train(d, s, n, l, lmbd=0.0, epochs=10000 , mdp_train = True)
