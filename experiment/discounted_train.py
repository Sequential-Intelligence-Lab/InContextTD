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
          log_interval: int = 250):

    tf = LinearTransformer(d, n, l, lmbd, mode='sequential')
    opt = optim.Adam(tf.parameters(), lr=lr, weight_decay=1e-5)
    features = Feature(d,s)

    # Transformer with hardcoded weights according to our analytical TD update
    # whats the TD learning rate for our case?
    hc_tf = HC_Transformer(l, d, n)

    xs = [] # epochs
    mstdes = [] # mean squared td errors
    msves = [] # mean squared value errors
    mspbes = [] # mean squared projected bellman errors
    ves = [] # value errors (absolute difference between true and learned tf predicted value)
    w_msve_error = [] # weight error norms between learned tf and the tf that minimizes MSVE
    w_mspbe_error = [] # weight error norms between learned tf and the tf that minimizes MSPBE

    for i in range(epochs): 
        #generate a new prompt
        boyan_mdp = BoyanChain(s, gamma, noise=0.0)
        pro =  MDP_Prompt(boyan_mdp, features, n, gamma)   # Markovian prompt based prompt from Boyan Chain

        Z_0 = pro.z()
        phi_query = Z_0[:d, [n]]

        # extract the learned weights from the transformer
        w_tf = manual_weight_extraction(tf, Z_0, d)

        mstde = mean_squared_td_error(w_tf, Z_0, d, n)

        opt.zero_grad()
        total_loss = mstde
        total_loss.backward(retain_graph=True) # how does this backward work here if we extract w_tf manually? 
        opt.step()

        if i % log_interval == 0:
            # Compare the learned weight with true weight value predictions
            w_msve, msve = solve_msve(boyan_mdp.P, features.phi, boyan_mdp.v)
            w_mspbe, mspbe = solve_mspbe(boyan_mdp.P, features.phi, boyan_mdp.r, boyan_mdp.gamma)

            w_msve = torch.tensor(w_msve, dtype=torch.float32)
            w_mspbe = torch.tensor(w_mspbe, dtype=torch.float32)
            
            # TODO: Compare with Batch TD
            # 1. compute the hc_tf predicted valule function
            #v_out, _ = hc_tf.forward(Z_0)
            #v_tf_hc = v_out[-1] 

            # 2. compute the value function using l batch TD updates
            # TODO: implement td_update for the MDP_Prompt class
            #w_manual = torch.zeros((d, 1))
            #for _ in range(l):
            #    w_manual, v_manual = pro.td_update(w_manual) #no preconditioning

            xs.append(i)
            mstdes.append(mstde.item())
            msves.append(msve)
            mspbes.append(mspbe)
            w_msve_error.append(weight_error_norm(w_tf, w_msve).item())
            w_mspbe_error.append(weight_error_norm(w_tf, w_mspbe).item())
            #ves.append(value_error(v_tf, true_v).item())
            #hc_ves.append(value_error(v_manual,true_v).item()) # compare VE btw hc_tf and the ground truth
            #hc_train_ves.append(value_error(v_manual, v_tf).item()) # compare prediction error between the learned tf with the hc_TD tf

            print('Epoch:', i)
            print('Transformer Learned Weight:\n', w_tf.detach().numpy())
            print('MVSE Minimizer:\n', w_msve.numpy())
            print('MSPBE Minimizer:\n', w_mspbe.numpy())

    xs.append(epochs)
    mstdes.append(mstde.item())
    msves.append(msve)
    mspbes.append(mspbe)
    w_msve_error.append(weight_error_norm(w_tf, w_msve).item())
    w_mspbe_error.append(weight_error_norm(w_tf, w_mspbe).item())
    #ves.append(value_error(v_tf, true_v).item())
    #hc_ves.append(value_error(true_v, v_manual).item())
    #hc_train_ves.append(value_error(v_tf, v_manual).item())

    print('Transformer Learned Weight:\n', w_tf.detach().numpy())
    plt.figure()
    plt.title('Learned Transformer Weights vs MSVE and MSPBE Minimizing Weights')
    #plt.yscale('log')
    plt.plot(xs, w_msve_error, label='Weight(w) Error Norm (MSVE)')
    plt.plot(xs, w_mspbe_error, label='Weight(w) Error Norm (MSPBE)')
    #plt.plot(xs, ves, label='Absolute Value Error (vs True Value)')
    #plt.plot(xs, hc_train_ves, label='AVE (Learned TF vs HC)')
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    plt.title('Learned Transformer Performance')
    #plt.yscale('log')
    plt.plot(xs, mstdes, label='Mean Squared TD Error')
    #plt.plot(xs, msves, label='Mean Squared Value Error')
    #plt.plot(xs, mspbes, label='Mean Squared Projected Bellman Error')
    plt.grid()
    plt.legend()
    plt.show()

if __name__ == '__main__':
    torch.manual_seed(2)
    np.random.seed(2)
    d = 5
    n = 300
    l = 6
    s= int(n/4) # number of states equal to the context length
    train(d, s, n, l, lmbd=0.0, epochs=100_000)
