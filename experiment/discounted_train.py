import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from experiment.loss import (mean_squared_td_error, self_consistency_loss,
                             value_error, weight_error_norm)
from experiment.model import LinearTransformer
from experiment.prompt import Feature, MDP_Prompt, Prompt
from experiment.utils import (compute_mspbe, compute_msve,
                              manual_weight_extraction, solve_mspbe_weight,
                              solve_msve_weight)
# from torch_in_context_td import HC_Transformer
from MRP.boyan import BoyanChain


def compute_tf_msve(tf: LinearTransformer,
                    context: torch.tensor,
                    X: np.ndarray,
                    true_v: np.ndarray,
                    steady_d: np.ndarray) -> float:
    d = X.shape[1]
    X = torch.from_numpy(X)
    tf_v = []
    with torch.no_grad():
        for feature in X:
            feature_col = torch.zeros((2*d+1, 1))
            feature_col[:d, 0] = feature
            Z_p = torch.cat([context, feature_col], dim=1)
            Z_tf = tf(Z_p)
            tf_v.append(-Z_tf[-1, -1].item())
    tf_v = np.array(tf_v).reshape(-1, 1)
    error = tf_v - true_v
    msve = steady_d.dot(error**2)
    return msve.item()


def train(d: int,
          s: int,
          n: int,
          l: int,
          gamma: float = 0.9,
          lmbd: float = 0.0,
          sample_weight: bool = True,
          lr: float = 0.001,
          weight_decay=1e-6,
          steps: int = 50_000,
          log_interval: int = 100):
    '''
    d: feature dimension
    s: number of states
    n: context length
    l: number of layers
    gamma: discount factor
    lmbd: eligibility trace decay
    sample_weight: sample a random true weight vector
    lr: learning rate
    weight_decay: regularization
    steps: number of training steps
    log_interval: logging interval
    '''

    tf = LinearTransformer(d, n, l, lmbd, mode='auto')
    opt = optim.Adam(tf.parameters(), lr=lr, weight_decay=weight_decay)
    features = Feature(d, s)

    writer = SummaryWriter(log_dir='./logs')

    for i in range(steps):
        # generate a new prompt
        if sample_weight:
            w_true = np.random.randn(d, 1).astype(np.float32)
            boyan_mdp = BoyanChain(
                n_states=s, gamma=gamma, weight=w_true, X=features.phi)
        else:
            boyan_mdp = BoyanChain(n_states=s, gamma=gamma)

        # Markovian prompt based prompt from Boyan Chain
        pro = MDP_Prompt(boyan_mdp, features, n, gamma)

        Z_0 = pro.z()
        phi_query = Z_0[:d, [n]]

        # extract the learned weights from the transformer
        w_tf = manual_weight_extraction(tf, Z_0, d)

        mstde = mean_squared_td_error(w_tf, Z_0, d, n)
        sc_loss = self_consistency_loss(w_tf, phi_query, Z_0)

        opt.zero_grad()
        total_loss = mstde + sc_loss
        total_loss.backward()
        opt.step()

        if i % log_interval == 0:
            loss_dict = {'Mean Square TD Error': mstde.item(),
                         'Self-Consistency Loss': sc_loss.item()}
            writer.add_scalars('Loss', loss_dict, i)

            w_msve = solve_msve_weight(
                boyan_mdp.steady_d, features.phi, boyan_mdp.v)
            w_msve_tensor = torch.from_numpy(w_msve)
            w_mspbe = solve_mspbe_weight(
                boyan_mdp.steady_d, boyan_mdp.P, features.phi, boyan_mdp.r, gamma)
            w_mspbe_tensor = torch.from_numpy(w_mspbe)
            weight_error_norm_dict = {'MSVE Weight Error Norm': weight_error_norm(w_tf.detach(), w_msve_tensor).item(),
                                      'MSPBE Weight Error Norm': weight_error_norm(w_tf.detach(), w_mspbe_tensor).item()}
            writer.add_scalars('Weight Error Norm', weight_error_norm_dict, i)

            true_msve = compute_msve(
                w_msve, boyan_mdp.steady_d, features.phi, boyan_mdp.v)
            tf_msve = compute_tf_msve(
                tf, pro.context(), features.phi, boyan_mdp.v, boyan_mdp.steady_d)
            msve_dict = {'True MSVE': true_msve, 'Transformer MSVE': tf_msve}
            writer.add_scalars('MSVE', msve_dict, i)

            print('Step:', i)
            print('Transformer Learned Weight:\n', w_tf.detach().numpy())
            print('MSVE Weight:\n', w_msve)
            print('MSPBE Weight:\n', w_mspbe)

    loss_dict = {'Mean Square TD Error': mstde.item(),
                 'Self-Consistency Loss': sc_loss.item()}
    writer.add_scalars('Loss', loss_dict, steps)

    w_msve = solve_msve_weight(boyan_mdp.steady_d, features.phi, boyan_mdp.v)
    w_msve_tensor = torch.from_numpy(w_msve)
    w_mspbe = solve_mspbe_weight(
        boyan_mdp.steady_d, boyan_mdp.P, features.phi, boyan_mdp.r, gamma)
    w_mspbe_tensor = torch.from_numpy(w_mspbe)
    weight_error_norm_dict = {'MSVE Weight Error Norm': weight_error_norm(w_tf.detach(), w_msve_tensor).item(),
                              'MSPBE Weight Error Norm': weight_error_norm(w_tf.detach(), w_mspbe_tensor).item()}
    writer.add_scalars('Weight Error Norm', weight_error_norm_dict, steps)

    true_msve = compute_msve(w_msve, boyan_mdp.steady_d, features.phi, boyan_mdp.v)
    tf_msve = compute_tf_msve(tf, pro.context(), features.phi, boyan_mdp.v, boyan_mdp.steady_d)
    msve_dict = {'True MSVE': true_msve, 'Transformer MSVE': tf_msve}
    writer.add_scalars('MSVE', msve_dict, steps)

    print('Step:', steps)
    print('Transformer Learned Weight:\n', w_tf.detach().numpy())
    print('MSVE Weight:\n', w_msve)
    print('MSPBE Weight:\n', w_mspbe)
    writer.flush()
    writer.close()


if __name__ == '__main__':
    torch.manual_seed(2)
    np.random.seed(2)
    d = 4
    n = 200
    l = 4
    s = int(n/10)  # number of states equal to the context length
    train(d, s, n, l, lmbd=0.0, sample_weight=False, steps=10_000)
