import os
from argparse import ArgumentParser, Namespace

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch
from tqdm import tqdm

from experiment.prompt import Feature, MRPPrompt
from rnn.train import train_rnn
from rnn.model import RNN
from MRP.loop import Loop
from MRP.boyan import BoyanChain
from utils import compute_msve, set_seed

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--dim_feature', type=int,
                        help='feature dimension', default=4)
    parser.add_argument('-l', '--num_layers', type=int,
                        help='number of transformer layers', default=3)
    parser.add_argument('-s', '--state_num', type=int,
                        help='number of states in the MRP for training', default=10)
    parser.add_argument('-smin', '--min_state_num', type=int,
                        help='minimum possible number of states', default=5)
    parser.add_argument('-smax', '--max_state_num', type=int,
                        help='maximum possible number of states', default=15)
    parser.add_argument('--gamma', type=float,
                        help='discount factor', default=0.9)
    parser.add_argument('--lr', type=float,
                        help='TD learning rate', default=0.001)
    parser.add_argument('-a', '--activation', type=str,
                        help='activation function for the transformer', default='tanh')
    parser.add_argument('--n_mrps_train', type=int,
                        help='total number of MRPs to run', default=4000)
    parser.add_argument('--n_mrps', type=int,
                        help='total number of MRPs to run', default=500)
    parser.add_argument('-n', '--ctxt_len', type=int,
                        help='context length for training', default=30)
    parser.add_argument('-nmin', '--min_ctxt_len', type=int,
                        help='minimum context length', default=1)
    parser.add_argument('-nmax', '--max_ctxt_len', type=int,
                        help='maximum context length', default=100)
    parser.add_argument('--ctxt_step', type=int,
                        help='context length step', default=2)
    parser.add_argument('--seed', type=int,
                        help='random seed', default=42)
    parser.add_argument('--save_dir', type=str,
                        help='directory to save demo result', default='logs')

    args: Namespace = parser.parse_args()
    d = args.dim_feature
    l = args.num_layers
    s = args.state_num
    n = args.ctxt_len
    min_s = args.min_state_num
    max_s = args.max_state_num
    gamma = args.gamma
    n_mrps = args.n_mrps
    context_lengths = list(range(args.min_ctxt_len,
                                 args.max_ctxt_len+1,
                                 args.ctxt_step))

    save_path = os.path.join(args.save_dir, 'demo_rnn')
    ckpt_path = os.path.join(save_path, 'ckpt')
    plot_path = os.path.join(save_path, 'plots')

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(plot_path, exist_ok=True)

    set_seed(args.seed)

    # train_rnn(d=d, s=s, n=n, l=l, save_dir=save_path, n_mrps=args.n_mrps_train, lr=args.lr)

    data = np.load(os.path.join(save_path, 'data.npz'))
    plt.style.use(['science', 'bright', 'no-latex'])
    fig = plt.figure()
    plt.plot(data['xs'], data['msve'])
    plt.xlabel('Number of MRPs')
    plt.ylabel('MSVE', rotation=0, labelpad=30)
    plt.grid(True)
    fig_path = os.path.join(plot_path, 'msve_vs_mrp.pdf')
    plt.savefig(fig_path, dpi=300, format='pdf')
    plt.close(fig)

    model = RNN(d, l, activation=args.activation)
    params = torch.load(os.path.join(ckpt_path, 'params.pt'), weights_only=True)
    model.load_state_dict(params)


    all_msves = []  # (n_mrps, len(context_lengths))
    for _ in tqdm(range(n_mrps)):
        s = np.random.randint(min_s, max_s + 1)  # sample number of states
        thd = np.random.uniform(low=0.1, high=0.9)
        feature = Feature(d, s)  # new feature
        true_w = np.random.randn(d, 1)  # sample true weight
        mrp = Loop(s, gamma, threshold=thd, weight=true_w, phi=feature.phi)
        # mrp = BoyanChain(s, weight=true_w, X=feature.phi)
        msve_n = []
        for n in context_lengths:
            prompt = MRPPrompt(d, n, gamma, mrp, feature)
            prompt.reset()
            ctxt = prompt.context()
            v_rnn = model.fit_value_func(ctxt, torch.from_numpy(feature.phi)).detach().numpy()
            msve = compute_msve(v_rnn, mrp.v, mrp.steady_d)
            msve_n.append(msve)
        all_msves.append(msve_n)

    all_msves = np.array(all_msves)
    mean = np.mean(all_msves, axis=0)
    ste = np.std(all_msves, axis=0) / np.sqrt(n_mrps)

    fig = plt.figure()
    plt.plot(context_lengths, mean)

    plt.fill_between(context_lengths,
                     np.clip(mean - ste, a_min=0, a_max=None),
                     mean + ste,
                     color='b', alpha=0.2)
    plt.xlabel('Context Length (t)')
    plt.ylabel('MSVE', rotation=0, labelpad=30)
    plt.grid(True)
    fig_path = os.path.join(plot_path, 'msve_vs_context_length.pdf')
    plt.savefig(fig_path, dpi=300, format='pdf')
    plt.close(fig)
