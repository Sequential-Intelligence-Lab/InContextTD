import os

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import torch
from tqdm import tqdm

from experiment.prompt import Feature, MRPPrompt
from experiment.utils import compute_msve, set_seed
from MRP.loop import Loop

if __name__ == '__main__':
    os.makedirs(os.path.join('logs', 'demo'), exist_ok=True)

    set_seed(42)

    d = 5
    l = 15
    min_s = 5
    max_s = 15
    gamma = 0.9
    n_mrps = 300
    alpha = 0.2
    context_lengths = list(range(1, 41, 2))

    all_msves = []  # (n_mrps, len(context_lengths))
    for _ in tqdm(range(n_mrps)):
        s = np.random.randint(min_s, max_s + 1)  # sample number of states
        thd = np.random.uniform(low=0.1, high=0.9)
        feature = Feature(d, s)  # new feature
        true_w = np.random.randn(d, 1)  # sample true weight
        mrp = Loop(s, gamma, threshold=thd, weight=true_w, Phi=feature.phi)
        msve_n = []
        for n in context_lengths:
            prompt = MRPPrompt(d, n, gamma, mrp, feature)
            prompt.reset()
            w = torch.zeros((d, 1))
            for _ in range(l):
                w, _ = prompt.td_update(w, lr=alpha)
            msve_n.append(compute_msve(feature.phi @ w.numpy(), 
                                       mrp.v, 
                                       mrp.steady_d))
        all_msves.append(msve_n)

    all_msves = np.array(all_msves)
    mean = np.mean(all_msves, axis=0)
    ste = np.std(all_msves, axis=0) / np.sqrt(n_mrps)

    plt.style.use(['science', 'bright', 'no-latex'])
    fig = plt.figure()
    plt.plot(context_lengths, mean)

    plt.fill_between(context_lengths,
                     np.clip(mean - ste, a_min=0, a_max=None),
                     mean + ste,
                     color='b', alpha=0.2)
    plt.xlabel('Context Length (t)')
    plt.ylabel('MSVE', rotation=0, labelpad=30)
    plt.grid(True)
    fig_path = os.path.join('logs', 'demo', 'msve_vs_context_length.pdf')
    plt.savefig(fig_path, dpi=300, format='pdf')
    plt.close(fig)
