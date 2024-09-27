import datetime
import json
import os
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from experiment.model import HardLinearTransformer, LinearTransformer
from experiment.plotter import (load_data, plot_attention_params,
                                plot_mean_attn_params, plot_multiple_runs)
from experiment.prompt import MDPPromptGenerator
from experiment.utils import (compare_sensitivity, compute_mspbe, compute_msve,
                              first_order_comparison, set_seed,
                              solve_msve_weight, zero_order_comparison)
from MRP.mrp import MRP


if __name__ == '__main__':
    from plotter import (compute_weight_metrics, plot_error_data,
                         plot_weight_metrics, process_log)
    from utils import get_hardcoded_P, get_hardcoded_Q
    d = 5
    #n = 30
    l = 20
    s = 5
    gamma = 0.9
    sample_weight = True
    mode = 'auto'
    startTime = datetime.datetime.now()
    save_dir = os.path.join('../logs')
    data_dirs = []
    n_mdps = 50
    alpha = 0.5
    context_lengths = list(range(1, 101, 10))
    msve_dict = {n: [] for n in context_lengths}
    for n in context_lengths:
        pro_gen = MDPPromptGenerator(s, d, n, gamma)
        msves= []
        for i in range(n_mdps):
            pro_gen.reset_feat()  # reset feature
            pro_gen.reset_mdp(sample_weight=True)  # reset MDP
            prompt = pro_gen.get_prompt()  # get prompt object
            prompt.reset()
            mdp: MRP = prompt.mdp
            phi: np.ndarray = prompt.get_feature_mat().numpy()
            #print(phi.shape)
            steady_d: np.ndarray = mdp.steady_d
            true_v: np.ndarray = mdp.v
            w = torch.zeros((d, 1))
            for _ in range(l):
                w, _= prompt.td_update(w, lr = alpha)
            msves.append(compute_msve(phi@w.numpy(), true_v, steady_d))
        msve_dict[n].append(msves)

    # Calculate the average MSVE for each context length
    avg_msve = {n: np.mean(msve_dict[n]) for n in context_lengths}

    # Plot the average MSVE as a function of context length
    plt.figure(figsize=(10, 6))
    plt.plot(context_lengths, list(avg_msve.values()), marker='o')
    # Calculate the standard error of the mean for each context length
    std_err = {n: np.std(msve_dict[n]) / np.sqrt(len(msve_dict[n])) if len(msve_dict[n]) > 0 else 0 for n in context_lengths}

    # Plot the average MSVE with standard error as a shaded region
    avg_msve_values = list(avg_msve.values())
    std_err_values = list(std_err.values())
    #import pdb; pdb.set_trace()
    plt.fill_between(context_lengths, 
                     [avg - err for avg, err in zip(avg_msve_values, std_err_values)], 
                     [avg + err for avg, err in zip(avg_msve_values, std_err_values)], 
                     color='b', alpha=0.2)
    plt.xlabel('Context Length')
    plt.ylabel('Average MSVE')
    plt.title('Average MSVE vs Context Length')
    plt.grid(True)
    plt.savefig('avg_msve_vs_context_length.png')
    plt.show()