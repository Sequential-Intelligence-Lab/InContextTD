import os

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.rcParams['text.usetex'] = True


def load_data():
    avg_reward = np.load('logs/theory/avg_reward_td.npy')
    td = np.load('logs/theory/discounted_td.npy')
    td_lambda = np.load('logs/theory/discounted_td_lambda.npy')
    rg = np.load('logs/theory/residual_gradient.npy')
    assert avg_reward.shape == td.shape == td_lambda.shape == rg.shape
    data = {'TD(0)': td,
            'Residual Gradient': rg,
            'TD($\lambda$)': td_lambda,
            'Avg Reward TD': avg_reward}
    xs = np.arange(1, len(avg_reward[0])+1)
    return xs, data


def plot_error(save_dir: str):
    plt.style.use('science')
    xs, data = load_data()
    styles = ['solid', 'dashdot', 'dotted', 'dashed']
    fig = plt.figure(figsize=(10, 5))
    for (key, value), style in zip(data.items(), styles):
        mean = np.mean(value, axis=0)
        log_mean = np.log(mean)
        ste = np.std(value, axis=0)/np.sqrt(len(value))
        plt.plot(xs, log_mean, label=key, linestyle=style)
        plt.fill_between(xs, log_mean - ste, log_mean + ste, alpha=0.3)
    plt.xlabel('Layers', fontsize=20)
    plt.ylabel('$\log \left| -\left<\phi_n, w_l\\right> - y_l^{(n+1)} \\right|$',
               rotation=0, labelpad=100, fontsize=20)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'log_error.pdf'))
    plt.close(fig)