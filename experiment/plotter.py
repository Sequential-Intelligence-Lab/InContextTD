import json
import os
from typing import List, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import seaborn as sns

from utils import check_params, compare_P, compare_Q, scale


def load_data(data_dir: str) -> Tuple[dict, dict]:
    """
    Load data from specified directory and return the relevant metrics.
    """
    with np.load(os.path.join(data_dir, 'data.npz')) as data, \
            open(os.path.join(data_dir, 'params.json'), 'r') as params_file:
        log = {key: data[key] for key in data}
        hyperparams = json.load(params_file)
    return log, hyperparams


def process_log(log: dict) -> Tuple[np.ndarray, dict, dict]:
    '''
    parse the log dictionary
    return the x ticks, error log, and aligned attention parameters
    '''
    Ps, Qs = log['P'], log['Q']

    aligned_Ps, aligned_Qs = align_matrix_sign(Ps, Qs)

    alphas = np.array(log['alpha'])
    attn_params = {'P': aligned_Ps, 'Q': aligned_Qs, 'alpha': alphas}

    error_log = {}
    for key in ('implicit_weight_sim',
                'sensitivity cos sim',
                'v_tf v_td msve'):
        if key in log:
            error_log[key] = np.expand_dims(log[key], axis=0)

    return log['xs'], error_log, attn_params


def _batch_runs(logs: List[dict]) -> dict:
    '''
    batch the logs from multiple runs
    '''
    keys = logs[0].keys()
    batched_logs = {key: np.vstack([log[key] for log in logs]) for key in keys}
    return batched_logs

def plot_attn_params(data_dirs: List[str],
                          save_dir: str,
                          log_step: int = -1) -> None:
    '''
    data_dirs: list of directories containing the data
    save_dir: directory to save the plots
    log_step: time step to visualize
    '''

    Ps = []
    Qs = []
    alphas = []
    for data_dir in data_dirs:
        log, hypers = load_data(data_dir)
        xs, _, attn_params = process_log(log)
        Ps.append(attn_params['P'][log_step])  # shape (l, 2d+1, 2d+1)
        Qs.append(attn_params['Q'][log_step])  # shape (l, 2d+1, 2d+1)
        alphas.append(attn_params['alpha'])

    step = xs[log_step]
    # mean over seeds
    mean_Ps = np.mean(Ps, axis=0)
    mean_Qs = np.mean(Qs, axis=0)

    for l, (P, Q) in enumerate(zip(mean_Ps, mean_Qs)):  # shape (2d+1, 2d+1)
        P = scale(P)
        Q = scale(Q)
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
        cax1 = axs[0].matshow(P, vmin=-1, vmax=1)
        if log_step == -1:
            if hypers['mode'] == 'sequential':
                axs[0].set_title(
                    f'Final $P_{l}$', fontsize=26)
                axs[1].set_title(
                    f'Final $Q_{l}$', fontsize=26)
            else:
                axs[0].set_title(f'Final $P$', fontsize=26)
                axs[1].set_title(f'Final $Q$', fontsize=26)
        else:
            axs[0].set_title(f'Mean $P_{l}$ at MRP {step}', fontsize=26)
            axs[1].set_title(f'Mean $Q_{l}$ at MRP {step}', fontsize=26)
        axs[1].matshow(Q, vmin=-1, vmax=1)
        fig.colorbar(cax1, ax=axs, orientation='vertical')
        axs[0].tick_params(axis='both', which='both',
                           bottom=False, top=False,
                           left=False, right=False)
        axs[1].tick_params(axis='both', which='both',
                           bottom=False, top=False,
                           left=False, right=False)
        save_path = os.path.join(save_dir, f'PQ_mean_{l}_{step}.pdf')
        plt.savefig(save_path, dpi=300, format="pdf")
        plt.close(fig)

def plot_error_data(data_dirs: str,
                    save_dir: str) -> None:
    '''
    plot the error data from validation
    xs: x-axis values
    error_log: dictionary containing the error data
    save_dir: directory to save the plots
    '''
    
    error_log_lst = []
    for data_dir in data_dirs: # list of directories of the seeds you want to plot
        log, hyperparams = load_data(data_dir)
        xs, error_log, _ = process_log(log)
        error_log_lst.append(error_log)

    batched_error_log = _batch_runs(error_log_lst)  # shape (num_seeds, T, l)
    plt.style.use(['science', 'vibrant', 'no-latex'])

    num_seeds = batched_error_log['implicit_weight_sim'].shape[0]

    # Value function similarity
    metrics = ['v_tf v_td msve', 'implicit_weight_sim', 'sensitivity cos sim']
    means, stdes = {}, {}
    
    for metric in metrics:
        means[metric] = np.mean(batched_error_log[metric], axis=0)
        stdes[metric] = np.std(batched_error_log[metric], axis=0) / np.sqrt(num_seeds)
    
    mean_vf_sim = means['v_tf v_td msve']
    stde_vf_sim = stdes['v_tf v_td msve']
    mean_iws = means['implicit_weight_sim']
    stde_iws = stdes['implicit_weight_sim']
    mean_sensitivity_cos_sim = means['sensitivity cos sim']
    stde_sensitivity_cos_sim = stdes['sensitivity cos sim']

    #plt.figure()
    fig, ax1 = plt.subplots()
    plt.title("Learned TF and Batch TD Comparison")
    ax1.set_xlabel('# MRPs')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_ylim(0, 1.1)
    plt.minorticks_off()
    b, = ax1.plot(xs, mean_iws, label='IWS',
                  color=sns.color_palette()[0])
    ax1.fill_between(xs, mean_iws - stde_iws,
                     mean_iws + stde_iws, lw=0, alpha=0.2, color=sns.color_palette()[0])
    a, = ax1.plot(xs, mean_sensitivity_cos_sim,
                  label='SS', color=sns.color_palette()[5], linestyle='dashed')
    ax1.fill_between(xs, mean_sensitivity_cos_sim - stde_sensitivity_cos_sim,
                     mean_sensitivity_cos_sim + stde_sensitivity_cos_sim, lw=0, alpha=0.3, color=sns.color_palette()[5])
    ax2 = ax1.twinx()
    plt.minorticks_off()
    ax2.set_ylabel('Value Difference')
    if hyperparams['sample_weight']:
        ax2.set_ylim(0, 3.0)
    else:
        ax2.set_ylim(0, 0.3)
    c, = ax2.plot(xs, mean_vf_sim, label='VD', color=sns.color_palette()[1])
    ax2.fill_between(xs, mean_vf_sim - stde_vf_sim,
                     mean_vf_sim + stde_vf_sim, lw=0, alpha=0.3, color=sns.color_palette()[1])
    p = [a, b, c]
    if hyperparams['linear']:
        ax2.legend(p, [p_.get_label() for p_ in p], frameon=True, framealpha=0.8, loc='center right').set_alpha(0.5)
    else:
        ax2.legend(p, [p_.get_label() for p_ in p], frameon=True, framealpha=0.8, loc='upper left').set_alpha(0.5)
    plt.savefig(os.path.join(save_dir, 'batch_td_comparison.pdf'), dpi=300, format='pdf')
    plt.close()



def generate_attention_params_gif(xs: dict,
                                  l: int,
                                  params: dict,
                                  save_dir: str) -> None:
    '''
    generate a gif of the attention parameters for each layer
    xs: x-axis values
    l: number of layers
    params: attention parameters
    save_dir: directory to save the gif
    '''
    gif_dir = os.path.join(save_dir, 'attention_params_gif')
    if not os.path.exists(gif_dir):
        os.makedirs(gif_dir)

    # list of lists to store the images for each layer
    gif_list = [[] for _ in range(l)]
    for step in range(len(xs)):
        paths = plot_attn_params(xs, params, gif_dir, step, False)
        for i, path in enumerate(paths):
            gif_list[i].append(imageio.imread(path))
            os.remove(path)

    for i, gif in enumerate(gif_list):
        imageio.mimwrite(os.path.join(gif_dir, f'PQ_{i+1}.gif'), gif, fps=2)

    # remove the empty directory containing the temporary images
    os.rmdir(os.path.join(gif_dir, 'attention_params_plots'))


def plot_weight_metrics(data_dirs: str,
                        save_dir: str) -> None:
    '''
    plot the metrics for P and Q
    data_dirs: list of directories containing the data
    save_dir: directory to save the plots
    '''

    P_metrics_lst = []
    Q_metrics_lst = []
    log, hyperparams_0 = load_data(data_dirs[0])
    d = hyperparams_0['d']
    l = hyperparams_0['l'] if hyperparams_0['mode'] == 'sequential' else 1
    is_linear = hyperparams_0['linear']

    # Load data from directories
    for data_dir in data_dirs:
        log, params = load_data(data_dir)
        check_params(params, hyperparams_0)
        xs, _, attn_params = process_log(log)
        if is_linear:
            P_metrics, Q_metrics = compute_weight_metrics(attn_params, d)
            P_metrics_lst.append(P_metrics)
            Q_metrics_lst.append(Q_metrics)

    if is_linear:
        batched_Q_metrics = _batch_runs(Q_metrics_lst)
        batched_P_metrics = _batch_runs(P_metrics_lst)


    plt.style.use(['science', 'bright', 'no-latex'])
    # same layer, different metrics

    ### P metrics ###
    for i in range(l):
        fig = plt.figure()
        for key, metric in batched_P_metrics.items():
            plt.title(f'$P_{i}$ Metrics', fontsize=20)
            if key == 'bottom_right':
                label = (r'$P_{%s}[-1, -1]$' % (str(i)))
            elif key == 'avg_abs_all_others':
                label = 'Avg Abs Others'
            mean_metric = np.mean(metric, axis=0)  # shape (T, l)
            std_metric = np.std(metric, axis=0)
            stde_metric = std_metric / np.sqrt(metric.shape[0])
            assert mean_metric.shape == stde_metric.shape
            plt.plot(xs, mean_metric[:, i], label=label)
            plt.fill_between(xs, mean_metric[:, i] - stde_metric[:, i],
                            mean_metric[:, i] + stde_metric[:, i], alpha=0.3)
        plt.xlabel('# MRPs')
        plt.ylim(0,2)
        plt.minorticks_off()
        plt.legend(frameon=True, framealpha=0.8,
                    fontsize='medium').set_alpha(0.5) 
        plt.savefig(os.path.join(save_dir,
                    f'P_metrics_{i}.pdf'), dpi=300, format='pdf')
        plt.close(fig)


    ### Q metrics ###

    # same layer, different metrics
    for i in range(l):
        fig = plt.figure()
        for key, metric in batched_Q_metrics.items():
            # for the standard setting in the main paper, don't plot the transformer title
            plt.title(f'$Q_{i}$ Metrics', fontsize=20)
            if key == 'upper_left_trace':
                label = (r'tr$(Q_{%s}[:d, :d])$' % str(i))
            elif key == 'upper_right_trace':
                label = (r'tr$(Q_{%s}[:d, d:2d])$' % str(i))
            elif key == 'avg_abs_all_others':
                label = 'Avg Abs Others'

            mean_metric = np.mean(metric, axis=0)
            std_metric = np.std(metric, axis=0)
            stde_metric = std_metric / np.sqrt(metric.shape[0])
            assert mean_metric.shape == stde_metric.shape
            plt.plot(xs, mean_metric[:, i], label=label)
            plt.fill_between(xs, mean_metric[:, i] - stde_metric[:, i],
                            mean_metric[:, i] + stde_metric[:, i], alpha=0.3)
        plt.xlabel('# MRPs')
        plt.minorticks_off()
        plt.legend(frameon=True, framealpha=0.8).set_alpha(0.5)
        plt.savefig(os.path.join(save_dir,
                    f'Q_metrics_{i}.pdf'), dpi=300, format='pdf')
        plt.close(fig)


def align_matrix_sign(Ps: np.ndarray, Qs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    align the sign of the matrices using the sign of the bottom right element of P
    '''
    aligned_Ps = []
    aligned_Qs = []
    for P_mats, Q_mats in zip(Ps, Qs):  # shape (l, 2d+1, 2d+1)
        P_layers = []
        Q_layers = []
        for P, Q in zip(P_mats, Q_mats):  # shape (2d+1, 2d+1)
            P_layers.append(P * np.sign(P[-1, -1]))
            Q_layers.append(Q * np.sign(P[-1, -1]))
        aligned_Ps.append(P_layers)
        aligned_Qs.append(Q_layers)
    return np.array(aligned_Ps), np.array(aligned_Qs)


def compute_weight_metrics(attn_params: dict,
                           d: int) -> Tuple[dict, dict]:
    '''
    compute the metrics for the attention parameters
    attn_params: attention parameters from the transformer
    d: feature dimension
    '''

    P_metrics = {'bottom_right': [], 'avg_abs_all_others': []}
    Q_metrics = {'upper_left_trace': [],
                 'upper_right_trace': [], 'avg_abs_all_others': []}

    for P_t in attn_params['P']:  # shape (l, 2d+1, 2d+1)
        bottom_right_layers = []
        avg_abs_all_others_layers = []
        for P_layer in P_t:  # shape (2d+1, 2d+1)
            bottom_right, avg_abs_all_others = compare_P(
                P_layer, d)
            bottom_right_layers.append(bottom_right)
            avg_abs_all_others_layers.append(avg_abs_all_others)
        P_metrics['bottom_right'].append(bottom_right_layers)
        P_metrics['avg_abs_all_others'].append(avg_abs_all_others_layers)
    for key, value in P_metrics.items():
        P_metrics[key] = np.array([value])  # shape (1, T, l)

    for Q_t in attn_params['Q']:  # shape (l, 2d+1, 2d+1)
        upper_left_trace_layers = []
        upper_right_trace_layers = []
        avg_abs_all_others_layers = []
        for Q_layer in Q_t:  # shape (2d+1, 2d+1)
            upper_left_trace, upper_right_trace, avg_abs_all_others = compare_Q(
                Q_layer, d)
            upper_left_trace_layers.append(upper_left_trace)
            upper_right_trace_layers.append(upper_right_trace)
            avg_abs_all_others_layers.append(avg_abs_all_others)
        Q_metrics['upper_left_trace'].append(upper_left_trace_layers)
        Q_metrics['upper_right_trace'].append(upper_right_trace_layers)
        Q_metrics['avg_abs_all_others'].append(avg_abs_all_others_layers)
    for key, value in Q_metrics.items():
        Q_metrics[key] = np.array([value])  # shape (1, T, l)

    return P_metrics, Q_metrics


if __name__ == '__main__':
    runs_directory = os.path.join(
        './logs', 'nonlinear_discounted_train', '2024-05-10-01-06-39_standard')
    runs_to_plot = [run for run in os.listdir(
        runs_directory) if run.startswith('seed')]
    plot_attn_params([os.path.join(runs_directory, run)
                           for run in runs_to_plot], runs_directory)
