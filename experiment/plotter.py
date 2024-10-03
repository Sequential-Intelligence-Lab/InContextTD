import json
import os
from typing import List, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import seaborn as sns

from experiment.utils import (check_params, compare_P, compare_Q,
                              get_hardcoded_P, get_hardcoded_Q, scale)


def load_data(data_dir: str) -> Tuple[dict, dict]:
    """
    Load data from specified directory and return the relevant metrics.
    """
    with np.load(os.path.join(data_dir, 'data.npz')) as data, \
            open(os.path.join(data_dir, 'params.json'), 'r') as params_file:
        log = {key: data[key] for key in data}
        # Assuming params is used somewhere else
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
    for key in ('zero order cos sim',
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


def plot_multiple_runs(data_dirs: List[str],
                       save_dir: str,
                       final_figures_dir: str) -> None:
    '''
    data_dirs: list of directories containing the data
    save_dir: directory to save the plots
    plot the data from multiple runs
    '''
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log, params_0 = load_data(data_dirs[0])

    error_log_lst = []
    is_linear = params_0['linear']
    if is_linear:
        P_metrics_lst = []
        Q_metrics_lst = []
    # Load data from directories
    for data_dir in data_dirs:
        log, params = load_data(data_dir)
        d = params['d']
        l = params['l'] if params['mode'] == 'sequential' else 1
        check_params(params, params_0)
        xs, error_log, attn_params = process_log(log)
        error_log_lst.append(error_log)
        if is_linear:
            P_metrics, Q_metrics = compute_weight_metrics(attn_params,
                                                          get_hardcoded_P(d),
                                                          get_hardcoded_Q(d),
                                                          d)
            P_metrics_lst.append(P_metrics)
            Q_metrics_lst.append(Q_metrics)

    batched_error_log = _batch_runs(error_log_lst)
    plot_error_data(xs, batched_error_log, save_dir, final_figures_dir, params)

    if is_linear:
        batched_Q_metrics = _batch_runs(Q_metrics_lst)
        batched_P_metrics = _batch_runs(P_metrics_lst)
        plot_weight_metrics(xs, l, batched_P_metrics,
                            batched_Q_metrics, save_dir, final_figures_dir, params)


def plot_mean_attn_params(data_dirs: List[str],
                          save_dir: str,
                          final_figures_dir: str,
                          log_step: int = -1) -> None:
    '''
    data_dirs: list of directories containing the data
    save_dir: directory to save the plots not in the final paper
    final_figures_dir: directory to save the final figures for the paper
    log_step: time step to visualize
    '''
    attn_dir = os.path.join(save_dir, 'mean_attention_params_plots')
    if not os.path.exists(attn_dir):
        os.makedirs(attn_dir)

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
        if hypers['mode'] == 'sequential':
            axs[0].set_title(
                f'Final $P_{l}$', fontsize=26)
            axs[1].set_title(
                f'Final $Q_{l}$', fontsize=26)
        else:
            axs[0].set_title(f'Final $P_{0}$', fontsize=26)
            axs[1].set_title(f'Final $Q_{0}$', fontsize=26)
        axs[1].matshow(Q, vmin=-1, vmax=1)
        fig.colorbar(cax1, ax=axs, orientation='vertical')
        axs[0].tick_params(axis='both', which='both',
                           bottom=False, top=False,
                           left=False, right=False)
        axs[1].tick_params(axis='both', which='both',
                           bottom=False, top=False,
                           left=False, right=False)
        save_path = os.path.join(attn_dir, f'PQ_mean_{l+1}_{step}.pdf')
        plt.savefig(save_path, dpi=300, format="pdf")
        # save a copy in the final dir
        final_save_path = os.path.join(final_figures_dir, f'PQ_mean_{l+1}_{step}.pdf')
        plt.savefig(final_save_path, dpi=300, format="pdf")
        plt.close(fig)

def get_tf_title(params: dict) -> str:
    # specify that there is no activation function for linear transformers
    if params['linear'] == True:
        params['act'] = 'None'

    # abbreviate the labels for plotting
    if params['mode'] == 'sequential':
        mode = 'seq'
    else:
        mode = 'auto'

    if params['sample_weight'] == True:
        sample_weight = 'rep'
    else:
        sample_weight = 'nonrep'

    transformer_title = f"TF(mode={mode}, L={params['l']}, act={params['act']}, vf={sample_weight})"
    return transformer_title

def plot_error_data(xs: np.ndarray,
                    error_log: dict,
                    save_dir: str,
                    final_figures_dir: str,
                    params: dict) -> None:
    '''
    plot the error data from validation
    xs: x-axis values
    error_log: dictionary containing the error data
    save_dir: directory to save the plots
    '''
    error_dir = os.path.join(save_dir, 'error_metrics_plots')
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

    # set plotting style (ieee supports color blindness)
    plt.style.use(['science', 'vibrant', 'no-latex'])

    num_seeds = error_log['mstde'].shape[0]

    # Value function similarity
    mean_vf_sim = np.mean(error_log['v_tf v_td msve'], axis=0)
    std_vf_sim = np.std(error_log['v_tf v_td msve'], axis=0)
    stde_vf_sim = std_vf_sim / np.sqrt(num_seeds)
    mean_zo_cos_sim = np.mean(error_log['zero order cos sim'], axis=0)
    std_zo_cos_sim = np.std(error_log['zero order cos sim'], axis=0)
    stde_zo_cos_sim = std_zo_cos_sim / np.sqrt(num_seeds)
    mean_sensitivity_cos_sim = np.mean(error_log['sensitivity cos sim'], axis=0)
    std_sensitivity_cos_sim = np.std(error_log['sensitivity cos sim'], axis=0)
    stde_sensitivity_cos_sim = std_sensitivity_cos_sim / np.sqrt(num_seeds)

    #plt.figure()
    fig, ax1 = plt.subplots()
    plt.title("Learned TF and Batch TD Comparison")
    ax1.set_xlabel('# MRPs')
    ax1.set_ylabel('Cosine Similarity')
    ax1.set_ylim(0, 1.1)
    plt.minorticks_off()
    b, = ax1.plot(xs, mean_zo_cos_sim, label='IWS',
                  color=sns.color_palette()[0])
    ax1.fill_between(xs, mean_zo_cos_sim - stde_zo_cos_sim,
                     mean_zo_cos_sim + stde_zo_cos_sim, lw=0, alpha=0.2, color=sns.color_palette()[0])
    a, = ax1.plot(xs, mean_sensitivity_cos_sim,
                  label='SS', color=sns.color_palette()[5], linestyle='dashed')
    ax1.fill_between(xs, mean_sensitivity_cos_sim - stde_sensitivity_cos_sim,
                     mean_sensitivity_cos_sim + stde_sensitivity_cos_sim, lw=0, alpha=0.3, color=sns.color_palette()[5])
    ax2 = ax1.twinx()
    plt.minorticks_off()
    ax2.set_ylabel('Value Difference')
    if params['sample_weight']:
        ax2.set_ylim(0, 3.0)
    else:
        ax2.set_ylim(0, 0.3)
    c, = ax2.plot(xs, mean_vf_sim, label='VD', color=sns.color_palette()[1])
    ax2.fill_between(xs, mean_vf_sim - stde_vf_sim,
                     mean_vf_sim + stde_vf_sim, lw=0, alpha=0.3, color=sns.color_palette()[1])
    p = [a, b, c]
    if params['linear']:
        ax2.legend(p, [p_.get_label() for p_ in p], frameon=True, framealpha=0.8, loc='center right').set_alpha(0.5)
    else:
        ax2.legend(p, [p_.get_label() for p_ in p], frameon=True, framealpha=0.8, loc='upper left').set_alpha(0.5)
    plt.savefig(os.path.join(error_dir, 'cos_similarity.pdf'), dpi=300, format='pdf')
    # save a copy in the final figures directory
    plt.savefig(os.path.join(final_figures_dir, 'cos_similarity.pdf'), dpi=300, format='pdf')
    plt.close()


def plot_attention_params(xs: np.ndarray,
                          params: dict,
                          save_dir: str,
                          log_step: int = -1,
                          ) -> List[str]:
    '''
    visualize the attention parameters at a specific time step
    xs: x-axis values
    params: attention parameters
    save_dir: directory to save the plots
    log_step: time step to visualize
    return the paths to the saved plots
    '''
    attn_dir = os.path.join(save_dir, 'attention_params_plots')
    if not os.path.exists(attn_dir):
        os.makedirs(attn_dir)

    # both have shape (l, 2d+1, 2d+1)
    P_mats, Q_mats = params['P'][log_step], params['Q'][log_step]
    step = xs[log_step]
    assert P_mats.shape == Q_mats.shape

    paths = []
    for l, (P, Q) in enumerate(zip(P_mats, Q_mats)):  # shape (2d+1, 2d+1)
        P = scale(P)
        Q = scale(Q)
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
        cax1 = axs[0].matshow(P, vmin=-1, vmax=1)
        axs[0].set_title(f'Layer {l+1} P Matrix at MRP {step}')
        axs[1].matshow(Q, vmin=-1, vmax=1)
        axs[1].set_title(f'Layer {l+1} Q Matrix at MRP {step}')
        fig.colorbar(cax1, ax=axs, orientation='vertical')
        axs[0].tick_params(axis='both', which='both',
                           bottom=False, top=False, left=False, right=False)
        axs[1].tick_params(axis='both', which='both',
                           bottom=False, top=False, left=False, right=False)
        save_path = os.path.join(attn_dir, f'PQ_{l+1}_{step}.pdf')
        plt.savefig(save_path, dpi=300, format = 'pdf')
        plt.close(fig)
        paths.append(save_path)

    return paths


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
        paths = plot_attention_params(xs, params, gif_dir, step, False)
        for i, path in enumerate(paths):
            gif_list[i].append(imageio.imread(path))
            os.remove(path)

    for i, gif in enumerate(gif_list):
        imageio.mimwrite(os.path.join(gif_dir, f'PQ_{i+1}.gif'), gif, fps=2)

    # remove the empty directory containing the temporary images
    os.rmdir(os.path.join(gif_dir, 'attention_params_plots'))


def plot_weight_metrics(xs: np.ndarray,
                        l: int,
                        P_metrics: dict,
                        Q_metrics: dict,
                        save_dir: str,
                        final_figures_dir: str,
                        params: dict) -> None:
    '''
    plot the metrics for P and Q
    xs: x-axis values
    l: number of layers
    P_metrics: metrics for P matrix
    Q_metrics: metrics for Q matrix
    save_dir: directory to save the plots
    '''
    P_metrics_dir = os.path.join(save_dir, 'P_metrics_plots')
    Q_metrics_dir = os.path.join(save_dir, 'Q_metrics_plots')
    if not os.path.exists(P_metrics_dir):
        os.makedirs(P_metrics_dir)
    if not os.path.exists(Q_metrics_dir):
        os.makedirs(Q_metrics_dir)

    transformer_title = get_tf_title(params)

    plt.style.use(['science', 'bright', 'no-latex'])
    # same layer, different metrics
    for i in range(l):
        fig = plt.figure()
        for key, metric in P_metrics.items():
            # for the standard setting in the main paper, don't plot the transformer title
            if params['mode'] == 'auto' and params['l'] == 3 and params['linear'] == True: 
                plt.title(f'$P_{i}$ Metrics', fontsize=20)
            else:
                plt.title(
                    f'$P_{i}$ Metrics')
            if key == 'norm_diff':
                label = (r'$||P_{%s} - P^{TD}_{%s}||_2$' % (str(i), str(i)))
            elif key == 'bottom_right':
                label = (r'$P_{%s}[-1, -1]$' % (str(i)))
            elif key == 'avg_abs_all_others':
                label = 'Avg Abs Others'
            mean_metric = np.mean(metric, axis=0)  # shape (T, l)
            std_metric = np.std(metric, axis=0)
            stde_metric = std_metric / np.sqrt(metric.shape[0])
            assert mean_metric.shape == stde_metric.shape
            if key != 'norm_diff':
                plt.plot(xs, mean_metric[:, i], label=label)
                plt.fill_between(xs, mean_metric[:, i] - stde_metric[:, i],
                                mean_metric[:, i] + stde_metric[:, i], alpha=0.3)
        plt.xlabel('# MRPs')
        plt.ylim(0,2)
        plt.minorticks_off()
        plt.legend(frameon=True, framealpha=0.8,
                    fontsize='medium').set_alpha(0.5) 
        plt.savefig(os.path.join(P_metrics_dir,
                    f'P_metrics_{i+1}.pdf'), dpi=300, format='pdf')
        # save a copy in the final figures directory
        plt.savefig(os.path.join(final_figures_dir, f'P_metrics_{i+1}.pdf'), dpi=300, format='pdf')
        plt.close(fig)

    # same metric, different layers
    for key, metric in P_metrics.items():
        mean_metric = np.mean(metric, axis=0)  # shape (T, l)
        std_metric = np.std(metric, axis=0)
        assert mean_metric.shape == std_metric.shape
        fig = plt.figure()
        for i in range(l):
            plt.plot(xs, mean_metric[:, i], label=f'layer={i+1}')
            plt.fill_between(xs, mean_metric[:, i] - std_metric[:, i],
                             mean_metric[:, i] + std_metric[:, i], alpha=0.2)
            plt.xlabel('# MRPs')
            plt.title(f'Transformer P Matrix {key.replace("_", " ").title()}')
            plt.legend()
        plt.savefig(os.path.join(P_metrics_dir, f'P_{key}.pdf'), dpi=300, format='pdf')
        plt.close(fig)

    # same layer, different metrics
    for i in range(l):
        fig = plt.figure()
        for key, metric in Q_metrics.items():
            # for the standard setting in the main paper, don't plot the transformer title
            plt.title(f'$Q_{i}$ Metrics', fontsize=20)
            if key == 'norm_diff':
                label = (r'$||Q_{%s} - Q^{TD}_{%s}||_2$' % (str(i), str(i)))
            elif key == 'upper_left_trace':
                label = (r'tr$(Q_{%s}[:d, :d])$' % str(i))
            elif key == 'upper_right_trace':
                label = (r'tr$(Q_{%s}[:d, d:2d])$' % str(i))
            elif key == 'avg_abs_all_others':
                label = 'Avg Abs Others'
            mean_metric = np.mean(metric, axis=0)
            std_metric = np.std(metric, axis=0)
            stde_metric = std_metric / np.sqrt(metric.shape[0])
            assert mean_metric.shape == stde_metric.shape
            if key != 'norm_diff':
                plt.plot(xs, mean_metric[:, i], label=label)
                plt.fill_between(xs, mean_metric[:, i] - stde_metric[:, i],
                                mean_metric[:, i] + stde_metric[:, i], alpha=0.3)
        plt.xlabel('# MRPs')
        plt.minorticks_off()
        plt.legend(frameon=True, framealpha=0.8).set_alpha(0.5)
        plt.savefig(os.path.join(Q_metrics_dir,
                    f'Q_metrics_{i+1}.pdf'), dpi=300, format='pdf')
        plt.savefig(os.path.join(final_figures_dir,
            f'Q_metrics_{i+1}.pdf'), dpi=300, format='pdf')
        plt.close(fig)

    # same metric, different layers
    for key, metric in Q_metrics.items():
        mean_metric = np.mean(metric, axis=0)
        std_metric = np.std(metric, axis=0)
        assert mean_metric.shape == std_metric.shape
        fig = plt.figure()
        for i in range(l):
            plt.plot(xs, mean_metric[:, i], label=f'layer={i+1}')
            plt.fill_between(xs, mean_metric[:, i] - std_metric[:, i],
                             mean_metric[:, i] + std_metric[:, i], alpha=0.3)
            plt.xlabel('# MRPs')
            plt.title(f'Transformer Q Matrix {key.replace("_", " ").title()}')
            plt.legend()
        plt.savefig(os.path.join(Q_metrics_dir, f'Q_{key}.pdf'), dpi=300,format="pdf")
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
                           P_true: np.ndarray,
                           Q_true: np.ndarray,
                           d: int) -> Tuple[dict, dict]:
    '''
    compute the metrics for the attention parameters
    attn_params: attention parameters from the transformer
    P_true: hard coded true P matrix
    Q_true: hard coded true Q matrix
    d: feature dimension
    '''

    P_metrics = {'norm_diff': [], 'bottom_right': [], 'avg_abs_all_others': []}
    Q_metrics = {'norm_diff': [], 'upper_left_trace': [],
                 'upper_right_trace': [], 'avg_abs_all_others': []}

    for P_t in attn_params['P']:  # shape (l, 2d+1, 2d+1)
        norm_diff_layers = []
        bottom_right_layers = []
        avg_abs_all_others_layers = []
        for P_layer in P_t:  # shape (2d+1, 2d+1)
            norm_diff, bottom_right, avg_abs_all_others = compare_P(
                P_layer, P_true, d)
            norm_diff_layers.append(norm_diff)
            bottom_right_layers.append(bottom_right)
            avg_abs_all_others_layers.append(avg_abs_all_others)
        P_metrics['norm_diff'].append(norm_diff_layers)
        P_metrics['bottom_right'].append(bottom_right_layers)
        P_metrics['avg_abs_all_others'].append(avg_abs_all_others_layers)
    for key, value in P_metrics.items():
        P_metrics[key] = np.array([value])  # shape (1, T, l)

    for Q_t in attn_params['Q']:  # shape (l, 2d+1, 2d+1)
        norm_diff_layers = []
        upper_left_trace_layers = []
        upper_right_trace_layers = []
        avg_abs_all_others_layers = []
        for Q_layer in Q_t:  # shape (2d+1, 2d+1)
            norm_diff, upper_left_trace, upper_right_trace, avg_abs_all_others = compare_Q(
                Q_layer, Q_true, d)
            norm_diff_layers.append(norm_diff)
            upper_left_trace_layers.append(upper_left_trace)
            upper_right_trace_layers.append(upper_right_trace)
            avg_abs_all_others_layers.append(avg_abs_all_others)
        Q_metrics['norm_diff'].append(norm_diff_layers)
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
    plot_multiple_runs([os.path.join(runs_directory, run)
                       for run in runs_to_plot], runs_directory)
    plot_mean_attn_params([os.path.join(runs_directory, run)
                           for run in runs_to_plot], runs_directory)
