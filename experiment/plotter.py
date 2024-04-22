import json
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

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
        params = json.load(params_file)
    return log, params


def process_log(log: dict) -> Tuple[np.ndarray, dict, dict]:
    '''
    parse the log dictionary
    return the x ticks, error log, and aligned attention parameters
    '''
    Ps, Qs = log['P'], log['Q']
    aligned_Ps, aligned_Qs = align_matrix_sign(Ps, Qs)
    attn_params = {'P': aligned_Ps, 'Q': aligned_Qs}

    error_log = {}
    for key in ('mstde',
                'msve weight error norm',
                'mspbe weight error norm',
                'true msve',
                'transformer msve',
                'transformer mspbe',
                'implicit w_tf and w_td cos sim',
                'w_tf w_td diff l2'):
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
                       save_dir: str) -> None:
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
        P_metrics, Q_metrics = compute_weight_metrics(attn_params,
                                                      get_hardcoded_P(d),
                                                      get_hardcoded_Q(d),
                                                      d)
        P_metrics_lst.append(P_metrics)
        Q_metrics_lst.append(Q_metrics)

    batched_error_log = _batch_runs(error_log_lst)
    plot_error_data(xs, batched_error_log, save_dir)

    batched_Q_metrics = _batch_runs(Q_metrics_lst)
    batched_P_metrics = _batch_runs(P_metrics_lst)
    plot_weight_metrics(xs, l, batched_P_metrics, batched_Q_metrics, save_dir)

def plot_error_data(xs: np.ndarray,
                    error_log: dict,
                    save_dir: str) -> None:
    '''
    plot the error data from validation
    xs: x-axis values
    error_log: dictionary containing the error data
    save_dir: directory to save the plots
    '''
    error_dir = os.path.join(save_dir, 'error_metrics_plots')
    if not os.path.exists(error_dir):
        os.makedirs(error_dir)

    # MSTDE
    mean_mstde = np.mean(error_log['mstde'], axis=0)
    std_mstde = np.std(error_log['mstde'], axis=0)
    plt.figure()
    plt.plot(xs, mean_mstde, label='MSTDE')
    plt.fill_between(xs, mean_mstde - std_mstde,
                     mean_mstde + std_mstde, alpha=0.2)
    plt.xlabel('# MDPs')
    plt.ylabel('Loss (MSTDE)')
    plt.title('Loss (MSTDE) vs # MDPs')
    plt.legend()
    plt.savefig(os.path.join(error_dir, 'loss_mstde.png'), dpi=300)
    plt.close()

    # Weight norm
    mean_msve_weight_error_norm = np.mean(
        error_log['msve weight error norm'], axis=0)
    mean_mspbe_weight_error_norm = np.mean(
        error_log['mspbe weight error norm'], axis=0)
    std_msve_weight_error_norm = np.std(
        error_log['msve weight error norm'], axis=0)
    std_mspbe_weight_error_norm = np.std(
        error_log['mspbe weight error norm'], axis=0)
    plt.figure()
    plt.plot(xs, mean_msve_weight_error_norm,
             label='MSVE Weight Error Norm')
    plt.fill_between(xs, mean_msve_weight_error_norm - std_msve_weight_error_norm,
                     mean_msve_weight_error_norm + std_msve_weight_error_norm, alpha=0.2)
    plt.plot(xs, mean_mspbe_weight_error_norm,
             label='MSPBE Weight Error Norm')
    plt.fill_between(xs, mean_mspbe_weight_error_norm - std_mspbe_weight_error_norm,
                     mean_mspbe_weight_error_norm + std_mspbe_weight_error_norm, alpha=0.2)
    plt.xlabel('# MDPs')
    plt.ylabel('Weight Error L2 Norm')
    plt.title('Weight Error Norm vs # MDPs')
    plt.legend()
    plt.savefig(os.path.join(error_dir, 'weight_error_norm.png'), dpi=300)
    plt.close()

    # Value error
    mean_true_msve = np.mean(error_log['true msve'], axis=0)
    mean_tf_msve = np.mean(error_log['transformer msve'], axis=0)
    std_true_msve = np.std(error_log['true msve'], axis=0)
    std_tf_msve = np.std(error_log['transformer msve'], axis=0)

    plt.figure()
    plt.plot(xs, mean_true_msve, label='True MSVE')
    plt.fill_between(xs, mean_true_msve - std_true_msve,
                     mean_true_msve + std_true_msve, alpha=0.2)
    plt.plot(xs, mean_tf_msve, label='Transformer MSVE')
    plt.fill_between(xs, mean_tf_msve - std_tf_msve,
                     mean_tf_msve + std_tf_msve, alpha=0.2)
    plt.xlabel('# MDPs')
    plt.ylabel('MSVE')
    plt.title('MSVE vs # MDPs')
    plt.legend()
    plt.savefig(os.path.join(error_dir, 'msve.png'), dpi=300)
    plt.close()

    # MSPBE
    mean_tf_mspbe = np.mean(error_log['transformer mspbe'], axis=0)
    std_tf_mspbe = np.std(error_log['transformer mspbe'], axis=0)
    plt.figure()
    plt.plot(xs, mean_tf_mspbe, label='Transformer MSPBE')
    plt.fill_between(xs, mean_tf_mspbe - std_tf_mspbe,
                     mean_tf_mspbe + std_tf_mspbe, alpha=0.2)
    plt.xlabel('# MDPs')
    plt.ylabel('MSPBE')
    plt.title('MSPBE vs # MDPs')
    plt.legend()
    plt.savefig(os.path.join(error_dir, 'mspbe.png'), dpi=300)
    plt.close()

    # TF weight and TD weight comparison
    mean_cos_sim = np.mean(error_log['implicit w_tf and w_td cos sim'], axis=0)
    std_cos_sim = np.std(error_log['implicit w_tf and w_td cos sim'], axis=0)
    mean_weight_diff = np.mean(error_log['w_tf w_td diff l2'], axis=0)
    std_weight_diff = np.std(error_log['w_tf w_td diff l2'], axis=0)
    plt.figure()
    plt.plot(xs, mean_cos_sim, label='Cosine Similarity')
    plt.fill_between(xs, mean_cos_sim - std_cos_sim,
                     mean_cos_sim + std_cos_sim, alpha=0.2)
    plt.plot(xs, mean_weight_diff, label='L2 Norm Weight Difference')
    plt.fill_between(xs, mean_weight_diff - std_weight_diff,
                     mean_weight_diff + std_weight_diff, alpha=0.2)
    plt.fill_between(xs, mean_cos_sim - std_cos_sim,
                     mean_cos_sim + std_cos_sim, alpha=0.2)
    plt.xlabel('# MDPs')
    plt.title('Transformer Implicit weight and l-step TD weight Cosine Similarity')
    plt.legend()
    plt.savefig(os.path.join(error_dir, 'tf_td_weight_comparison.png'), dpi=300)
    plt.close()


def plot_attention_params(xs: np.ndarray,
                          params: dict,
                          save_dir: str,
                          log_step: int = -1) -> None:
    '''
    visualize the attention parameters at a specific time step
    xs: x-axis values
    params: attention parameters
    save_dir: directory to save the plots
    log_step: time step to visualize
    '''
    attn_dir = os.path.join(save_dir, 'attention_params_plots')
    if not os.path.exists(attn_dir):
        os.makedirs(attn_dir)

    Ps, Qs = params['P'], params['Q']  # both have shape (T, l, 2d+1, 2d+1)
    assert Ps.shape == Qs.shape
    ckpt = xs[log_step]
    P_mats, Q_mats = Ps[log_step], Qs[log_step]

    for l, (P, Q) in enumerate(zip(P_mats, Q_mats)):
        P = scale(P)
        Q = scale(Q)
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
        cax1 = axs[0].matshow(P, vmin=-1, vmax=1)
        axs[0].set_title(f'Layer {l+1} P Matrix at MDP {ckpt}')
        axs[1].matshow(Q, vmin=-1, vmax=1)
        axs[1].set_title(f'Layer {l+1} Q Matrix at MDP {ckpt}')
        fig.colorbar(cax1, ax=axs, orientation='vertical')
        plt.savefig(os.path.join(attn_dir, f'PQ_{l+1}_{ckpt}.png'), dpi=300)
        plt.close(fig)


def plot_weight_metrics(xs: np.ndarray,
                        l: int,
                        P_metrics: dict,
                        Q_metrics: dict,
                        save_dir: str) -> None:
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

    # same layer, different metrics
    for i in range(l):
        plt.figure()
        for key, metric in P_metrics.items():
            mean_metric = np.mean(metric, axis=0)  # shape (T, l)
            std_metric = np.std(metric, axis=0)
            assert mean_metric.shape == std_metric.shape
            plt.plot(xs, mean_metric[:, i], label=key.replace('_', ' '))
            plt.fill_between(xs, mean_metric[:, i] - std_metric[:, i],
                             mean_metric[:, i] + std_metric[:, i], alpha=0.2)
            plt.xlabel('# MDPs')
            plt.title(f'Transformer P Matrix Layer {i+1} Metrics')
            plt.legend()
        plt.savefig(os.path.join(P_metrics_dir, f'P_metrics_{i+1}.png'), dpi=300)
        plt.close()


    # same metric, different layers
    for key, metric in P_metrics.items():
        mean_metric = np.mean(metric, axis=0)  # shape (T, l)
        std_metric = np.std(metric, axis=0)
        assert mean_metric.shape == std_metric.shape
        plt.figure()
        for i in range(l):
            plt.plot(xs, mean_metric[:, i], label=f'layer={i+1}')
            plt.fill_between(xs, mean_metric[:, i] - std_metric[:, i],
                             mean_metric[:, i] + std_metric[:, i], alpha=0.2)
            plt.xlabel('# MDPs')
            plt.title(f'Transformer P Matrix {key.replace("_", " ").title()}')
            plt.legend()
        plt.savefig(os.path.join(P_metrics_dir, f'P_{key}.png'), dpi=300)
        plt.close()

    # same layer, different metrics
    for i in range(l):
        plt.figure()
        for key, metric in Q_metrics.items():
            mean_metric = np.mean(metric, axis=0)
            std_metric = np.std(metric, axis=0)
            assert mean_metric.shape == std_metric.shape
            plt.plot(xs, mean_metric[:, i], label=key.replace('_', ' '))
            plt.fill_between(xs, mean_metric[:, i] - std_metric[:, i],
                             mean_metric[:, i] + std_metric[:, i], alpha=0.2)
            plt.xlabel('# MDPs')
            plt.title(f'Transformer Q Matrix Layer {i+1} Metrics')
            plt.legend()
        plt.savefig(os.path.join(Q_metrics_dir, f'Q_metrics_{i+1}.png'), dpi=300)
        plt.close()

    # same metric, different layers
    for key, metric in Q_metrics.items():
        mean_metric = np.mean(metric, axis=0)
        std_metric = np.std(metric, axis=0)
        assert mean_metric.shape == std_metric.shape
        plt.figure()
        for i in range(l):
            plt.plot(xs, mean_metric[:, i], label=f'layer={i+1}')
            plt.fill_between(xs, mean_metric[:, i] - std_metric[:, i],
                             mean_metric[:, i] + std_metric[:, i], alpha=0.2)
            plt.xlabel('# MDPs')
            plt.title(f'Transformer Q Matrix {key.replace("_", " ").title()}')
            plt.legend()
        plt.savefig(os.path.join(Q_metrics_dir, f'Q_{key}.png'), dpi=300)
        plt.close()


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
        './logs', 'discounted_train', '2024-04-20-17-31-58')
    runs_to_plot = [run for run in os.listdir(
        runs_directory) if run.startswith('seed')]
    plot_multiple_runs([os.path.join(runs_directory, run)
                       for run in runs_to_plot], runs_directory)
