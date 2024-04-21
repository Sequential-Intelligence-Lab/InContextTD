import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from experiment.utils import (check_params, compare_P, compare_Q,
                              get_hardcoded_P, get_hardcoded_Q)


def load_data(data_dir):
    """Load data from specified directory and return the relevant metrics."""

    with np.load(os.path.join(data_dir, 'data.npz')) as data, \
            open(os.path.join(data_dir, 'params.json'), 'r') as params_file:
        log = {key: data[key] for key in data}
        # Assuming params is used somewhere else
        params = json.load(params_file)
    return log, params


def process_log(log: dict):
    Ps, Qs = log['P'], log['Q']
    aligned_Ps, aligned_Qs = align_matrix_sign(Ps, Qs)
    attn_params = {'P': aligned_Ps, 'Q': aligned_Qs}

    error_log = {}
    for key in ('mstde',
                'msve weight error norm',
                'mspbe weight error norm',
                'true msve',
                'transformer msve',
                'transformer mspbe'):
        error_log[key] = np.expand_dims(log[key], axis=0)

    return log['xs'], error_log, attn_params


def _batch_runs(logs: List[dict]):
    keys = logs[0].keys()
    batched_logs = {key: np.vstack([log[key] for log in logs]) for key in keys}
    return batched_logs


def plot_multiple_runs(data_dirs, save_dir):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log, params_0 = load_data(data_dirs[0])

    error_log_lst = []
    # Load data from directories
    for data_dir in data_dirs:
        log, params = load_data(data_dir)
        check_params(params, params_0)
        xs, error_log, attn_params = process_log(log)
        error_log_lst.append(error_log)

    batched_error_log = _batch_runs(error_log_lst)
    plot_error_data(xs, batched_error_log, save_dir)

    # Compute the axis=0 mean for all the categories
    # mean_logs = {category: np.mean(
    #     data_logs[category], axis=0) for category in data_logs.keys()}
    # std_logs = {category: np.std(
    #     data_logs[category], axis=0) for category in data_logs.keys()}

    # # plot all the metrics
    # for category in [key for key in mean_logs.keys() if key != 'P' and key != 'Q']:
    #     plt.figure()
    #     plt.xlabel('Epochs')
    #     plt.ylabel(category)
    #     plt.title(
    #         f'{category} vs Epochs (l={params["l"]}, s={params["s"]}, sw={params["sample_weight"]})')
    #     plt.plot(mean_logs['xs'], mean_logs[category])
    #     plt.fill_between(mean_logs['xs'], mean_logs[category] - std_logs[category],
    #                      mean_logs[category] + std_logs[category], alpha=0.2)
    #     plt.savefig(os.path.join(save_dir, f'{category}.png'), dpi=300)
    #     plt.close()

    # # plot the averaged final matrices
    # for key in ['P', 'Q']:
    #     plt.figure()
    #     plt.matshow(mean_logs[key][-1])
    #     plt.colorbar()
    #     plt.title(f'Final {key} Matrix')
    #     plt.savefig(os.path.join(save_dir, f'avg_final_{key}.png'), dpi=300)
    #     plt.close()


def plot_error_data(xs: np.ndarray,
                    error_log: dict,
                    save_dir: str):

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
    plt.savefig(os.path.join(save_dir, 'loss_mstde.png'), dpi=300)
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
    plt.savefig(os.path.join(save_dir, 'weight_error_norm.png'), dpi=300)
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
    plt.savefig(os.path.join(save_dir, 'msve.png'), dpi=300)
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
    plt.savefig(os.path.join(save_dir, 'mspbe.png'), dpi=300)
    plt.close()


def plot_attention_params(xs: np.ndarray,
                          params: dict,
                          save_dir: str,
                          log_step: int = -1):
    Ps, Qs = params['P'], params['Q']  # both have shape (T, l, 2d+1, 2d+1)
    assert Ps.shape == Qs.shape
    ckpt = xs[log_step]
    P_mats, Q_mats = Ps[log_step], Qs[log_step]

    def scale(matrix: np.ndarray):
        return matrix / np.max(np.abs(matrix))

    for l, (P, Q) in enumerate(zip(P_mats, Q_mats)):
        P = scale(P)
        Q = scale(Q)
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
        cax1 = axs[0].matshow(P, vmin=-1, vmax=1)
        axs[0].set_title(f'Layer {l+1} P Matrix at MDP {ckpt}')
        axs[1].matshow(Q, vmin=-1, vmax=1)
        axs[1].set_title(f'Layer {l+1} Q Matrix at MDP {ckpt}')
        fig.colorbar(cax1, ax=axs, orientation='vertical')
        plt.savefig(os.path.join(save_dir, f'PQ_{l+1}_{ckpt}.png'), dpi=300)
        plt.close(fig)


def evaluate_weights(data_dirs, save_dir, debug=False):
    log, params_0 = load_data(data_dirs[0])
    data_logs = {key: [] for key in ['P_norm_diff', 'P_bottom_right', 'P_avg_abs_all_others',
                                     'Q_norm_diff', 'Q_upper_left_trace', 'Q_upper_right_trace', 'Q_avg_abs_all_others']}
    d = params_0['d']

    # compute the hardcoded P and Q matrices that implement TD
    P_true = get_hardcoded_P(d)
    Q_true = get_hardcoded_Q(d)

    for data_dir in data_dirs:
        log, params = load_data(data_dir)
        check_params(params, params_0)
        log = align_matrix_sign(log)
        P_metrics, Q_metrics = compute_metrics(log, P_true, Q_true, d)
        update_data_logs(data_logs, P_metrics, Q_metrics)

    mean_logs = {category: np.mean(
        data_logs[category], axis=0) for category in data_logs.keys()}
    std_logs = {category: np.std(
        data_logs[category], axis=0) for category in data_logs.keys()}

    p_keys = [key for key in mean_logs.keys() if key.startswith('P')]
    q_keys = [key for key in mean_logs.keys() if key.startswith('Q')]
    plt.figure()
    for key in p_keys:
        plt.plot(log['xs'], mean_logs[key], label=key)
        plt.fill_between(log['xs'], mean_logs[key] - std_logs[key],
                         mean_logs[key] + std_logs[key], alpha=0.2)
    plt.xlabel('Epochs')
    plt.title('Transformer P Matrix Metrics')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'P_metrics.png'), dpi=300)

    plt.figure()
    for key in q_keys:
        plt.plot(log['xs'], mean_logs[key], label=key)
        plt.fill_between(log['xs'], mean_logs[key] - std_logs[key],
                         mean_logs[key] + std_logs[key], alpha=0.2)
    plt.xlabel('Epochs')
    plt.title('Transformer Q Matrix Metrics')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'Q_metrics.png'), dpi=300)


def align_matrix_sign(Ps: np.ndarray, Qs: np.ndarray):
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


def compute_metrics(log, P_true, Q_true, d):
    P_metrics = {'norm_diff': [], 'bottom_right': [], 'avg_abs_all_others': []}
    Q_metrics = {'norm_diff': [], 'upper_left_trace': [],
                 'upper_right_trace': [], 'avg_abs_all_others': []}

    for P_t in log['P']:  # shape (l, 2d+1, 2d+1)
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
        P_metrics[key] = np.array(value)

    for Q_t in log['Q']:  # shape (l, 2d+1, 2d+1)
        norm_diff_layers = []
        upper_left_trace_layers = []
        upper_right_trace_layers = []
        avg_abs_all_others_layers = []
        for Q_layer in Q_t:  # shape (2d+1, 2d+1)
            norm_diff, upper_left_trace, upper_right_trace, avg_abs_all_others = compare_Q(
                Q_layer, Q_true, d)
            Q_metrics['norm_diff'].append(norm_diff)
            Q_metrics['upper_left_trace'].append(upper_left_trace)
            Q_metrics['upper_right_trace'].append(upper_right_trace)
            Q_metrics['avg_abs_all_others'].append(avg_abs_all_others)
    for key, value in Q_metrics.items():
        Q_metrics[key] = np.array(value)

    return P_metrics, Q_metrics


def update_data_logs(data_logs, P_metrics, Q_metrics):
    for key in P_metrics:
        data_logs[f'P_{key}'].append(P_metrics[key])
    for key in Q_metrics:
        data_logs[f'Q_{key}'].append(Q_metrics[key])


if __name__ == '__main__':
    runs_directory = os.path.join(
        './logs', 'discounted_train', '2024-04-18-21-07-30')
    runs_to_plot = [run for run in os.listdir(
        runs_directory) if run.startswith('seed')]
    plot_multiple_runs([os.path.join(runs_directory, run)
                       for run in runs_to_plot], runs_directory)
    evaluate_weights([os.path.join(runs_directory, run)
                     for run in runs_to_plot], runs_directory)
