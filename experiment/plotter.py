import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json

from experiment.utils import compare_P, compare_Q, check_params, stack_four_np

def load_data(data_dir):
    """Load data from specified directory and return the relevant metrics."""

    with open(os.path.join(data_dir, 'discounted_train.pkl'), 'rb') as train_file, \
         open(os.path.join(data_dir, 'params.json'), 'r') as params_file:
        log = pickle.load(train_file)
        params = json.load(params_file)  # Assuming params is used somewhere else

    return log, params

def plot_multiple_runs(data_dirs, save_dir):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize lists to store data
    log, params_0 = load_data(data_dirs[0])
    data_logs = {category: [] for category in log.keys() if category != 'P' and category != 'Q'}

    # Load data from directories
    for data_dir in data_dirs:
        log, params = load_data(data_dir)
        check_params(params, params_0)
        for category in [k for k in log.keys() if k != 'P' and k != 'Q']:
            data_logs[category].append(log[category])

    # Compute the axis=0 mean for all the categories
    mean_logs = {category: np.mean(data_logs[category], axis=0) for category in data_logs.keys()}
    std_logs = {category: np.std(data_logs[category], axis=0) for category in data_logs.keys()}
    for category in mean_logs.keys():
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel(category)
        plt.title(f'{category} vs Epochs (l={params["l"]}, s={params["s"]}, sw={params["sample_weight"]})')
        plt.plot(mean_logs['xs'], mean_logs[category])
        plt.fill_between(mean_logs['xs'], mean_logs[category] - std_logs[category], mean_logs[category] + std_logs[category], alpha=0.2)
        plt.savefig(os.path.join(save_dir, f'{category}.png'), dpi=300)
        plt.close()
   

def plot_data(log, save_dir):

    # Loss Plot
    plt.figure()
    plt.plot(log['xs'], log['mstde'], label='MSTDE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_mstde.png'), dpi=300)
    plt.close()

    # Weight norm plot
    plt.figure()
    plt.plot(log['xs'], log['msve weight error norm'],
             label='MSVE Weight Error Norm')
    plt.plot(log['xs'], log['mspbe weight error norm'],
             label='MSPBE Weight Error Norm')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Error L2 Norm')
    plt.title('Weight Error Norm vs Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'weight_error_norm.png'), dpi=300)
    plt.close()

    # Value Error Plot
    plt.figure()
    plt.plot(log['xs'], log['true msve'], label='True MSVE')
    plt.plot(log['xs'], log['transformer msve'], label='Transformer MSVE')
    plt.xlabel('Epochs')
    plt.ylabel('MSVE')
    plt.title('MSVE vs Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'msve.png'), dpi=300)
    plt.close()

    # MSPBE Plot
    plt.figure()
    plt.plot(log['xs'], log['transformer mspbe'], label='Transformer MSPBE')
    plt.xlabel('Epochs')
    plt.ylabel('MSPBE')
    plt.title('MSPBE vs Epochs')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'mspbe.png'), dpi=300)
    plt.close()


def print_final_weights(tf, save_dir):
    # Save the final P and Q matrices
    final_P = tf.attn.P.detach().numpy()
    final_Q = tf.attn.Q.detach().numpy()
    final_M = tf.attn.M.numpy()

    plt.figure()
    plt.matshow(final_P)
    plt.colorbar()
    plt.title('Final P Matrix')
    plt.savefig(os.path.join(save_dir, 'final_P.png'), dpi=300)

    plt.figure()
    plt.matshow(final_Q)
    plt.colorbar()
    plt.title('Final Q Matrix')
    plt.savefig(os.path.join(save_dir, 'final_Q.png'), dpi=300)

    plt.figure()
    plt.matshow(final_M)
    plt.colorbar()
    plt.title('Final M Matrix')
    plt.savefig(os.path.join(save_dir, 'final_M.png'), dpi=300)

def evaluate_weights(data_dirs, save_dir, debug=False):
    log, params_0 = load_data(data_dirs[0])
    data_logs = {key: [] for key in ['P_norm_diff', 'P_bottom_right', 'P_avg_abs_all_others', 
                                 'Q_norm_diff', 'Q_upper_left_trace', 'Q_upper_right_trace', 'Q_avg_abs_all_others']}
    d = params_0['d']

    # compute the hardcoded P and Q matrices that implement TD
    P_true = np.zeros((2*params_0['d']+1, 2*params_0['d']+1))
    P_true[-1, -1] = 1

    I = np.eye(d)
    O = np.zeros((d, d))
    C = np.eye(d) # just use the identity matrix as pre-conditioner
    A = stack_four_np(-C.T, C.T, O, O)
    Q_true= np.zeros(P_true.shape)
    Q_true[:2*d, :2*d] = A

    for data_dir in data_dirs:
        log, params = load_data(data_dir)
        check_params(params, params_0)
        log = negate_matrices_if_needed(log)
        P_metrics, Q_metrics = compute_metrics(log, P_true, Q_true, d)
        update_data_logs(data_logs, P_metrics, Q_metrics)

    mean_logs = {category: np.mean(data_logs[category], axis=0) for category in data_logs.keys()}
    std_logs = {category: np.std(data_logs[category], axis=0) for category in data_logs.keys()}

    p_keys = [key for key in mean_logs.keys() if key.startswith('P')]
    q_keys = [key for key in mean_logs.keys() if key.startswith('Q')]
    plt.figure()
    for key in p_keys:
        plt.plot(log['xs'], mean_logs[key], label=key)
        plt.fill_between(log['xs'], mean_logs[key] - std_logs[key], mean_logs[key] + std_logs[key], alpha=0.2)
    plt.xlabel('Epochs')
    plt.title('Transformer P Matrix Metrics')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'P_metrics.png'), dpi=300)

    plt.figure()
    for key in q_keys:
        plt.plot(log['xs'], mean_logs[key], label=key)
        plt.fill_between(log['xs'], mean_logs[key] - std_logs[key], mean_logs[key] + std_logs[key], alpha=0.2)
    plt.xlabel('Epochs')
    plt.title('Transformer Q Matrix Metrics')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'Q_metrics.png'), dpi=300)    


def negate_matrices_if_needed(log):
    final_P = log['P'][-1]
    if final_P[-1, -1] < 0:
        log['P'] = [-p for p in log['P']]
        log['Q'] = [-q for q in log['Q']]
    return log

def compute_metrics(log, P_true, Q_true, d):
    P_metrics = {'norm_diff': [], 'bottom_right': [], 'avg_abs_all_others': []}
    Q_metrics = {'norm_diff': [], 'upper_left_trace': [], 'upper_right_trace': [], 'avg_abs_all_others': []}
    
    for P_tf in log['P']:
        P_norm_diff, P_bottom_right, P_sum_all_others = compare_P(P_tf, P_true, d)
        P_metrics['norm_diff'].append(P_norm_diff)
        P_metrics['bottom_right'].append(P_bottom_right)
        P_metrics['avg_abs_all_others'].append(P_sum_all_others)

    for Q_tf in log['Q']:
        Q_norm_diff, Q_upper_left_trace, Q_upper_right_trace, Q_sum_all_others = compare_Q(Q_tf, Q_true, d)
        Q_metrics['norm_diff'].append(Q_norm_diff)
        Q_metrics['upper_left_trace'].append(Q_upper_left_trace)
        Q_metrics['upper_right_trace'].append(Q_upper_right_trace)
        Q_metrics['avg_abs_all_others'].append(Q_sum_all_others)

    return P_metrics, Q_metrics

def update_data_logs(data_logs, P_metrics, Q_metrics):
    for key in P_metrics:
        data_logs[f'P_{key}'].append(P_metrics[key])
    for key in Q_metrics:
        data_logs[f'Q_{key}'].append(Q_metrics[key])

if __name__ == '__main__':
    runs_directory = os.path.join('./logs', 'discounted_train', '2024-04-18-21-07-30')
    runs_to_plot = [run for run in os.listdir(runs_directory) if run.startswith('seed')]
    plot_multiple_runs([os.path.join(runs_directory, run) for run in runs_to_plot], runs_directory)
    evaluate_weights([os.path.join(runs_directory, run) for run in runs_to_plot], runs_directory)