import os
import matplotlib.pyplot as plt
import numpy as np
import pickle
import json

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
    data_logs = {category: [] for category in log.keys()}

    # Load data from directories
    for data_dir in data_dirs:
        log, params = load_data(data_dir)
        for category in log.keys():
            data_logs[category].append(log[category])

    # Compute the axis=0 mean for all the categories
    mean_logs = {category: np.mean(data_logs[category], axis=0) for category in data_logs.keys()}
    std_logs = {category: np.std(data_logs[category], axis=0) for category in data_logs.keys()}
    #import pdb; pdb.set_trace()
    for category in mean_logs.keys():
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel(category)
        plt.title(f'{category} vs Epochs (l={params["l"]}, s={params["s"]}, sw={params["sample_weight"]})')
        #plt.errorbar(mean_logs['xs'], mean_logs[category], yerr=std_logs[category], fmt='o')
        plt.plot(mean_logs['xs'], mean_logs[category])
        plt.fill_between(mean_logs['xs'], mean_logs[category] - std_logs[category], mean_logs[category] + std_logs[category], alpha=0.2)
        #plt.legend()
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


def evaluate_weights(tf, save_dir):
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

if __name__ == '__main__':
    runs_directory = os.path.join('./logs', 'discounted_train', '2024-04-18-14-28-52')
    runs_to_plot = [run for run in os.listdir(runs_directory) if run.startswith('seed')]
    import pdb; pdb.set_trace()
    plot_multiple_runs([os.path.join(runs_directory, run) for run in runs_to_plot], runs_directory)