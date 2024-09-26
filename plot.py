import os

from experiment.plotter import (compute_weight_metrics,
                                generate_attention_params_gif, get_hardcoded_P,
                                get_hardcoded_Q, load_data,
                                plot_attention_params, plot_error_data,
                                plot_mean_attn_params, plot_multiple_runs,
                                plot_weight_metrics, process_log)

if __name__ == '__main__':
    log_path = os.path.join('logs', 'linear_discounted_train')
    tasks = ['standard', 'sequential', '1layer', '2layers',
             '4layers', 'sequential_2layers', 'sequential_4layers']
    # tasks = ['standard']
    for task in tasks:
        task_path = os.path.join(log_path, task)
        data_paths = []
        seeds = []
        for dir in os.listdir(task_path):
            if dir.startswith('seed_'):
                seeds.append(dir)
        for seed in seeds:
            data_path = os.path.join(task_path, seed)
            data_paths.append(data_path)
            # log, hyperparams = load_data(data_path)
            # xs, error_log, attn_params = process_log(log)
            # l_tf = hyperparams['l'] if hyperparams['mode'] == 'sequential' else 1
            # plot_error_data(xs, error_log, save_dir=data_path,
            #                 params=hyperparams)
            # plot_attention_params(xs, attn_params, save_dir=data_path)
            # generate_attention_params_gif(xs, l_tf, attn_params, data_path)
            # if hyperparams['linear']:
            #     P_true = get_hardcoded_P(hyperparams['d'])
            #     Q_true = get_hardcoded_Q(hyperparams['d'])
            #     P_metrics, Q_metrics = compute_weight_metrics(attn_params, P_true,
            #                                                   Q_true, hyperparams['d'])
            #     plot_weight_metrics(xs, l_tf, P_metrics, Q_metrics,
            #                         data_path, hyperparams)
        plot_multiple_runs(data_paths, save_dir=task_path)
        plot_mean_attn_params(data_paths, save_dir=task_path)
