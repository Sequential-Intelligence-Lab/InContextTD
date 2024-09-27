import os
from argparse import ArgumentParser, Namespace

from experiment.plotter import (compute_weight_metrics,
                                generate_attention_params_gif, get_hardcoded_P,
                                get_hardcoded_Q, load_data,
                                plot_attention_params, plot_error_data,
                                plot_mean_attn_params, plot_multiple_runs,
                                plot_weight_metrics, process_log)

if __name__ == '__main__':
    linear_path = os.path.join('logs', 'linear_discounted_train', 'linear_good')
    nonlinear_path = os.path.join('logs', 'nonlinear_discounted_train', 'nonlinear_good')
    linear_tasks = ['standard', 
             'sequential', '1layer', '2layers',
             '4layers', 'sequential_2layers', 'sequential_4layers', 
             'large']
    nonlinear_tasks = ['standard', 'sequential', 'representable', 'large', 'relu']

    # We only want a subset of the final figures to be in the paper
    # We save those in a separate directory we can upload to Overleaf directly
    final_plot_path = os.path.join('logs', 'final_plots') 
    # Linear tasks
    for task in linear_tasks:
        task_path = os.path.join(linear_path, task)
        final_task_path = os.path.join(final_plot_path, "linear",task)
        if not os.path.exists(final_task_path):
            os.makedirs(final_task_path)
        data_paths = []
        seeds = []
        for dir in os.listdir(task_path):
            if dir.startswith('seed_'):
                seeds.append(dir)
        for seed in seeds:
            data_path = os.path.join(task_path, seed)
            data_paths.append(data_path)
        plot_multiple_runs(data_paths, save_dir=task_path, final_figures_dir=final_task_path)
        plot_mean_attn_params(data_paths, save_dir=task_path, final_figures_dir=final_task_path)

    # Nonlinear tasks
    for task in nonlinear_tasks:
        task_path = os.path.join(nonlinear_path, task)
        final_task_path = os.path.join(final_plot_path,"nonlinear", task)
        if not os.path.exists(final_task_path):
            os.makedirs(final_task_path)
        data_paths = []
        seeds = []
        for dir in os.listdir(task_path):
            if dir.startswith('seed_'):
                seeds.append(dir)
        for seed in seeds:
            data_path = os.path.join(task_path, seed)
            data_paths.append(data_path)
        plot_multiple_runs(data_paths, save_dir=task_path, final_figures_dir=final_task_path)
        plot_mean_attn_params(data_paths, save_dir=task_path, final_figures_dir=final_task_path)