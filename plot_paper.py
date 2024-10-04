import os
from argparse import ArgumentParser, Namespace

from experiment.plotter import (plot_mean_attn_params, plot_multiple_runs,
                                plot_attention_params, plot_error_data, plot_weight_metrics)

if __name__ == '__main__':
    source_path = os.path.join('logs', 'EXPERIEMNT NUMBER')
    experiments = [d for d in os.listdir(source_path) if os.path.isdir(os.path.join(data_path, d))]
    for experiment in experiments:
        seeds = [d for d in os.listdir(os.path.join(data_path, experiment)) if os.path.isdir(os.path.join(data_path, experiment, d)) and d.startswith('seed_')]
        for seed in seeds:
            data_path = os.path.join(source_path,experiment, seed) #path to directory that has data.npz
            save_path = os.path.join(source_path,experiment,seed) #path to directory where you want the plots to be saved

            plot_error_data(data_path = data_path, save_path =save_path)
            plot_attention_params(data_path=data_path, save_path=save_path)
            data_paths.append(data_path)


    
        plot_multiple_runs(data_paths, save_dir=task_path, final_figures_dir=final_task_path)
        plot_mean_attn_params(data_paths, save_dir=task_path, final_figures_dir=final_task_path)




    linear_path = os.path.join('logs', 'linear_discounted_train', 'linear_good')
    nonlinear_path = os.path.join('logs', 'nonlinear_discounted_train', 'nonlinear_good')
    linear_tasks = [d for d in os.listdir(linear_path) if os.path.isdir(os.path.join(linear_path, d))]
    nonlinear_tasks = [d for d in os.listdir(nonlinear_path) if os.path.isdir(os.path.join(nonlinear_path, d))]

    # directory structure is /run / linear/nonlinear / seed


    final_plot_path = os.path.join('logs', 'final_plots') 
    # Linear tasks
    for task in linear_tasks:
        task_path = os.path.join(linear_path, task)
        final_task_path = os.path.join(final_plot_path, "linear",task)
        if not os.path.exists(final_task_path):
            os.makedirs(final_task_path)
        data_paths = []
        seeds = []


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