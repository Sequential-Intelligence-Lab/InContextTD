import datetime
import os
from argparse import ArgumentParser, Namespace

from experiment.linear_discounted_train import train as linear_train
from experiment.nonlinear_discounted_train import train as nonlinear_train
from experiment.plotter import (compute_weight_metrics,
                                generate_attention_params_gif, load_data,
                                plot_attention_params, plot_error_data,
                                plot_mean_attn_params, plot_multiple_runs,
                                plot_weight_metrics, process_log)
from experiment.utils import get_hardcoded_P, get_hardcoded_Q

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--linear', help='specify whether to train a linear or nonlinear transformer',
                        action='store_true')
    parser.add_argument('-d', '--dim_feature', type=int,
                        help='feature dimension', default=4)
    parser.add_argument('-s', '--num_states', type=int,
                        help='number of states', default=10)
    parser.add_argument('-n', '--context_length', type=int,
                        help='context length', default=30)
    parser.add_argument('-l', '--num_layers', type=int,
                        help='number of layers', default=3)
    parser.add_argument('--gamma', type=float,
                        help='discount factor', default=0.9)
    parser.add_argument('--lmbd', type=float,
                        help='eligibility trace decay rate', default=0.0)
    parser.add_argument('--activation', type=str,
                        help='activation function for the transformer', default='softmax')
    parser.add_argument('--sample_weight', action='store_true',
                        help='sample a random true weight vector, such that the value function is fully representable by the features')
    parser.add_argument('--n_mdps', type=int,
                        help='total number of MDPs for training ', default=5_000)
    parser.add_argument('--batch_size', type=int,
                        help='mini batch size', default=64)
    parser.add_argument('--n_batch_per_mdp', type=int,
                        help='number of mini-batches sampled from each MDP', default=5)
    parser.add_argument('--lr', type=float,
                        help='learning rate', default=0.001)
    parser.add_argument('--weight_decay', type=float,
                        help='regularization term', default=1e-6)
    parser.add_argument('--log_interval', type=int,
                        help='logging interval', default=10)
    parser.add_argument('--mode', type=str,
                        help='training mode: auto-regressive or sequential', default='auto', choices=['auto', 'sequential'])
    parser.add_argument('--seed', type=int, nargs='+',
                        help='random seed', default=[1, 2, 3, 4, 5])
    parser.add_argument('--save_dir', type=str,
                        help='directory to save logs', default=None)
    parser.add_argument('--suffix', type=str,
                        help='suffix to add to the save directory', default=None)
    parser.add_argument('--gen_gif',
                        help='generate a GIF for the evolution of weights',
                        action='store_true')

    args: Namespace = parser.parse_args()
    if args.save_dir:
        save_dir = args.save_dir
    else:
        start_time = datetime.datetime.now()
        sub_dir = 'linear' if args.linear else 'nonlinear'
        save_dir = os.path.join('./logs',
                                f"{sub_dir}_discounted_train",
                                start_time.strftime("%Y-%m-%d-%H-%M-%S"))
    if args.suffix:
        save_dir += f'_{args.suffix}'

    data_dirs = []
    for seed in args.seed:
        data_dir = os.path.join(save_dir, f'seed_{seed}')
        data_dirs.append(data_dir)
        train_args = dict(
            d=args.dim_feature,
            s=args.num_states,
            n=args.context_length,
            l=args.num_layers,
            gamma=args.gamma,
            lmbd=args.lmbd,
            sample_weight=args.sample_weight,
            mode=args.mode,
            lr=args.lr,
            weight_decay=args.weight_decay,
            n_mdps=args.n_mdps,
            mini_batch_size=args.batch_size,
            n_batch_per_mdp=args.n_batch_per_mdp,
            log_interval=args.log_interval,
            save_dir=data_dir,
            random_seed=seed
        )
        if args.linear:
            linear_train(**train_args)
        else:
            train_args['activation'] = args.activation
            nonlinear_train(**train_args)

        log, hyperparams = load_data(data_dir)
        xs, error_log, attn_params = process_log(log)
        l_tf = args.num_layers if args.mode == 'sequential' else 1
        plot_error_data(xs, error_log, save_dir=data_dir, params=hyperparams)
        plot_attention_params(xs, attn_params, save_dir=data_dir)
        if args.gen_gif:
            generate_attention_params_gif(xs, l_tf, attn_params, data_dir)
        if args.linear:
            P_true = get_hardcoded_P(args.dim_feature)
            Q_true = get_hardcoded_Q(args.dim_feature)
            P_metrics, Q_metrics = compute_weight_metrics(attn_params, P_true,
                                                          Q_true, args.dim_feature)
            plot_weight_metrics(xs, l_tf, P_metrics, Q_metrics,
                                data_dir, hyperparams)
    plot_multiple_runs(data_dirs, save_dir=save_dir)
    plot_mean_attn_params(data_dirs, save_dir=save_dir)
