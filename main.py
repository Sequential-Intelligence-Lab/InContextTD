import datetime
import os
from argparse import ArgumentParser, Namespace

from experiment.train import train
from experiment.plotter import ( load_data,
                                plot_attention_params, plot_error_data,
                                plot_mean_attn_params,
                                process_log)

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
                        help='total number of MDPs for training ', default=4_000)
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
                        help='random seed', default=list(range(1,30)))
    parser.add_argument('--save_dir', type=str,
                        help='directory to save logs', default=None)
    parser.add_argument('--suffix', type=str,
                        help='suffix to add to the save directory', default=None)
    parser.add_argument('--gen_gif',
                        help='generate a GIF for the evolution of weights',
                        action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='print training details')

    args: Namespace = parser.parse_args()
    if args.save_dir:
        save_dir = args.save_dir
    else:
        start_time = datetime.datetime.now()
        sub_dir = 'linear' if args.linear else 'nonlinear'
        save_dir = os.path.join('./logs',
                                f"{sub_dir}_train",
                                start_time.strftime("%Y-%m-%d-%H-%M-%S"))
    if args.suffix:
        save_dir += f'_{args.suffix}'

    if args.verbose:
        if args.linear:
            print(
                f"Training a linear {args.mode} transformer of {args.num_layers} layer(s).")
        else:
            print(
                f"Training a nonlinear {args.mode} transformer of {args.num_layers} layer(s) with {args.activation} activation.")
        print(f"Feature dimension: {args.dim_feature}")
        print(f"Context length: {args.context_length}")
        print(f"Number of states in the MDP: {args.num_states}")
        print(f"Discount factor: {args.gamma}")
        print(f"Eligibility trace decay rate: {args.lmbd}")
        tf_v = 'representable' if args.sample_weight else 'unrepresentable'
        print(f"Value function is {tf_v} by the features.")
        print(f"Number of MDPs for training: {args.n_mdps}")
        print(f'Number of mini-batches per MDP: {args.n_batch_per_mdp}')
        print(f'Mini-batch size: {args.batch_size}')
        print(
            f'Total number of prompts for training: {args.n_mdps * args.n_batch_per_mdp * args.batch_size}')
        print(f'Learning rate: {args.lr}')
        print(f'Regularization term: {args.weight_decay}')
        print(f'Logging interval: {args.log_interval}')
        print(f'Save directory: {save_dir}')
        print(f'Random seeds: {",".join(map(str, args.seed))}')

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
            train_args['activation'] = 'identity'
        else:
            train_args['activation'] = 'softmax'
            
        train(**train_args)

        log, hyperparams = load_data(data_dir)
        xs, error_log, attn_params = process_log(log)
        l_tf = args.num_layers if args.mode == 'sequential' else 1
        plot_error_data(xs, error_log, save_dir=data_dir, params=hyperparams)
        plot_attention_params(xs, attn_params, save_dir=data_dir)
    plot_mean_attn_params(data_dirs, save_dir=save_dir)
