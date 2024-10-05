import os
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
from tqdm import tqdm

from verification.model import (AVGREWTDTransformer, DiscountedTDTransformer,
                                RGTransformer)
from verification.prompt import Prompt

torch.set_default_dtype(torch.float64)


def verify_one_trial(d: int, n: int, l: int) -> np.ndarray:
    '''
    d: feature dimension
    n: context length
    l: number of layers (updates)
    '''
    pro_discounted = Prompt(d, n, 0.9)
    pro_avg = Prompt(d, n, 1.0)

    td0_tf = DiscountedTDTransformer(l, d, n, 0.0)
    tdlmbd_tf = DiscountedTDTransformer(l, d, n, 0.5)
    rg_tf = RGTransformer(l, d, n)
    avg_rew_td_tf = AVGREWTDTransformer(l, d, n)
    td0_tf_value = td0_tf(pro_discounted.z()).numpy()
    tdlmbd_tf_value = tdlmbd_tf(pro_discounted.z()).numpy()
    rg_tf_value = rg_tf(pro_discounted.z()).numpy()
    avg_rew_td_tf_value = avg_rew_td_tf(pro_avg.z_avg_rew()).numpy()

    w_td0 = torch.zeros((d, 1))
    w_tdlmbd = torch.zeros((d, 1))
    w_rg = torch.zeros((d, 1))
    w_avg_rew_td = torch.zeros((d, 1))
    td0_value = []
    tdlmbd_value = []
    rg_value = []
    avg_rew_rd_value = []
    for i in range(l):
        w_td0, v_td0 = pro_discounted.td_update(w_td0, td0_tf.Cs[i], 0.0)
        td0_value.append(v_td0)

        w_tdlmbd, v_tdlmbd = pro_discounted.td_update(w_tdlmbd,
                                                      tdlmbd_tf.Cs[i], 0.5)
        tdlmbd_value.append(v_tdlmbd)

        w_rg, v_rg = pro_discounted.rg_update(w_rg, rg_tf.Cs[i])
        rg_value.append(v_rg)

        w_avg_rew_td, v_avg_rew_td = pro_avg.avg_rew_td_update(w_avg_rew_td,
                                                               avg_rew_td_tf.Cs[i])
        avg_rew_rd_value.append(v_avg_rew_td)

    td0_value = np.array(td0_value)
    tdlmbd_value = np.array(tdlmbd_value)
    rg_value = np.array(rg_value)
    avg_rew_rd_value = np.array(avg_rew_rd_value)

    return dict(td0=np.absolute(td0_tf_value - td0_value),
                tdlambda=np.absolute(tdlmbd_tf_value - tdlmbd_value),
                rg=np.absolute(rg_tf_value - rg_value),
                avg_rew_td=np.absolute(avg_rew_td_tf_value - avg_rew_rd_value))


def verify(d: int, n: int, l: int, num_trials: int, save_dir: str):
    '''
    d: feature dimension
    n: context length
    l: number of layers (updates)
    num_trials: number of trials
    save_dir: directory to save verification result
    '''
    td0_error = []
    tdlmbd_error = []
    rg_error = []
    avg_rew_td_error = []
    for _ in tqdm(range(num_trials)):
        error = verify_one_trial(d, n, l)
        td0_error.append(error['td0'])
        tdlmbd_error.append(error['tdlambda'])
        rg_error.append(error['rg'])
        avg_rew_td_error.append(error['avg_rew_td'])

    td0_error = np.array(td0_error)
    tdlmbd_error = np.array(tdlmbd_error)
    rg_error = np.array(rg_error)
    avg_rew_td_error = np.array(avg_rew_td_error)

    np.save(os.path.join(save_dir, 'discounted_td.npy'), td0_error)
    np.save(os.path.join(save_dir, 'discounted_td_lambda.npy'), tdlmbd_error)
    np.save(os.path.join(save_dir, 'residual_gradient.npy'), rg_error)
    np.save(os.path.join(save_dir, 'avg_reward_td.npy'), avg_rew_td_error)


if __name__ == '__main__':
    from verification.plot import plot_error
    from utils import set_seed

    parser = ArgumentParser()
    parser.add_argument('-d', '--dim_feature', type=int,
                        help='feature dimension', default=3)
    parser.add_argument('-n', '--context_length', type=int,
                        help='context length', default=100)
    parser.add_argument('-l', '--num_layers', type=int,
                        help='number of transformer layers', default=40)
    parser.add_argument('--num_trials', type=int,
                        help='number of trials', default=30)
    parser.add_argument('--seed', type=int,
                        help='random seed', default=42)
    parser.add_argument('--save_dir', type=str,
                        help='directory to save verification result', default='logs')
    args: Namespace = parser.parse_args()

    set_seed(args.seed)

    save_dir = os.path.join(args.save_dir, 'theory')
    os.makedirs(save_dir, exist_ok=True)

    verify(args.dim_feature,
           args.context_length,
           args.num_layers,
           args.num_trials,
           save_dir)
    plot_error(save_dir)
