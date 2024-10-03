from typing import Dict

import numpy as np
import torch

from theory_verification.model import DiscountedTDTransformer, RGTransformer, AVGREWTDTransformer
from theory_verification.prompt import Prompt


def verify(d: int, n: int, l: int) -> np.ndarray:
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


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    print(verify(10, 5, 3))
