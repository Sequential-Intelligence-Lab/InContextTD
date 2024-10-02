import torch.nn as nn


def get_activation(activation: str) -> nn.Module:
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'softmax':
        return nn.Softmax(dim=1)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU()
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'identity':
        return nn.Identity()
    else:
        raise ValueError(f"Invalid activation function: {activation}")
