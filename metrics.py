import numpy as np
import paddle.nn.functional as F
from paddle import nn
import paddle


class ELBO(nn.Layer):
    def __init__(self, train_size):
        super(ELBO, self).__init__()
        self.train_size = train_size

    def forward(self, input, target, kl, beta):
        # assert not target.requires_grad
        assert target.stop_gradient
        return F.nll_loss(input, target, reduction='mean') * self.train_size + beta * kl  # 此loss函数公式是否合理？加号两边量级是否匹配？
        # return beta * kl - F.cross_entropy(input, target)
        # return F.cross_entropy(input, target)


def acc(outputs, targets):
    return np.mean(outputs.numpy().argmax(axis=1) == targets.numpy())


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * paddle.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


def get_beta(batch_idx, m, beta_type, epoch, num_epochs):
    if type(beta_type) is float:
        return beta_type

    if beta_type == "Blundell":
        beta = 2 ** (m - (batch_idx + 1)) / (2 ** m - 1)
    elif beta_type == "Soenderby":
        if epoch is None or num_epochs is None:
            raise ValueError('Soenderby method requires both epoch and num_epochs to be passed.')
        beta = min(epoch / (num_epochs // 4), 1)
    elif beta_type == "Standard":
        beta = 1 / m
    else:
        beta = 0
    return beta
