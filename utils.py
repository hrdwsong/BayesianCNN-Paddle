import os
import paddle
import numpy as np
from paddle.nn import functional as F

import config_bayesian as cfg


# cifar10 classes
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


def logmeanexp(x, axis=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    
    if axis is None:
        x, axis = x.reshape([-1]), 0
    x_max = paddle.max(x, axis, keepdim=True)
    x = x_max + paddle.log(paddle.mean(paddle.exp(x - x_max), axis, keepdim=True))
    return x if keepdim else x.squeeze(axis)

# check if dimension is correct

# def dimension_check(x, dim=None, keepdim=False):
#     if dim is None:
#         x, dim = x.view(-1), 0

#     return x if keepdim else x.squeeze(dim)


def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_array_to_file(numpy_array, filename):
    file = open(filename, 'a')
    shape = " ".join(map(str, numpy_array.shape))
    np.savetxt(file, numpy_array.flatten(), newline=" ", fmt="%.3f")
    file.write("\n")
    file.close()
