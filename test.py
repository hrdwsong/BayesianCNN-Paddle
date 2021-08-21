from __future__ import print_function

import os
import argparse
import paddle
import numpy as np
from paddle.nn import functional as F

import data
import metrics
import config_bayesian as cfg
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet
from models.BayesianModels.BayesianOriginNet import BBBOriginNet

# 开启0号GPU训练
use_gpu = True
device = paddle.set_device('gpu:0') if use_gpu else paddle.set_device('cpu')


def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == '3conv3fc'):
        return BBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    elif(net_type == 'originet'):
        return BBBOriginNet(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def test_model(net, testloader, num_ens=1):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    accs = []

    for i, (inputs, labels) in enumerate(testloader):
        inputs = inputs / 128.0  # 与论文对齐
        labels = labels[:, 0]
        kl = 0.0

        net_out, _kl = net(inputs)
        kl += _kl
        outputs = F.log_softmax(net_out, axis=1)

        accs.append(metrics.acc(outputs, labels))

    return np.mean(accs)


def run(dataset, net_type):

    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type)

    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pdparams'
    net_stat = paddle.load(ckpt_name)
    net.set_state_dict(net_stat)

    test_acc = test_model(net, test_loader, num_ens=valid_ens)
    print('Testing Accuracy: {:.4f}'.format(test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Paddle Bayesian Model Testing")
    parser.add_argument('--net_type', default='3conv3fc', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)
