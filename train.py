from __future__ import print_function

import os
import argparse

# import torch
import paddle
import numpy as np
from paddle.optimizer import Adam
from paddle.nn import functional as F

import data
import utils
import metrics
import config_bayesian as cfg
from models.BayesianModels.BayesianOriginNet import BBBOriginNet
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet

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


def train_model(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):  # i从1开始计数

        optimizer.clear_grad()
        inputs = inputs / 128.0  # 与论文对齐
        labels = labels[:, 0]
        # outputs = paddle.zeros([inputs.shape[0], net.num_classes, num_ens])

        kl = 0.0
        # for j in range(num_ens):
        net_out, _kl = net(inputs)
        kl += _kl
        # outputs[:, :, j] = F.log_softmax(net_out, axis=1)
        # outputs = F.softmax(net_out, axis=1)
        outputs = F.log_softmax(net_out, axis=1)

        # ---调试代码
        # loss = paddle.nn.functional.cross_entropy(net_out, labels)
        # loss.backward()
        # optimizer.step()
        # accs.append(metrics.acc(net_out, labels))
        # ---

        kl = kl / num_ens
        kl_list.append(kl.item())
        # log_outputs = utils.logmeanexp(outputs, axis=2)

        beta = metrics.get_beta(i-1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(outputs, labels))
        training_loss += loss.item()
    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    valid_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        # inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs / 128.0  # 与论文对齐
        labels = labels[:, 0]
        outputs = paddle.zeros([inputs.shape[0], net.num_classes, num_ens])
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, axis=1)

        log_outputs = utils.logmeanexp(outputs, axis=2)

        beta = metrics.get_beta(i-1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels))

    return valid_loss/len(validloader), np.mean(accs)


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

    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pdparams'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(len(trainset))
    lr_scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=lr_start, patience=6, verbose=True)
    optimizer = Adam(learning_rate=lr_scheduler, parameters=net.parameters())  # 暂时不用lr decay方法，后续提高精度，找现有API

    valid_loss_max = np.Inf
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        lr_scheduler.step(valid_loss)  # 改为Variable(valid_loss)
        # optimizer.current_step_lr()
        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc, train_kl))

        # save model if validation accuracy has increased
        if valid_loss <= valid_loss_max:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_max, valid_loss))
            paddle.save(net.state_dict(), ckpt_name)
            valid_loss_max = valid_loss

        test_acc = test_model(net, test_loader, num_ens=valid_ens)
        print('Testing Accuracy: {:.4f}'.format(test_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Paddle Bayesian Model Training")
    parser.add_argument('--net_type', default='3conv3fc', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    # Random seed
    seed = 0
    paddle.seed(seed)
    np.random.seed(seed)

    run(args.dataset, args.net_type)
