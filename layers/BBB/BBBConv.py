import sys

sys.path.append("..")

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import create_parameter

from metrics import calculate_kl as KL_DIV
from ..misc import ModuleWrapper


class BBBConv2d(ModuleWrapper):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True, priors=None):

        super(BBBConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if priors is None:
            priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),
                'posterior_rho_initial': (-3, 0.1),
            }
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.posterior_mu_initial = priors['posterior_mu_initial']
        self.posterior_rho_initial = priors['posterior_rho_initial']

        self.W_mu = create_parameter(shape=[out_channels, in_channels, *self.kernel_size],
                                     dtype='float32',
                                     default_initializer=nn.initializer.Normal(mean=self.posterior_mu_initial[0],
                                                                               std=self.posterior_mu_initial[1])
                                     )  # 确认下是否需要交换输入输出顺序
        self.W_rho = create_parameter(shape=[out_channels, in_channels, *self.kernel_size],
                                      dtype='float32',
                                      default_initializer=nn.initializer.Normal(mean=self.posterior_rho_initial[0],
                                                                                std=self.posterior_rho_initial[1])
                                      )

        if self.use_bias:
            self.bias_mu = create_parameter(shape=[out_channels], dtype='float32',
                                            default_initializer=nn.initializer.Normal(mean=self.posterior_mu_initial[0],
                                                                                      std=self.posterior_mu_initial[1])
                                            )
            self.bias_rho = create_parameter(shape=[out_channels], dtype='float32',
                                             default_initializer=nn.initializer.Normal(mean=self.posterior_rho_initial[0],
                                                                                      std=self.posterior_rho_initial[1])
                                             )
        else:
            self.add_parameter('bias_mu', None)
            self.add_parameter('bias_rho', None)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     # self.W_mu.data.normal_(*self.posterior_mu_initial)
    #     # self.W_rho.data.normal_(*self.posterior_rho_initial)
    #     self.W_mu = paddle.normal(mean=self.posterior_mu_initial[0], std=self.posterior_mu_initial[1],
    #                               shape=self.W_mu.shape)
    #     self.W_rho = paddle.normal(mean=self.posterior_rho_initial[0], std=self.posterior_rho_initial[1],
    #                                shape=self.W_rho.shape)
    #
    #     if self.use_bias:
    #         # self.bias_mu.data.normal_(*self.posterior_mu_initial)
    #         # self.bias_rho.data.normal_(*self.posterior_rho_initial)
    #         self.bias_mu.data = paddle.normal(mean=self.posterior_mu_initial[0], std=self.posterior_mu_initial[1],
    #                                           shape=self.bias_mu.data.shape)
    #         self.bias_rho.data = paddle.normal(mean=self.posterior_rho_initial[0], std=self.posterior_rho_initial[1],
    #                                            shape=self.bias_rho.data.shape)

    def forward(self, input, sample=True):
        if self.training or sample:
            # W_eps = paddle.empty(self.W_mu.shape).normal_(0, 1).to(self.device)
            W_eps = paddle.normal(0, 1, shape=self.W_mu.shape)
            self.W_sigma = paddle.log1p(paddle.exp(self.W_rho))  # log(x+1)
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                # bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).to(self.device)
                bias_eps = paddle.normal(0, 1, shape=self.bias_mu.shape)
                self.bias_sigma = paddle.log1p(paddle.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv2d(input, weight, bias=bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
