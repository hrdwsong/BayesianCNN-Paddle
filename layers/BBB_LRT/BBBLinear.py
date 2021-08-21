import sys
sys.path.append("..")

import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import create_parameter

from metrics import calculate_kl as KL_DIV
from ..misc import ModuleWrapper


class BBBLinear(ModuleWrapper):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BBBLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

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

        self.W_mu = create_parameter(shape=[in_features, out_features],
                                     dtype='float32',
                                     default_initializer=nn.initializer.Normal(mean=self.posterior_mu_initial[0],
                                                                               std=self.posterior_mu_initial[1])
                                     )  # 确认下是否需要交换输入输出顺序
        self.W_rho = create_parameter(shape=[in_features, out_features],
                                      dtype='float32',
                                      default_initializer=nn.initializer.Normal(mean=self.posterior_rho_initial[0],
                                                                                std=self.posterior_rho_initial[1])
                                      )
        if self.use_bias:
            self.bias_mu = create_parameter(shape=[out_features], dtype='float32',
                                            default_initializer=nn.initializer.Normal(mean=self.posterior_mu_initial[0],
                                                                                      std=self.posterior_mu_initial[1])
                                            )
            self.bias_rho = create_parameter(shape=[out_features], dtype='float32',
                                             default_initializer=nn.initializer.Normal(
                                                 mean=self.posterior_rho_initial[0],
                                                 std=self.posterior_rho_initial[1])
                                             )
        else:
            self.add_parameter('bias_mu', None)
            self.add_parameter('bias_rho', None)

        # self.reset_parameters()

    # def reset_parameters(self):
    #     self.W_mu.data.normal_(*self.posterior_mu_initial)
    #     self.W_rho.data.normal_(*self.posterior_rho_initial)
    #
    #     if self.use_bias:
    #         self.bias_mu.data.normal_(*self.posterior_mu_initial)
    #         self.bias_rho.data.normal_(*self.posterior_rho_initial)

    def forward(self, x, sample=True):

        self.W_sigma = paddle.log1p(paddle.exp(self.W_rho))
        if self.use_bias:
            self.bias_sigma = paddle.log1p(paddle.exp(self.bias_rho))
            bias_var = self.bias_sigma ** 2
        else:
            self.bias_sigma = bias_var = None

        act_mu = F.linear(x, self.W_mu, self.bias_mu)
        act_var = 1e-16 + F.linear(x ** 2, self.W_sigma ** 2, bias_var)
        act_std = paddle.sqrt(act_var)

        if self.training or sample:
            eps = paddle.normal(0, 1, shape=act_mu.shape)
            return act_mu + act_std * eps
        else:
            return act_mu

    def kl_loss(self):
        kl = KL_DIV(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += KL_DIV(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl
