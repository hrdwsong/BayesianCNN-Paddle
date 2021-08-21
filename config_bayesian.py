############### Configuration file for Bayesian ###############
layer_type = 'lrt'  # 'bbb' or 'lrt' 改
activation_type = 'relu'  # 'softplus' or 'relu' 改
priors={
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

n_epochs = 200
lr_start = 0.001
num_workers = 0  # 改0
valid_size = 0.2  # 改：0.16667分割为50,000和10,000
batch_size = 256
train_ens = 1
valid_ens = 1
beta_type = 'Blundell'  # 'Blundell', 'Standard', etc. Use float for const value 改为文献形式‘Blundell’(代表公式（9）的pi)
