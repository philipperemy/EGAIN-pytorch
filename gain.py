'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
# import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from utils import binary_sampler, uniform_sampler, sample_batch_index, rmse_loss
from utils import normalization, renormalization, rounding


def d_loss(M, D_prob):
    return -torch.mean(M * torch.log(D_prob + 1e-8) + (1 - M) * torch.log(1. - D_prob + 1e-8))


def g_loss(M, D_prob, alpha, X, G_sample):
    return -torch.mean((1 - M) * torch.log(D_prob + 1e-8)) + \
           alpha * torch.mean((M * X - M * G_sample) ** 2) / torch.mean(M)


class Generator(nn.Module):
    def __init__(self, dim: int, h_dim: int):
        super(Generator, self).__init__()
        self.d_w1 = nn.Linear(dim * 2, h_dim)
        # nn.init.xavier_normal_(self.d_w1.weight)
        self.d_w2 = nn.Linear(h_dim, h_dim)
        # nn.init.xavier_normal_(self.d_w2.weight)
        self.d_w3 = nn.Linear(h_dim, dim)
        # nn.init.xavier_normal_(self.d_w3.weight)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        x = torch.cat([x, m], 1)
        x = self.relu(self.d_w1(x))
        # x += torch.normal(0.0, 0.01, size=x.shape)
        x = self.relu(self.d_w2(x))
        # x += torch.normal(0.0, 0.01, size=x.shape)
        x = self.sigmoid(self.d_w3(x))
        return x


class Discriminator(nn.Module):
    def __init__(self, dim: int, h_dim: int):
        super(Discriminator, self).__init__()
        self.d_w1 = nn.Linear(dim * 2, h_dim)
        # nn.init.xavier_normal_(self.d_w1.weight)
        self.d_w2 = nn.Linear(h_dim, h_dim)
        # nn.init.xavier_normal_(self.d_w2.weight)
        self.d_w3 = nn.Linear(h_dim, dim)
        # nn.init.xavier_normal_(self.d_w3.weight)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        x = torch.cat([x, h], 1)
        x = self.relu(self.d_w1(x))
        x = self.relu(self.d_w2(x))
        x = self.sigmoid(self.d_w3(x))
        return x


def gain(data_x, gain_parameters, ori_data_x, train_index, test_index):
    '''Impute missing values in data_x

    Args:
      - data_x: original data with missing values
      - gain_parameters: GAIN network parameters:
        - batch_size: Batch size
        - hint_rate: Hint rate
        - alpha: Hyper-parameter
        - iterations: Iterations

    Returns:
      - imputed_data: imputed data
    '''
    # Define mask matrix
    data_m = 1 - np.isnan(data_x)

    # System parameters
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    # Other parameters
    no, dim = data_x.shape

    no_train = len(train_index)

    # Hidden state dimensions
    h_dim = int(dim)

    # Normalization
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, nan=0)

    # pytorch.
    generator = Generator(dim, h_dim)
    discriminator = Discriminator(dim, h_dim)
    discriminator2 = Discriminator(dim, h_dim)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    discriminator_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.0001)
    discriminator2_optimizer = torch.optim.SGD(discriminator2.parameters(), lr=0.0001)
    for i in tqdm(range(iterations), desc='pytorch'):

        if i % 1000 == 0:
            test(data_m, data_x, dim, generator, no, norm_data_x, norm_parameters, ori_data_x, test_index)

        # Sample batch
        batch_idx = sample_batch_index(no_train, batch_size)
        X_mb = norm_data_x[train_index][batch_idx, :]
        M_mb = data_m[train_index][batch_idx, :]
        # Sample random vectors
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        # Sample hint vectors

        H_mb = M_mb * binary_sampler(hint_rate, batch_size, dim)
        H_mb_2 = M_mb * binary_sampler(hint_rate, batch_size, dim)

        # Combine random vectors with observed vectors
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        X_mb = torch.Tensor(X_mb)
        M_mb = torch.Tensor(M_mb)
        H_mb = torch.Tensor(H_mb)
        H_mb_2 = torch.Tensor(H_mb_2)

        G_sample = generator(X_mb, M_mb)
        Hat_X = X_mb * M_mb + G_sample * (1 - M_mb)
        D_prob = discriminator(Hat_X, H_mb)
        D_prob2 = discriminator2(Hat_X, H_mb_2)

        d_loss_value = d_loss(M_mb, D_prob)
        d_loss_value2 = d_loss(M_mb, D_prob2)
        y = torch.rand(1)
        g_loss_value = g_loss(M_mb, y * D_prob + (1 - y) * D_prob2, alpha, X_mb, G_sample)

        discriminator2_optimizer.zero_grad()
        discriminator_optimizer.zero_grad()
        generator_optimizer.zero_grad()

        g_loss_value.backward(retain_graph=True)
        d_loss_value.backward(retain_graph=True)
        d_loss_value2.backward(retain_graph=True)

        generator_optimizer.step()
        discriminator_optimizer.step()
        discriminator2_optimizer.step()


def test(data_m, data_x, dim, generator, no, norm_data_x, norm_parameters, ori_data_x, test_index):
    # Return imputed data
    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
    imputed_data = generator(torch.Tensor(X_mb), torch.Tensor(M_mb)).detach().numpy()
    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data
    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)
    # Rounding
    imputed_data = rounding(imputed_data, data_x)
    rmse, rmse_mean = rmse_loss(ori_data_x[test_index], imputed_data[test_index], data_m[test_index])
    rmse_full, rmse_full_mean = rmse_loss(ori_data_x, imputed_data, data_m)
    print(f'RMSE Performance (mean): {rmse_mean:.4f} (test), {rmse_full_mean:.4f} (full).')
    print(f'RMSE Performance: {rmse:.4f} (test), {rmse_full:.4f} (full).')
    return rmse
