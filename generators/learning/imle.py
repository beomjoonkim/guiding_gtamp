'''
Code for Implicit Maximum Likelihood Estimation

This code implements the method described in the Implicit Maximum Likelihood
Estimation paper, which can be found at https://arxiv.org/abs/1809.09087

Copyright (C) 2018    Ke Li


This file is part of the Implicit Maximum Likelihood Estimation reference
implementation.

The Implicit Maximum Likelihood Estimation reference implementation is free
software: you can redistribute it and/or modify it under the terms of the GNU
Affero General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

The Implicit Maximum Likelihood Estimation reference implementation is
distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with the Dynamic Continuous Indexing reference implementation.  If
not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import sys

sys.path.append('./dci_code')
from dci import DCI
import collections

Hyperparams = collections.namedtuple('Hyperarams',
                                     'base_lr batch_size num_epochs decay_step decay_rate staleness num_samples_factor')
Hyperparams.__new__.__defaults__ = (None, None, None, None, None, None, None)


class ConvolutionalImplicitModel(nn.Module):
    def __init__(self, z_dim):
        super(ConvolutionalImplicitModel, self).__init__()
        self.z_dim = z_dim
        self.tconv1 = nn.ConvTranspose2d(64, 1024, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(1024)
        self.tconv2 = nn.ConvTranspose2d(1024, 128, 7, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.tconv3 = nn.ConvTranspose2d(128, 64, 4, 2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.tconv4 = nn.ConvTranspose2d(64, 1, 4, 2, padding=1, bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, z):
        z = self.relu(self.bn1(self.tconv1(z)))
        z = self.relu(self.bn2(self.tconv2(z)))
        z = self.relu(self.bn3(self.tconv3(z)))
        z = torch.sigmoid(self.tconv4(z))
        return z


class IMLE():
    def __init__(self, z_dim):
        self.z_dim = z_dim
        self.model = ConvolutionalImplicitModel(z_dim).cuda()
        self.dci_db = None

    def train(self, data_np, hyperparams, shuffle_data=True):
        loss_fn = nn.MSELoss().cuda()
        self.model.train()

        batch_size = hyperparams.batch_size
        num_batches = data_np.shape[0] // batch_size
        num_samples = num_batches * hyperparams.num_samples_factor # number of generated samples

        if shuffle_data:
            data_ordering = np.random.permutation(data_np.shape[0])
            data_np = data_np[data_ordering]

        data_flat_np = np.reshape(data_np, (data_np.shape[0], np.prod(data_np.shape[1:])))

        if self.dci_db is None:
            self.dci_db = DCI(np.prod(data_np.shape[1:]), num_comp_indices=2, num_simp_indices=7)

        for epoch in range(hyperparams.num_epochs):

            if epoch % hyperparams.decay_step == 0:
                lr = hyperparams.base_lr * hyperparams.decay_rate ** (epoch // hyperparams.decay_step)
                optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)

            # Data generation step - do you re-sample training steps?
            # It does not seem like it is selecting the samples S? It always uses the original dataset data_np
            if epoch % hyperparams.staleness == 0:
                z_np = np.empty((num_samples * batch_size, self.z_dim, 1, 1))
                samples_np = np.empty((num_samples * batch_size,) + data_np.shape[1:])
                for i in range(num_samples):
                    z = torch.randn(batch_size, self.z_dim, 1, 1).cuda()
                    samples = self.model(z)
                    z_np[i * batch_size:(i + 1) * batch_size] = z.cpu().data.numpy()
                    samples_np[i * batch_size:(i + 1) * batch_size] = samples.cpu().data.numpy()

                samples_flat_np = np.reshape(samples_np, (samples_np.shape[0], np.prod(samples_np.shape[1:])))

                self.dci_db.reset()
                self.dci_db.add(samples_flat_np, num_levels=2, field_of_view=10, prop_to_retrieve=0.002)
                nearest_indices, _ = self.dci_db.query(data_flat_np, num_neighbours=1, field_of_view=20,
                                                       prop_to_retrieve=0.02)
                nearest_indices = np.array(nearest_indices)[:, 0]

                z_np = z_np[nearest_indices]
                z_np += 0.01 * np.random.randn(*z_np.shape)

                del samples_np, samples_flat_np

            # z_np consists of z values whose value makes up the closest sample to each data point
            # But why do we do it this way?
            # We want to compare the real data point x_i with the closest generated point.
            # Call this point hat{x}_{sigma(i)}, and this was generated by the random noise, say, G_{theta}(z_i).
            # Once we have determined this point, how can we change G such that hat{x}_{sigma(i)} is closer to x_i?
            # We minimize || G_{theta)(z_i) - x_i||
            # What happens in the conditional case, where we condition on a random variable A?
            # The idea is to generate m different samples for each a in A, and then out of
            # m samples, find the one that is closest to the data point

            err = 0.
            # batch-gradient descent
            for i in range(num_batches):
                self.model.zero_grad()
                cur_z = torch.from_numpy(z_np[i * batch_size:(i + 1) * batch_size]).float().cuda()
                cur_data = torch.from_numpy(data_np[i * batch_size:(i + 1) * batch_size]).float().cuda()
                cur_samples = self.model(cur_z)
                loss = loss_fn(cur_samples, cur_data)
                loss.backward()
                err += loss.item()
                optimizer.step()

            print("Epoch %d: Error: %f" % (epoch, err / num_batches))


def main(*args):
    if len(args) > 0:
        device_id = int(args[0])
    else:
        device_id = 0

    torch.cuda.set_device(device_id)

    # train_data is of shape N x C x H x W, where N is the number of examples, C is the number of channels, H is the height and W is the width
    train_data = np.random.randn(128, 1, 28, 28)

    z_dim = 64
    imle = IMLE(z_dim)

    # Hyperparameters:

    # base_lr: Base learning rate
    # batch_size: Batch size
    # num_epochs: Number of epochs
    # decay_step: Number of epochs before learning rate decay
    # decay_rate: Rate of learning rate decay
    # staleness: Number of times to re-use nearest samples
    # num_samples_factor: Ratio of the number of generated samples to the number of real data examples
    imle.train(train_data,
               Hyperparams(base_lr=1e-3, batch_size=64, num_epochs=100, decay_step=25, decay_rate=1.0, staleness=5,
                           num_samples_factor=10))

    torch.save(imle.model.state_dict(), 'net_weights.pth')


if __name__ == '__main__':
    main(*sys.argv[1:])
