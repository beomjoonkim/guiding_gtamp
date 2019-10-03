import socket
import pickle
import numpy as np

from genetic_algorithm.cmaes import genetic_algorithm
from PlaceAdMonWithPose import PlaceAdmonWithPose

if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'


def slice_pick_pose_from_action(x):
    return x[:, :4]


def slice_place_pose_from_action(x):
    return x[:, 4:]


def slice_prepick_robot_pose_from_pose(x):
    return x[:, 4:]


def slice_object_pose_from_pose(x):
    return x[:, :4]


class CMAESAdversarialMonteCarloWithPose(PlaceAdmonWithPose):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlaceAdmonWithPose.__init__(self, dim_action, dim_collision, save_folder, tau, config)

    def train(self, states, poses, actions, sum_rewards, epochs=500, d_lr=1e-3, g_lr=1e-4):
        idxs = pickle.load(open('data_idxs_seed_%s' % self.seed, 'r'))
        train_idxs, test_idxs = idxs['train'], idxs['test']
        train_data, test_data = self.get_train_and_test_data(states, poses, actions, sum_rewards,
                                                             train_idxs, test_idxs)
        self.disc_mse_model.load_weights(self.save_folder + self.pretraining_file_name)

        states = train_data['states']
        poses = train_data['poses']
        actions = train_data['actions']
        sum_rewards = train_data['sum_rewards']

        n_data = len(train_data['actions'])
        batch_size = self.get_batch_size(n_data)
        # how to define the domain?
        domain = np.array([[0, -20, -1, -1], [10, 0, 1, 1]])

        # get data batch
        for i in range(1, epochs):
            batch_idxs = range(0, actions.shape[0], batch_size)
            for _ in batch_idxs:
                s_batch, pose_batch, a_batch, sum_rewards_batch = self.get_batch(states, poses, actions,
                                                                                 sum_rewards,
                                                                                 batch_size)
                # generate T sequence data from the cma-es
                cmaes_objective = lambda x: self.disc_mse_model.predict([a_batch, s_batch, pose_batch])

                max_x, max_y = genetic_algorithm(cmaes_objective, domain)
                # update discriminator
                raise NotImplementedError
