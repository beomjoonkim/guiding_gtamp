from PlaceAdMonWithPose import PlaceAdmonWithPose
from AdversarialPolicy import INFEASIBLE_SCORE
from genetic_algorithm.voo import VOO
from genetic_algorithm.cmaes import genetic_algorithm

import os
import socket
import pickle
import numpy as np
import time

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

    def get_max_x(self, state, pose):
        domain = np.array([[0, 0, -1, -1], [1, 1, 1, 1]])
        objective = lambda action: float(self.disc_mse_model.predict([action[None, :], state, pose])[0, 0])
        is_cmaes = False
        n_evals = 50
        if is_cmaes:
            max_x, max_y = genetic_algorithm(objective, n_evals)
        else:
            voo = VOO(domain, 0.3, 'gaussian', 1)
            max_x, max_y = voo.optimize(objective, n_evals)
        return max_x, max_y

    def save_weights(self, additional_name=''):
        fdir = ROOTDIR + '/' + self.save_folder + '/'
        fname = self.weight_file_name + additional_name + '.h5'
        if not os.path.isdir(fdir):
            os.makedirs(fdir)

        self.disc.save_weights(fdir + fname)

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
        pretrain_mse = self.compute_pure_mse(test_data)

        # get data batch
        curr_tau = 1.0
        for i in range(1, epochs):
            batch_idxs = range(0, actions.shape[0], batch_size)
            stime = time.time()
            for batch_idx, _ in enumerate(batch_idxs):
                batch_stime = time.time()
                print "Batch progress %d / %d" % (batch_idx, len(batch_idxs))
                s_batch, pose_batch, a_batch, sum_rewards_batch = self.get_batch(states, poses, actions,
                                                                                 sum_rewards,
                                                                                 batch_size)
                # generate T sequence data from the cma-es
                fake_actions = []
                max_ys = []
                for s, p in zip(s_batch, pose_batch):
                    max_x, max_y = self.get_max_x(s[None, :], p[None, :])

                    fake_actions.append(max_x)
                    max_ys.append(max_y)

                fake_actions = np.array(fake_actions)

                # make their scores
                tau_values = np.tile(curr_tau, (batch_size * 2, 1))
                fake_action_q = np.ones((batch_size, 1)) * INFEASIBLE_SCORE  # marks fake data
                real_action_q = sum_rewards_batch.reshape((batch_size, 1))
                batch_a = np.vstack([fake_actions, a_batch])
                batch_s = np.vstack([s_batch, s_batch])
                batch_rp = np.vstack([pose_batch, pose_batch])
                batch_scores = np.vstack([fake_action_q, real_action_q])

                self.disc.fit({'a': batch_a, 's': batch_s, 'pose': batch_rp, 'tau': tau_values},
                              batch_scores,
                              epochs=1,
                              verbose=False)
                batch_time_taken = time.time() - batch_stime
                print "Batch time", batch_time_taken

            posttrain_mse = self.compute_pure_mse(test_data)
            drop_in_mse = pretrain_mse - posttrain_mse
            self.save_weights(additional_name='_epoch_%d_drop_in_mse_%.5f' % (i, drop_in_mse))
            time_taken = time.time() - stime
            print "Epoch time", time_taken
