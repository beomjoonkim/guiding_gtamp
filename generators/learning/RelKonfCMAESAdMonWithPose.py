from keras.layers import *
from keras.layers.merge import Concatenate
from AdversarialPolicy import INFEASIBLE_SCORE
from generators.learning.AdversarialPolicy import AdversarialPolicy
from genetic_algorithm.voo import VOO
from genetic_algorithm.cmaes import genetic_algorithm

import pickle
import time


class RelKonfCMAESAdversarialMonteCarloWithPose(AdversarialPolicy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        AdversarialPolicy.__init__(self, dim_action, dim_collision, save_folder, tau)
        self.key_config_input = Input(shape=(615, 4, 1), name='konf', dtype='float32')
        self.goal_flag_input = Input(shape=(4,), name='goal_flag', dtype='float32')

        self.dim_poses = 8
        self.dim_collision = dim_collision
        self.action_input = Input(shape=(dim_action,), name='a', dtype='float32')  # action
        self.collision_input = Input(shape=dim_collision, name='s', dtype='float32')  # collision vector
        self.pose_input = Input(shape=(self.dim_poses,), name='pose', dtype='float32')  # collision vector
        self.weight_file_name = 'admonpose_seed_%d' % config.seed
        self.pretraining_file_name = 'pretrained_%d.h5' % config.seed
        self.seed = config.seed
        self.disc_output = self.construct_relevance_network()
        # todo
        #   1. create a model from disc output
        #   2.

    def construct_relevance_network(self):
        tiled_action = self.get_tiled_input(self.action_input)
        tiled_goal_flag = self.get_tiled_input(self.goal_flag_input)
        H_konf_action_goal_flag = Concatenate(axis=2)([self.key_config_input, tiled_action, tiled_goal_flag])
        dim_combined = H_konf_action_goal_flag.shape[2]._value
        H_relevance = self.create_conv_layers(H_konf_action_goal_flag, dim_combined, use_pooling=False,
                                              use_flatten=False)
        H_relevance = Conv2D(filters=1,
                             kernel_size=(1, 1),
                             strides=(1, 1),
                             activation='relu')(H_relevance)
        self.relevance_output = H_relevance

        H_col_relevance = Concatenate(axis=2)([self.collision_input, H_relevance])
        H_col_relevance = self.create_conv_layers(H_col_relevance, n_dim=4)

        dense_num = 64
        H_place = Dense(dense_num, activation='relu')(H_col_relevance)
        H_place = Dense(dense_num, activation='relu')(H_place)
        place_value = Dense(1, activation='linear',
                            kernel_initializer=self.initializer,
                            bias_initializer=self.initializer)(H_place)
        disc_output = place_value
        return disc_output

    def get_train_and_test_data(self, states, poses, rel_konfs, actions, sum_rewards, train_indices, test_indices):
        train = {'states': states[train_indices, :],
                 'poses': poses[train_indices, :],
                 'actions': actions[train_indices, :],
                 'rel_konfs': rel_konfs[train_indices, :],
                 'sum_rewards': sum_rewards[train_indices, :]}
        test = {'states': states[test_indices, :],
                'poses': poses[test_indices, :],
                'actions': actions[test_indices, :],
                'rel_konfs': rel_konfs[test_indices, :],
                'sum_rewards': sum_rewards[test_indices, :]}

        return train, test

    def get_batch(self, states, poses, rel_konfs, actions, sum_rewards, batch_size):
        indices = np.random.randint(0, actions.shape[0], size=batch_size)
        s_batch = np.array(states[indices, :])  # collision vector
        a_batch = np.array(actions[indices, :])
        pose_batch = np.array(poses[indices, :])
        konf_batch = np.array(rel_konfs[indices, :])
        sum_reward_batch = np.array(sum_rewards[indices, :])
        return s_batch, pose_batch, konf_batch, a_batch, sum_reward_batch

    def get_max_x(self, state, pose, rel_konfs):
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

    def train(self, states, poses, rel_konf, actions, sum_rewards, epochs=500, d_lr=1e-3, g_lr=1e-4):
        idxs = pickle.load(open('data_idxs_seed_%s' % self.seed, 'r'))
        train_idxs, test_idxs = idxs['train'], idxs['test']
        train_data, test_data = self.get_train_and_test_data(states, poses, actions, sum_rewards,
                                                             train_idxs, test_idxs)
        self.disc_mse_model.load_weights(self.save_folder + self.pretraining_file_name)

        states = train_data['states']
        poses = train_data['poses']
        actions = train_data['actions']
        sum_rewards = train_data['sum_rewards']
        rel_konfs = train_data['rel_konfs']

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
                s_batch, pose_batch, rel_pose_batch, a_batch, sum_rewards_batch = self.get_batch(states, poses,
                                                                                                 rel_konfs, actions,
                                                                                                 sum_rewards,
                                                                                                 batch_size)
                # generate T sequence data from the cma-es
                fake_actions = []
                max_ys = []
                for s, p in zip(s_batch, pose_batch):
                    max_x, max_y = self.get_max_x(s[None, :], p[None, :], rel_pose_batch)

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
