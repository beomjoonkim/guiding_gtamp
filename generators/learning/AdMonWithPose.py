from keras.layers import *
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.optimizers import *
from keras import backend as K
from keras import initializers

import time
import numpy as np
import socket
import os
import pickle

from AdversarialPolicy import tau_loss, G_loss, INFEASIBLE_SCORE
from AdversarialPolicy import AdversarialPolicy

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


def noise(n, z_size):
    # todo use the uniform over the entire action space here
    # return np.random.normal(size=(n, z_size)).astype('float32')
    domain = np.array([[0, -20, -1, -1], [10, 0, 1, 1]])
    return np.random.uniform(low=domain[0], high=domain[1], size=(n, 4))


class AdversarialMonteCarloWithPose(AdversarialPolicy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        AdversarialPolicy.__init__(self, dim_action, dim_collision, save_folder, tau)
        self.dim_poses = 8
        self.dim_collision = dim_collision
        self.action_input = Input(shape=(dim_action,), name='a', dtype='float32')  # action
        self.collision_input = Input(shape=dim_collision, name='s', dtype='float32')  # collision vector
        self.pose_input = Input(shape=(self.dim_poses,), name='pose', dtype='float32')  # collision vector
        self.a_gen, self.disc_mse_model, self.disc, self.DG, = self.create_models()
        self.weight_file_name = 'admonpose_seed_%d' % config.seed
        self.pretraining_file_name = 'pretrained_%d.h5' % config.seed
        self.seed = config.seed
        self.train_indices = None
        self.test_indices = None

    def get_train_and_test_data(self, states, poses, actions, sum_rewards, train_indices, test_indices):
        train = {'states': states[train_indices, :],
                 'poses': poses[train_indices, :],
                 'actions': actions[train_indices, :],
                 'sum_rewards': sum_rewards[train_indices, :]}
        test = {'states': states[test_indices, :],
                'poses': poses[test_indices, :],
                'actions': actions[test_indices, :],
                'sum_rewards': sum_rewards[test_indices, :]}

        return train, test

    def pretrain_discriminator_with_mse(self, states, poses, actions, sum_rewards):
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        self.train_data, self.test_data = self.get_train_and_test_data(states, poses, actions, sum_rewards,
                                                                       train_idxs, test_idxs)
        callbacks = self.create_callbacks_for_pretraining()

        pre_mse = self.compute_pure_mse(self.test_data)
        self.disc_mse_model.fit([self.train_data['actions'], self.train_data['states'],
                                 self.train_data['poses']], self.train_data['sum_rewards'], batch_size=32,
                                epochs=500,
                                verbose=2,
                                callbacks=callbacks,
                                validation_split=0.1)
        post_mse = self.compute_pure_mse(self.test_data)

        print "Pre-and-post test errors", pre_mse, post_mse

    def create_mse_model(self):
        mse_model = Model(inputs=[self.action_input, self.collision_input, self.pose_input],
                          outputs=self.disc_output,
                          name='disc_output')
        mse_model.compile(loss='mse', optimizer=self.opt_D)
        return mse_model

    def create_models(self):
        disc = self.create_discriminator()
        mse_model = self.create_mse_model()
        a_gen, a_gen_output = self.create_generator()

        for l in disc.layers:
            l.trainable = False
            # for some obscure reason, disc weights still get updated when self.disc.fit is called
            # I speculate that this has to do with the status of the layers at the time it was compiled
        DG_output = disc([a_gen_output, self.collision_input, self.pose_input, self.collision_input])
        DG = Model(inputs=[self.noise_input, self.collision_input, self.pose_input], outputs=[DG_output])
        DG.compile(loss={'disc_output': G_loss},
                   optimizer=self.opt_G,
                   metrics=[])
        return a_gen, mse_model, disc, DG

    def save_weights(self, additional_name=''):
        fdir = ROOTDIR + '/' + self.save_folder + '/'
        fname = self.weight_file_name + additional_name + '.h5'
        if not os.path.isdir(fdir):
            os.makedirs(fdir)

        self.a_gen.save_weights(fdir + fname)

    def load_weights(self, additional_name=''):
        self.a_gen.load_weights(self.save_folder + '/' + self.weight_file_name + additional_name + '.h5')

    def reset_weights(self, init=True):
        if init:
            self.a_gen.load_weights('a_gen_init.h5')
            self.disc.load_weights('disc_init.h5')
        else:
            self.a_gen.load_weights(self.save_folder + '/a_gen.h5')
            self.disc.load_weights(self.save_folder + '/disc.h5')

    def get_prepick_robot_pose(self):
        prepick_robot_pose = Lambda(slice_prepick_robot_pose_from_pose)(self.pose_input)
        prepick_robot_pose = RepeatVector(self.n_key_confs)(prepick_robot_pose)
        prepick_robot_pose = Reshape((self.n_key_confs, 4, 1))(prepick_robot_pose)
        return prepick_robot_pose

    def get_abs_obj_pose(self):
        abs_obj_pose = Lambda(slice_object_pose_from_pose)(self.pose_input)
        abs_obj_pose = RepeatVector(self.n_key_confs)(abs_obj_pose)
        abs_obj_pose = Reshape((self.n_key_confs, 4, 1))(abs_obj_pose)
        return abs_obj_pose

    def make_tiled_abs_obj_pose_and_pick_output(self, pick_output):
        dense_num = 64
        abs_obj_pose = Lambda(slice_object_pose_from_pose)(self.pose_input)
        H = Concatenate(axis=-1)([abs_obj_pose, pick_output])
        for _ in range(2):
            H = Dense(dense_num, activation='relu')(H)

        H = RepeatVector(self.n_key_confs)(H)
        H = Reshape((self.n_key_confs, dense_num, 1))(H)
        return H

    def create_a_gen_output(self):
        dense_num = 64

        # Collision vector
        C_H = Reshape((self.n_key_confs, self.dim_collision[1], 1))(self.collision_input)

        # process state var relevant to predicting pick
        prepick_robot_pose = self.get_prepick_robot_pose()
        H_col_robot_pose_pick = Concatenate(axis=2)([prepick_robot_pose, C_H])
        H_pick = self.create_conv_layers(H_col_robot_pose_pick, 6)
        H_pick = Dense(dense_num, activation='relu')(H_pick)
        H_pick = Dense(dense_num, activation='relu')(H_pick)
        H_pick = Concatenate(axis=-1)([H_pick, self.noise_input])
        pick_output = Dense(4, activation='linear')(H_pick)

        tiled_abs_pose_and_pick_output = self.make_tiled_abs_obj_pose_and_pick_output(pick_output)

        H_col_abs_obj_pose_place = Concatenate(axis=2)([tiled_abs_pose_and_pick_output, C_H])
        H_place = self.create_conv_layers(H_col_abs_obj_pose_place, 64 + 2)
        H_place = Dense(dense_num, activation='relu')(H_place)
        H_place = Dense(dense_num, activation='relu')(H_place)
        H_place = Concatenate(axis=-1)([H_place, self.noise_input])
        place_output = Dense(8, activation='linear')(H_place)

        # a_gen_output = Concatenate(axis=-1)([pick_output, place_output])
        a_gen_output = place_output
        return a_gen_output

    def create_generator(self):
        a_gen_output = self.create_a_gen_output()
        a_gen = Model(inputs=[self.noise_input, self.collision_input, self.pose_input], outputs=a_gen_output)
        return a_gen, a_gen_output

    def get_disc_output_with_preprocessing_layers(self):
        dense_num = 64

        # Collision vector
        C_H = Reshape((self.n_key_confs, self.dim_collision[1], 1))(self.collision_input)

        # For computing a sub-network for pick
        prepick_robot_pose = self.get_prepick_robot_pose()

        pick_action = Lambda(slice_pick_pose_from_action)(self.action_input)
        tiled_pick_action = RepeatVector(self.n_key_confs)(pick_action)
        tiled_pick_action = Reshape((self.n_key_confs, 4, 1))(tiled_pick_action)
        H_col_robot_pose_pick = Concatenate(axis=2)([tiled_pick_action, prepick_robot_pose, C_H])
        H_pick = self.create_conv_layers(H_col_robot_pose_pick, 10)
        H_pick = Dense(dense_num, activation='relu')(H_pick)
        H_pick = Dense(dense_num, activation='relu')(H_pick)

        # For computing a sub-network for place
        place_action = Lambda(slice_place_pose_from_action)(self.action_input)
        place_action = RepeatVector(self.n_key_confs)(place_action)
        place_action = Reshape((self.n_key_confs, 4, 1))(place_action)

        tiled_abs_pose_and_pick_output = self.make_tiled_abs_obj_pose_and_pick_output(pick_action)
        H_col_abs_obj_pose_place = Concatenate(axis=2)([tiled_abs_pose_and_pick_output, place_action, C_H])
        H_place = self.create_conv_layers(H_col_abs_obj_pose_place, 64 + 2 + 4)
        H_place = Dense(dense_num, activation='relu')(H_place)
        H_place = Dense(dense_num, activation='relu')(H_place)
        self.discriminator_feature_matching_layer = H_place  # Concatenate(axis=-1)([H_place])

        place_value = Dense(1, activation='linear',
                            kernel_initializer=self.initializer,
                            bias_initializer=self.initializer)(H_place)
        pick_value = Dense(1, activation='linear',
                           kernel_initializer=self.initializer,
                           bias_initializer=self.initializer)(H_pick)
        disc_output = place_value  # Add()([place_value])

        return disc_output

    def create_discriminator(self):
        disc_output = self.get_disc_output_with_preprocessing_layers()
        self.disc_output = disc_output
        disc = Model(inputs=[self.action_input, self.collision_input, self.pose_input, self.tau_input],
                     outputs=disc_output,
                     name='disc_output')
        disc.compile(loss=tau_loss(self.tau_input), optimizer=self.opt_D)
        return disc

    def generate(self, state, poses, n_samples=1):
        if state.shape[0] == 1 and n_samples > 1:
            a_z = noise(n_samples, self.dim_action)
            state = np.tile(state, (n_samples, 1))
            g = self.a_gen.predict([a_z, state])
        elif state.shape[0] == 1 and n_samples == 1:
            a_z = noise(state.shape[0], self.dim_action)
            g = self.a_gen.predict([a_z, state, poses])
        else:
            raise NotImplementedError
        return g

    def compare_to_data(self, states, poses, actions):
        n_data = len(states)
        a_z = noise(n_data, self.dim_noise)
        pred = self.a_gen.predict([a_z, states, poses])
        gen_ir_params = pred[:, 0:4]
        data_ir_params = actions[:, 0:4]
        gen_place_base = pred[:, 4:]
        data_place_base = actions[:, 0:4]
        print "IR params", np.mean(np.linalg.norm(gen_ir_params - data_ir_params, axis=-1))
        print "Place params", np.mean(np.linalg.norm(gen_place_base - data_place_base, axis=-1))

    def get_batch(self, states, poses, actions, sum_rewards, batch_size):
        indices = np.random.randint(0, actions.shape[0], size=batch_size)
        s_batch = np.array(states[indices, :])  # collision vector
        a_batch = np.array(actions[indices, :])
        pose_batch = np.array(poses[indices, :])
        sum_reward_batch = np.array(sum_rewards[indices, :])
        return s_batch, pose_batch, a_batch, sum_reward_batch

    def train(self, states, poses, actions, sum_rewards, epochs=500, d_lr=1e-3, g_lr=1e-4):
        idxs = pickle.load(open('data_idxs_seed_%s' % self.seed, 'r'))
        train_idxs, test_idxs = idxs['train'], idxs['test']
        train_data, test_data = self.get_train_and_test_data(states, poses, actions, sum_rewards,
                                                             train_idxs, test_idxs)

        n_data = len(train_data['actions'])
        batch_size = self.get_batch_size(n_data)

        states = train_data['states']
        poses = train_data['poses']
        actions = train_data['actions']
        sum_rewards = train_data['sum_rewards']

        self.set_learning_rates(d_lr, g_lr)
        curr_tau = 1  # self.tau
        self.disc_mse_model.load_weights(self.save_folder + self.pretraining_file_name)
        pretrain_mse = self.compute_pure_mse(test_data)
        print "Pretrain mse", pretrain_mse

        mse_patience = 10
        post_train_mses = [0] * mse_patience
        mse_idx = 0
        for i in range(1, epochs):
            stime = time.time()
            tau_values = np.tile(curr_tau, (batch_size * 2, 1))
            print "Current tau value", curr_tau
            gen_before = self.a_gen.get_weights()
            disc_before = self.disc.get_weights()
            batch_idxs = range(0, actions.shape[0], batch_size)
            for j in batch_idxs:
                s_batch, pose_batch, a_batch, sum_rewards_batch = self.get_batch(states, poses, actions,
                                                                                 sum_rewards,
                                                                                 batch_size)

                # train \hat{S}
                # make fake and reals
                a_z = noise(batch_size, self.dim_noise)
                fake = self.a_gen.predict([a_z, s_batch, pose_batch])
                real = a_batch

                # make their scores
                fake_action_q = np.ones((batch_size, 1)) * INFEASIBLE_SCORE  # marks fake data
                real_action_q = sum_rewards_batch.reshape((batch_size, 1))
                batch_a = np.vstack([fake, real])
                batch_s = np.vstack([s_batch, s_batch])
                batch_rp = np.vstack([pose_batch, pose_batch])
                batch_scores = np.vstack([fake_action_q, real_action_q])
                self.disc.fit({'a': batch_a, 's': batch_s, 'pose': batch_rp, 'tau': tau_values},
                              batch_scores,
                              epochs=1,
                              verbose=False)
                posttrain_mse = self.compute_pure_mse(test_data)
                # print 'mse diff', pretrain_mse - posttrain_mse
                post_train_mses[mse_idx] = pretrain_mse - posttrain_mse
                mse_idx = (mse_idx + 1) % mse_patience
                # print mse_idx, post_train_mses
                # if np.any(post_train_mses < -100):
                drop_in_mse = pretrain_mse - posttrain_mse
                if pretrain_mse - posttrain_mse < -10:
                    self.save_weights(additional_name='_epoch_%d_batch_idx_%d_drop_in_mse_%.5f' % (i, j, drop_in_mse))

                # train G
                a_z = noise(batch_size, self.dim_noise)
                y_labels = np.ones((batch_size,))  # dummy variable
                self.DG.fit({'z': a_z, 's': s_batch, 'pose': pose_batch},
                            {'disc_output': y_labels, 'a_gen_output': y_labels},
                            epochs=1,
                            verbose=0)

                tttau_values = np.tile(curr_tau, (batch_size, 1))
                a_z = noise(batch_size, self.dim_noise)
                s_batch, pose_batch, a_batch, sum_rewards_batch = self.get_batch(states, poses, actions, sum_rewards,
                                                                                 batch_size)
                real_score_values = np.mean((self.disc.predict([a_batch, s_batch, pose_batch, tttau_values]).squeeze()))
                fake_score_values = np.mean((self.DG.predict([a_z, s_batch, pose_batch]).squeeze()))

                if real_score_values <= fake_score_values:
                    g_lr = 1e-4  # / (1 + 1e-1 * i)
                    d_lr = 1e-3  # / (1 + 1e-1 * i)
                else:
                    g_lr = 1e-3  # / (1 + 1e-1 * i)
                    d_lr = 1e-4  # / (1 + 1e-1 * i)
                self.set_learning_rates(d_lr, g_lr)

            gen_after = self.a_gen.get_weights()
            disc_after = self.disc.get_weights()
            gen_w_norm = np.linalg.norm(np.hstack([(a - b).flatten() for a, b in zip(gen_before, gen_after)]))
            disc_w_norm = np.linalg.norm(np.hstack([(a - b).flatten() for a, b in zip(disc_before, disc_after)]))

            print 'Completed: %d / %d' % (i, float(epochs))
            print "g_lr %.5f d_lr %.5f" % (g_lr, d_lr)
            # curr_tau = self.tau / (1.0 + 1e-1 * i)
            # if i > 20:
            #    self.save_weights(additional_name='_epoch_' + str(i))
            self.compare_to_data(states, poses, actions)

            tttau_values = np.tile(curr_tau, (len(states), 1))
            print "MSE diff", post_train_mses
            print "Real %.4f Gen %.4f" % (real_score_values, fake_score_values)
            print "Discriminiator MSE error", np.mean(np.linalg.norm(
                np.array(sum_rewards).squeeze() - self.disc.predict([actions, states, poses, tttau_values]).squeeze()))
            print "Epoch took: %.2fs" % (time.time() - stime)
            print "Generator weight norm diff", gen_w_norm
            print "Disc weight norm diff", disc_w_norm
            print "================================"
