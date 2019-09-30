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

from AdversarialPolicy import tau_loss, G_loss, noise, INFEASIBLE_SCORE
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


class AdversarialMonteCarloWithPose(AdversarialPolicy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        AdversarialPolicy.__init__(self, dim_action, dim_collision, save_folder, tau, None, None)
        self.dim_poses = 8
        self.dim_collision = dim_collision
        self.action_input = Input(shape=(dim_action,), name='a', dtype='float32')  # action
        self.collision_input = Input(shape=dim_collision, name='s', dtype='float32')  # collision vector
        self.pose_input = Input(shape=(self.dim_poses,), name='pose', dtype='float32')  # collision vector
        self.a_gen, self.disc, self.DG, = self.createGAN()
        self.weight_file_name = 'admonpose_seed_%d' % config.seed

    def createGAN(self):
        disc = self.create_discriminator()
        a_gen, a_gen_output = self.create_generator()
        for l in disc.layers:
            l.trainable = False
        DG_output = disc([a_gen_output, self.collision_input, self.pose_input, self.collision_input])
        DG = Model(inputs=[self.noise_input, self.collision_input, self.pose_input], outputs=[DG_output])
        DG.compile(loss={'disc_output': G_loss},
                   optimizer=self.opt_G,
                   metrics=[])
        return a_gen, disc, DG

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

        # what if I used H_pick as an input to the network?
        # I have no way of measuring the impact of doing so.
        # I really have to get the evaluator going.
        #   Naive: the number of feasible parameters that you get within the 200 trials
        """
        pick_output = Dense(4, activation='linear',
                            kernel_initializer=self.initializer,
                            bias_initializer=self.initializer)(H_pick)
        """

        # process state var relevant to predicting place
        abs_obj_pose = self.get_abs_obj_pose()
        H_col_abs_obj_pose_place = Concatenate(axis=2)([abs_obj_pose, C_H])
        H_place = self.create_conv_layers(H_col_abs_obj_pose_place, 6)
        H_place = Dense(dense_num, activation='relu')(H_place)
        H_place = Dense(dense_num, activation='relu')(H_place)

        # Get the output from both processed pick and place
        H = Concatenate(axis=-1)([H_pick, H_place, self.noise_input])
        H = Dense(dense_num, activation='relu')(H)
        a_gen_output = Dense(self.dim_action,
                             activation='linear',
                             kernel_initializer=self.initializer,
                             bias_initializer=self.initializer,
                             name='a_gen_output')(H)
        return a_gen_output

    def create_generator(self):
        a_gen_output = self.create_a_gen_output()
        a_gen = Model(inputs=[self.noise_input, self.collision_input, self.pose_input], outputs=a_gen_output)
        return a_gen, a_gen_output

    def create_conv_layers(self, input, n_dim):
        n_filters = 64

        H = Conv2D(filters=n_filters,
                   kernel_size=(1, n_dim),
                   strides=(1, 1),
                   activation='relu')(input)
        for _ in range(4):
            H = Conv2D(filters=n_filters,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='relu')(H)
        H = MaxPooling2D(pool_size=(2, 1))(H)
        H = Flatten()(H)

        return H

    def get_disc_output_with_preprocessing_layers(self):
        dense_num = 64

        # Collision vector
        C_H = Reshape((self.n_key_confs, self.dim_collision[1], 1))(self.collision_input)

        # For computing a sub-network for pick
        prepick_robot_pose = self.get_prepick_robot_pose()

        pick_action = Lambda(slice_pick_pose_from_action)(self.action_input)
        pick_action = RepeatVector(self.n_key_confs)(pick_action)
        pick_action = Reshape((self.n_key_confs, 4, 1))(pick_action)
        H_col_robot_pose_pick = Concatenate(axis=2)([pick_action, prepick_robot_pose, C_H])
        H_pick = self.create_conv_layers(H_col_robot_pose_pick, 10)
        H_pick = Dense(dense_num, activation='relu')(H_pick)
        H_pick = Dense(dense_num, activation='relu')(H_pick)

        # For computing a sub-network for place
        abs_obj_pose = self.get_abs_obj_pose()
        place_action = Lambda(slice_place_pose_from_action)(self.action_input)
        place_action = RepeatVector(self.n_key_confs)(place_action)
        place_action = Reshape((self.n_key_confs, 4, 1))(place_action)

        H_col_abs_obj_pose_place = Concatenate(axis=2)([pick_action, place_action, abs_obj_pose, C_H])
        H_place = self.create_conv_layers(H_col_abs_obj_pose_place, 14)
        H_place = Dense(dense_num, activation='relu')(H_place)
        H_place = Dense(dense_num, activation='relu')(H_place)

        # Get the output from both processed pick and place
        H = Concatenate(axis=-1)([H_pick, H_place])
        H = Dense(dense_num, activation='relu')(H)
        self.discriminator_feature_matching_layer = H
        H = Dense(dense_num, activation='relu')(H)

        disc_output = Dense(1, activation='linear', kernel_initializer=self.initializer,
                            bias_initializer=self.initializer)(H)
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
            # state = state.reshape((n_samples, self.n_key_confs, self.dim_collision[1]))
            # g = self.action_scaler.inverse_transform(self.a_gen.predict([a_z, state]))
            g = self.a_gen.predict([a_z, state])
        elif state.shape[0] == 1 and n_samples == 1:
            a_z = noise(state.shape[0], self.dim_action)
            # state = state.reshape((1, self.n_key_confs, self.dim_collision[1]))
            # g = self.action_scaler.inverse_transform(self.a_gen.predict([a_z, state]))
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
        batch_size = np.min([32, int(len(actions) * 0.1)])
        if batch_size == 0:
            batch_size = 1
        print batch_size

        curr_tau = self.tau
        K.set_value(self.opt_G.lr, g_lr)
        K.set_value(self.opt_D.lr, d_lr)
        print self.opt_G.get_config()

        n_score_train = 1
        for i in range(1, epochs):
            self.compare_to_data(states, poses, actions)
            stime = time.time()
            tau_values = np.tile(curr_tau, (batch_size * 2, 1))
            print "Current tau value", curr_tau
            gen_before = self.a_gen.get_weights()
            disc_before = self.disc.get_weights()
            batch_idxs = range(0, actions.shape[0], batch_size)
            for k, idx in enumerate(batch_idxs):
                # print 'Epoch completion: %d / %d' % (k, len(batch_idxs))
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
                # print "Real %.4f Gen %.4f" % (real_score_values, fake_score_values)

                if real_score_values <= fake_score_values:
                    g_lr = 1e-4 / (1 + 1e-1 * i)
                    d_lr = 1e-3 / (1 + 1e-1 * i)
                    K.set_value(self.opt_G.lr, g_lr)
                    K.set_value(self.opt_D.lr, d_lr)
                else:
                    g_lr = 1e-3 / (1 + 1e-1 * i)
                    d_lr = 1e-4 / (1 + 1e-1 * i)
                    K.set_value(self.opt_G.lr, g_lr)
                    K.set_value(self.opt_D.lr, d_lr)

            gen_after = self.a_gen.get_weights()
            disc_after = self.disc.get_weights()
            gen_w_norm = np.linalg.norm(np.hstack([(a - b).flatten() for a, b in zip(gen_before, gen_after)]))
            disc_w_norm = np.linalg.norm(np.hstack([(a - b).flatten() for a, b in zip(disc_before, disc_after)]))

            print 'Completed: %d / %d' % (i, float(epochs))
            print "g_lr %.5f d_lr %.5f" % (g_lr, d_lr)
            # curr_tau = curr_tau * 1 /
            curr_tau = self.tau / (1.0 + 1e-1 * i)
            if i > 20:
                self.save_weights(additional_name='_epoch_' + str(i))
            self.compare_to_data(states, poses, actions)
            a_z = noise(len(states), self.dim_noise)

            tttau_values = np.tile(curr_tau, (len(states), 1))
            print "Real %.4f Gen %.4f" % (real_score_values, fake_score_values)
            print "Discriminiator MSE error", np.mean(np.linalg.norm(
                np.array(sum_rewards).squeeze() - self.disc.predict([actions, states, poses, tttau_values]).squeeze()))
            print "Epoch took: %.2fs" % (time.time() - stime)
            print "Generator weight norm diff", gen_w_norm
            print "Disc weight norm diff", disc_w_norm
            print "================================"


class FeatureMatchingAdMonWithPose(AdversarialMonteCarloWithPose):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        AdversarialMonteCarloWithPose.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.target_feature_match_input = Input(shape=(64,), name='feature', dtype='float32')  # action

    def create_discriminator(self):
        disc_output = self.get_disc_output_with_preprocessing_layers()
        self.disc_output = disc_output
        disc = Model(inputs=[self.action_input, self.collision_input, self.pose_input, self.tau_input],
                     outputs=disc_output,
                     name='disc_output')
        disc.compile(loss=tau_loss(self.tau_input), optimizer=self.opt_D)

        self.discriminator_feature_matching_model = Model(
            inputs=[self.action_input, self.collision_input, self.pose_input],
            outputs=self.discriminator_feature_matching_layer,
            name='feature_matching_model')

        self.discriminator_feature_matching_model.compile(loss='mse', optimizer=self.opt_D)
        return disc

    def createGAN(self):
        disc = self.create_discriminator()
        a_gen, a_gen_output = self.create_generator()
        for l in disc.layers:
            l.trainable = False
        DG_output = self.discriminator_feature_matching_model([a_gen_output, self.collision_input, self.pose_input])
        DG = Model(inputs=[self.noise_input, self.collision_input, self.pose_input], outputs=[DG_output])
        DG.compile(loss='mse', optimizer=self.opt_G)

        return a_gen, disc, DG

    def train(self, states, poses, actions, sum_rewards, epochs=500, d_lr=1e-3, g_lr=1e-4):
        batch_size = np.min([32, int(len(actions) * 0.1)])
        if batch_size == 0:
            batch_size = 1
        print batch_size

        curr_tau = self.tau
        K.set_value(self.opt_G.lr, g_lr)
        K.set_value(self.opt_D.lr, d_lr)
        print self.opt_G.get_config()

        for i in range(1, epochs):
            self.compare_to_data(states, poses, actions)
            stime = time.time()
            tau_values = np.tile(curr_tau, (batch_size * 2, 1))
            print "Current tau value", curr_tau
            gen_before = self.a_gen.get_weights()
            disc_before = self.disc.get_weights()
            batch_idxs = range(0, actions.shape[0], batch_size)
            for k, idx in enumerate(batch_idxs):
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
                batch_a_for_disc = np.vstack([fake, real])
                batch_s_for_disc = np.vstack([s_batch, s_batch])
                batch_rp_for_disc = np.vstack([pose_batch, pose_batch])
                batch_scores = np.vstack([fake_action_q, real_action_q])
                self.disc.fit({'a': batch_a_for_disc, 's': batch_s_for_disc, 'pose': batch_rp_for_disc, 'tau': tau_values},
                              batch_scores,
                              epochs=1,
                              verbose=False)

                # train G
                a_z = noise(batch_size, self.dim_noise)
                feature_to_match = self.discriminator_feature_matching_model.predict([a_batch, s_batch, pose_batch])
                self.DG.fit({'z': a_z, 's': s_batch, 'pose': pose_batch}, feature_to_match, epochs=1, verbose=0)

                tttau_values = np.tile(curr_tau, (batch_size, 1))
                a_z = noise(batch_size, self.dim_noise)
                s_batch, pose_batch, a_batch, sum_rewards_batch = self.get_batch(states, poses, actions, sum_rewards,
                                                                                 batch_size)
                real_score_values = np.mean((self.disc.predict([a_batch, s_batch, pose_batch, tttau_values]).squeeze()))
                fake_score_values = np.mean((self.DG.predict([a_z, s_batch, pose_batch]).squeeze()))
                # print "Real %.4f Gen %.4f" % (real_score_values, fake_score_values)

            gen_after = self.a_gen.get_weights()
            disc_after = self.disc.get_weights()
            gen_w_norm = np.linalg.norm(np.hstack([(a - b).flatten() for a, b in zip(gen_before, gen_after)]))
            disc_w_norm = np.linalg.norm(np.hstack([(a - b).flatten() for a, b in zip(disc_before, disc_after)]))

            print 'Completed: %d / %d' % (i, float(epochs))
            print "g_lr %.5f d_lr %.5f" % (g_lr, d_lr)
            # curr_tau = curr_tau * 1 /
            curr_tau = self.tau / (1.0 + 1e-1 * i)
            if i > 20:
                self.save_weights(additional_name='_epoch_' + str(i))
            self.compare_to_data(states, poses, actions)
            a_z = noise(len(states), self.dim_noise)

            tttau_values = np.tile(curr_tau, (len(states), 1))
            print "Real %.4f Gen %.4f" % (real_score_values, fake_score_values)
            print "Discriminiator MSE error", np.mean(np.linalg.norm(
                np.array(sum_rewards).squeeze() - self.disc.predict([actions, states, poses, tttau_values]).squeeze()))
            print "Epoch took: %.2fs" % (time.time() - stime)
            print "Generator weight norm diff", gen_w_norm
            print "Disc weight norm diff", disc_w_norm
            print "================================"
