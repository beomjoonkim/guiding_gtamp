from generators.learning.AdMon import AdversarialMonteCarlo
from generators.learning.AdMonWithPose import AdversarialMonteCarloWithPose
from keras.layers import *
from keras.models import Model

import numpy as np
import tensorflow as tf


class MSETrainer:
    def __init__(self, save_folder, tau, disc, opt_D, config):
        self.save_folder = save_folder
        self.tau = tau
        self.disc = disc
        for l in self.disc.layers:
            l.trainable = True
        self.disc.compile(loss='mse', optimizer=opt_D)
        self.weight_file_name = 'mse_seed_%d.h5' % config.seed

    def load_weights(self, weight_file):
        self.disc.load_weights(self.save_folder + weight_file)

    def create_callbacks(self):
        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=100, ),
            tf.keras.callbacks.ModelCheckpoint(filepath=self.save_folder + self.weight_file_name,
                                               verbose=False,
                                               save_best_only=True,
                                               save_weights_only=True),
            tf.keras.callbacks.TensorBoard()
        ]
        return callbacks

    def train(self, states, actions, sum_rewards):
        curr_tau = self.tau
        callbacks = self.create_callbacks()
        print "Mean target value", np.mean(np.abs(sum_rewards))
        self.disc.fit(
            [actions, states], sum_rewards, batch_size=32, epochs=500, verbose=2,
            callbacks=callbacks,
            validation_split=0.1)


def slice_pick_pose_from_action(x):
    return x[:, 0:4]


def slice_place_pose_from_action(x):
    return x[:, 4:]


def slice_relative_robot_pose_from_pose(x):
    return x[:, 2:]


def slice_object_pose_from_pose(x):
    return x[:, :2]


class QmseWithPose(AdversarialMonteCarloWithPose, MSETrainer):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        AdversarialMonteCarloWithPose.__init__(self, dim_action, dim_collision, save_folder, tau, None, None)

        #self.disc_output = self.get_disc_output()
        self.disc_output = self.get_disc_output_with_preprocessing_layers()
        self.disc = Model(inputs=[self.action_input, self.collision_input, self.pose_input],
                          outputs=self.disc_output,
                          name='disc_output')
        MSETrainer.__init__(self, save_folder, tau, self.disc, self.opt_D, config)

    def train(self, states, poses, actions, sum_rewards, epochs=100, d_lr=1e-3, g_lr=1e-4):
        callbacks = self.create_callbacks()
        print "Mean target value", np.mean(np.abs(sum_rewards))
        self.disc.fit(
            [actions, states, poses], sum_rewards, batch_size=32, epochs=epochs, verbose=2,
            callbacks=callbacks,
            validation_split=0.1)

    def get_disc_output(self):
        dense_num = 64
        n_filters = 64

        # Tile actions and poses
        P_H = RepeatVector(self.n_key_confs)(self.pose_input)
        P_H = Reshape((self.n_key_confs, self.dim_poses, 1))(P_H)
        A_H = RepeatVector(self.n_key_confs)(self.action_input)
        A_H = Reshape((self.n_key_confs, self.dim_action, 1))(A_H)

        C_H = Reshape((self.n_key_confs, self.dim_collision[1], 1))(self.collision_input)
        XK_H = Concatenate(axis=2)([A_H, P_H, C_H])

        H = Conv2D(filters=n_filters,
                   kernel_size=(1, self.dim_action + self.dim_collision[1] + self.dim_poses),
                   strides=(1, 1),
                   activation='relu')(XK_H)
        for _ in range(4):
            H = Conv2D(filters=n_filters,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='relu')(H)
        H = MaxPooling2D(pool_size=(2, 1))(H)
        H = Flatten()(H)
        H = Dense(dense_num, activation='relu')(H)
        H = Dense(dense_num, activation='relu')(H)

        disc_output = Dense(1, activation='linear', kernel_initializer=self.initializer,
                            bias_initializer=self.initializer)(H)
        return disc_output

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

        C_H = Reshape((self.n_key_confs, self.dim_collision[1], 1))(self.collision_input)

        # get relative robot pose to obj
        robot_pose_rel_to_obj = Lambda(slice_relative_robot_pose_from_pose)(self.pose_input)
        robot_pose_rel_to_obj = RepeatVector(self.n_key_confs)(robot_pose_rel_to_obj)
        robot_pose_rel_to_obj = Reshape((self.n_key_confs, 2, 1))(robot_pose_rel_to_obj)

        pick_action = Lambda(slice_pick_pose_from_action)(self.action_input)
        pick_action = RepeatVector(self.n_key_confs)(pick_action)
        pick_action = Reshape((self.n_key_confs, 4, 1))(pick_action)

        # input for processing pick
        H_col_rel_robot_pose_pick = Concatenate(axis=2)([pick_action, robot_pose_rel_to_obj, C_H])

        # get object pose
        abs_obj_pose = Lambda(slice_object_pose_from_pose)(self.pose_input)
        abs_obj_pose = RepeatVector(self.n_key_confs)(abs_obj_pose)
        abs_obj_pose = Reshape((self.n_key_confs, 2, 1))(abs_obj_pose)

        place_action = Lambda(slice_place_pose_from_action)(self.action_input)
        place_action = RepeatVector(self.n_key_confs)(place_action)
        place_action = Reshape((self.n_key_confs, 4, 1))(place_action)

        # input for place
        H_col_abs_obj_pose_place = Concatenate(axis=2)([pick_action, place_action, abs_obj_pose, C_H])
        H_place = self.create_conv_layers(H_col_abs_obj_pose_place, 12)
        H_place = Dense(dense_num, activation='relu')(H_place)
        H_place = Dense(dense_num, activation='relu')(H_place)

        H_pick = self.create_conv_layers(H_col_rel_robot_pose_pick, 8)
        H_pick = Dense(dense_num, activation='relu')(H_pick)
        H_pick = Dense(dense_num, activation='relu')(H_pick)

        H = Concatenate(axis=-1)([H_pick, H_place])
        H = Dense(dense_num, activation='relu')(H)
        H = Dense(dense_num, activation='relu')(H)

        disc_output = Dense(1, activation='linear', kernel_initializer=self.initializer,
                            bias_initializer=self.initializer)(H)
        return disc_output
