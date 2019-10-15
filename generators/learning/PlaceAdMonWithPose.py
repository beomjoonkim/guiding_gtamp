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

from AdversarialPolicy import tau_loss, G_loss, INFEASIBLE_SCORE
from FeatureMatchinAdMonWithPose import FeatureMatchingAdMonWithPose
from AdversarialPolicy import AdversarialPolicy
from AdMonWithPose import AdversarialMonteCarloWithPose

if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'


def noise(n, z_size):
    # todo use the uniform over the entire action space here
    # return np.random.normal(size=(n, z_size)).astype('float32')
    domain = np.array([[0, -20, -1, -1], [10, 0, 1, 1]])
    return np.random.uniform(low=domain[0], high=domain[1], size=(n, 4))


def slice_pick_pose_from_action(x):
    return x[:, :4]


def slice_place_pose_from_action(x):
    return x[:, 4:]


def slice_prepick_robot_pose_from_pose(x):
    return x[:, 4:]


def slice_object_pose_from_pose(x):
    return x[:, :4]


class PlaceAdmonWithPose(AdversarialMonteCarloWithPose):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        AdversarialMonteCarloWithPose.__init__(self, dim_action, dim_collision, save_folder, tau, config)

    def create_a_gen_output(self):
        dense_num = 64

        # Collision vector
        C_H = Reshape((self.n_key_confs, self.dim_collision[1], 1))(self.collision_input)

        obj_pose = self.get_abs_obj_pose()

        H_col_abs_obj_pose_place = Concatenate(axis=2)([obj_pose, C_H])
        H_place = self.create_conv_layers(H_col_abs_obj_pose_place, 4 + 6)
        H_place = Dense(dense_num, activation='relu')(H_place)
        H_place = Dense(dense_num, activation='relu')(H_place)
        H_place = Concatenate(axis=-1)([H_place, self.noise_input])
        place_output = Dense(4, activation='linear')(H_place)
        a_gen_output = place_output
        return a_gen_output

    def create_discriminator(self):
        disc_output = self.get_disc_output_with_preprocessing_layers()
        self.disc_output = disc_output
        disc = Model(inputs=[self.action_input, self.collision_input, self.pose_input, self.tau_input],
                     outputs=disc_output,
                     name='disc_output')
        disc.compile(loss=tau_loss(self.tau_input), optimizer=self.opt_D)
        return disc

    def create_disc_output_with_relevance_network(self):
        relevance_network = self.construct_relevance_network()

    def get_disc_output_with_preprocessing_layers(self):
        dense_num = 64

        # Collision vector
        C_H = Reshape((self.n_key_confs, self.dim_collision[1], 1))(self.collision_input)

        # For computing a sub-network for pick

        abs_obj_pose = Lambda(slice_object_pose_from_pose)(self.pose_input)

        # For computing a sub-network for place
        H_preprocess = Concatenate(axis=-1)([abs_obj_pose, self.action_input])
        H_preprocess = Dense(dense_num, activation='relu')(H_preprocess)
        H_preprocess = Dense(dense_num, activation='relu')(H_preprocess)
        H_preprocess = RepeatVector(self.n_key_confs)(H_preprocess)
        H_preprocess = Reshape((self.n_key_confs, dense_num, 1))(H_preprocess)

        H_col_abs_obj_pose_place = Concatenate(axis=2)([H_preprocess, C_H])
        H_place = self.create_conv_layers(H_col_abs_obj_pose_place, 64 + 2)
        H_place = Dense(dense_num, activation='relu')(H_place)
        H_place = Dense(dense_num, activation='relu')(H_place)
        self.discriminator_feature_matching_layer = H_place  # Concatenate(axis=-1)([H_place])

        place_value = Dense(1, activation='linear',
                            kernel_initializer=self.initializer,
                            bias_initializer=self.initializer)(H_place)

        disc_output = place_value  # Add()([place_value])
        return disc_output

    def compare_to_data(self, states, poses, actions):
        n_data = len(states)
        a_z = noise(n_data, self.dim_noise)
        pred = self.a_gen.predict([a_z, states, poses])
        gen_place_base = pred[:, :4]
        data_place_base = actions[:, :4]
        print "Place params", np.mean(np.linalg.norm(gen_place_base - data_place_base, axis=-1))
