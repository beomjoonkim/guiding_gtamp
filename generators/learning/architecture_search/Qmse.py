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


def slice_pick_pose_from_action(x):
    return x[:, :4]


def slice_place_pose_from_action(x):
    return x[:, 4:]


def slice_prepick_robot_pose_from_pose(x):
    return x[:, 4:]


def slice_object_pose_from_pose(x):
    return x[:, :4]


class QmseWithPose(AdversarialMonteCarloWithPose, MSETrainer):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        AdversarialMonteCarloWithPose.__init__(self, dim_action, dim_collision, save_folder, tau, config)

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



