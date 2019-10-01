from Qmse import MSETrainer
from generators.learning.PlaceAdMonWithPose import PlaceAdmonWithPose
from keras.models import Model
from keras.layers import *

import numpy as np


class PolicyWithPose(PlaceAdmonWithPose, MSETrainer):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        PlaceAdmonWithPose.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        # self.disc_output = self.get_disc_output()
        self.a_gen_output = self.create_a_gen_output()
        self.a_gen = Model(inputs=[self.collision_input, self.pose_input],
                           outputs=self.a_gen_output,
                           name='agen_output')
        self.a_gen.compile(loss='mse', optimizer=self.opt_G)
        for l in self.a_gen.layers:
            l.trainable = True
        MSETrainer.__init__(self, save_folder, tau, self.disc, self.opt_D, config)

    def load_weights(self):
        print "Loading weights", self.save_folder + self.weight_file_name
        self.a_gen.load_weights(self.save_folder + self.weight_file_name)
        print "Weight loaded"

    def create_a_gen_output(self):
        dense_num = 64

        # Collision vector
        C_H = Reshape((self.n_key_confs, self.dim_collision[1], 1))(self.collision_input)

        obj_pose = self.get_abs_obj_pose()

        H_col_abs_obj_pose_place = Concatenate(axis=2)([obj_pose, C_H])
        H_place = self.create_conv_layers(H_col_abs_obj_pose_place, 4 + 6)
        H_place = Dense(dense_num, activation='relu')(H_place)
        H_place = Dense(dense_num, activation='relu')(H_place)
        place_output = Dense(4, activation='linear')(H_place)
        a_gen_output = place_output
        return a_gen_output

    def train(self, states, poses, actions, epochs=100, d_lr=1e-3, g_lr=1e-4):
        callbacks = self.create_callbacks()
        self.a_gen.fit(
            [states, poses], actions, batch_size=32, epochs=epochs, verbose=2,
            callbacks=callbacks,
            validation_split=0.1)


