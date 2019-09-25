from AdMon import AdversarialMonteCarlo
from keras.models import Sequential, Model

import numpy as np
import tensorflow as tf


class Qmse(AdversarialMonteCarlo):
    def __init__(self, dim_action, dim_state, save_folder, tau, explr_const, key_configs=None,
                     action_scaler=None):
        AdversarialMonteCarlo.__init__(self, dim_action, dim_state, save_folder, tau, explr_const,
                                       key_configs=None, action_scaler=None)

        self.disc = Model(inputs=[self.action_input, self.state_input],
                          outputs=self.disc_output,
                          name='disc_output')
        for l in self.disc.layers:
            l.trainable = True
        self.disc.compile(loss='mse', optimizer=self.opt_D)

    def create_callbacks(self):
        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=100, ),
            tf.keras.callbacks.ModelCheckpoint(filepath=self.save_folder+'mse.h5',
                                               verbose=False,
                                               save_best_only=True,
                                               save_weights_only=True),
            tf.keras.callbacks.TensorBoard()
        ]
        return callbacks

    def train(self, states, actions, sum_rewards, epochs=500, d_lr=1e-3, g_lr=1e-4):
        curr_tau = self.tau
        tau_values = np.tile(curr_tau, (len(states), 1))
        callbacks = self.create_callbacks()
        print "Mean target value", np.mean(np.abs(sum_rewards))
        self.disc.fit(
                [actions, states], sum_rewards, batch_size=32, epochs=500, verbose=2,
                callbacks=callbacks,
                validation_split=0.1)


