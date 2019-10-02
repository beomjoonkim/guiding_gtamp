from keras.optimizers import *
from keras import initializers
from keras.layers import *
from keras.models import Model

import os
import sys
import numpy as np

INFEASIBLE_SCORE = -sys.float_info.max


def tau_loss(tau):
    def augmented_mse(score_data, D_pred):
        # Determine which of Dpred correspond to fake val
        neg_mask = tf.equal(score_data, INFEASIBLE_SCORE)
        y_neg = tf.boolean_mask(D_pred, neg_mask)

        # Determine which of Dpred correspond to true fcn val
        pos_mask = tf.not_equal(score_data, INFEASIBLE_SCORE)
        y_pos = tf.boolean_mask(D_pred, pos_mask)
        score_pos = tf.boolean_mask(score_data, pos_mask)

        # compute mse w.r.t true function values
        mse_on_true_data = K.mean((K.square(score_pos - y_pos)), axis=-1)
        return mse_on_true_data + tau[0] * K.mean(y_neg)  # try to minimize the value of y_neg

    return augmented_mse


def G_loss(dummy, pred):
    return -K.mean(pred, axis=-1)  # try to maximize the value of pred


def noise(n, z_size):
    # todo use the uniform over the entire action space here
    return np.random.normal(size=(n, z_size)).astype('float32')


def tile(x):
    reps = [1, 1, 32]
    return K.tile(x, reps)


class AdversarialPolicy:
    def __init__(self, dim_action, dim_state, save_folder, tau):
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        self.opt_G = Adam(lr=1e-4, beta_1=0.5)
        self.opt_D = Adam(lr=1e-3, beta_1=0.5)

        # initialize
        self.initializer = initializers.glorot_uniform()

        if dim_action < 10:
            dim_z = dim_action
        else:
            dim_z = int(dim_action / 2)

        self.dim_noise = dim_z

        # get setup dimensions for inputs
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.n_key_confs = dim_state[0]

        self.tau = tau

        self.noise_input = Input(shape=(self.dim_noise,), name='z', dtype='float32')
        self.tau_input = Input(shape=(1,), name='tau', dtype='float32')  # collision vector

        self.save_folder = save_folder

        self.test_data = None
        self.desired_test_err = None
        self.disc = None

    @staticmethod
    def get_batch_size(n_data):
        batch_size = np.min([32, n_data])
        if batch_size == 0:
            batch_size = 1
        return batch_size

    def set_learning_rates(self, d_lr, g_lr):
        K.set_value(self.opt_G.lr, g_lr)
        K.set_value(self.opt_D.lr, d_lr)

    def create_callbacks_for_pretraining(self):
        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=10, ),
            tf.keras.callbacks.ModelCheckpoint(filepath=self.save_folder + self.pretraining_file_name,
                                               verbose=False,
                                               save_best_only=True,
                                               save_weights_only=True),
            # tf.keras.callbacks.TensorBoard()
        ]
        return callbacks




