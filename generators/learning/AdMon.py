from keras.layers import *
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.optimizers import *
from keras import backend as K
from keras import initializers

import time
import tensorflow as tf
import sys
import numpy as np
import os

from gtamp_utils import utils

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
    return np.random.normal(size=(n, z_size)).astype('float32')


def tile(x):
    reps = [1, 1, 32]
    return K.tile(x, reps)


class AdversarialMonteCarlo:
    def __init__(self, dim_action, dim_state, save_folder, tau, explr_const, key_configs=None,
                 action_scaler=None):

        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)

        self.opt_G = Adam(lr=1e-4, beta_1=0.5)
        self.opt_D = Adam(lr=1e-3, beta_1=0.5)

        # initialize
        self.initializer = initializers.glorot_normal()

        # get setup dimensions for inputs
        self.dim_action = dim_action
        self.dim_state = dim_state
        self.n_key_confs = dim_state[0]
        self.key_configs = key_configs

        self.action_scaler = action_scaler

        # define inputs
        self.action_input = Input(shape=(dim_action,), name='a', dtype='float32')  # action
        self.state_input = Input(shape=dim_state, name='s', dtype='float32')  # collision vector
        self.tau_input = Input(shape=(1,), name='tau', dtype='float32')  # collision vector

        self.explr_const = explr_const

        if dim_action < 10:
            dim_z = dim_action
        else:
            dim_z = int(dim_action / 2)

        self.dim_noise = dim_z
        self.noise_input = Input(shape=(self.dim_noise,), name='z', dtype='float32')

        self.a_gen, self.disc, self.DG, = self.createGAN()
        self.save_folder = save_folder
        self.tau = tau

    def createGAN(self):
        disc = self.create_discriminator()
        a_gen, a_gen_output = self.create_generator()
        for l in disc.layers:
            l.trainable = False
        DG_output = disc([a_gen_output, self.state_input, self.state_input])
        DG = Model(inputs=[self.noise_input, self.state_input], outputs=[DG_output])
        DG.compile(loss={'disc_output': G_loss, },
                   optimizer=self.opt_G,
                   metrics=[])
        return a_gen, disc, DG

    def save_weights(self, additional_name=''):
        self.a_gen.save_weights(self.save_folder + '/a_gen' + additional_name + '.h5')
        self.disc.save_weights(self.save_folder + '/disc' + additional_name + '.h5')

    def load_weights(self, agen_file, disc_file):
        self.a_gen.load_weights(self.save_folder + agen_file)
        self.disc.load_weights(self.save_folder + disc_file)

    def reset_weights(self, init=True):
        if init:
            self.a_gen.load_weights('a_gen_init.h5')
            self.disc.load_weights('disc_init.h5')
        else:
            self.a_gen.load_weights(self.save_folder + '/a_gen.h5')
            self.disc.load_weights(self.save_folder + '/disc.h5')

    def create_generator(self):
        dense_num = 64
        n_filters = 64

        # K_H = self.k_input
        W_H = Reshape((self.n_key_confs, self.dim_state[1], 1))(self.state_input)
        H = Conv2D(filters=n_filters,
                   kernel_size=(1, self.dim_state[1]),
                   strides=(1, 1),
                   activation='relu')(W_H)
        for _ in range(4):
            H = Conv2D(filters=n_filters,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='relu')(H)
        H = MaxPooling2D(pool_size=(2, 1))(H)
        H = Flatten()(H)
        H = Dense(dense_num, activation='relu')(H)
        H = Dense(dense_num, activation='relu')(H)
        Z_H = Dense(dense_num, activation='relu')(self.noise_input)
        H = Concatenate()([H, Z_H])
        a_gen_output = Dense(self.dim_action,
                             activation='linear',
                             kernel_initializer=self.initializer,
                             bias_initializer=self.initializer,
                             name='a_gen_output')(H)
        a_gen = Model(inputs=[self.noise_input, self.state_input], outputs=a_gen_output)
        return a_gen, a_gen_output

    def create_discriminator(self):
        init_ = self.initializer
        dense_num = 64
        n_filters = 64

        # K_H = self.k_input
        X_H = RepeatVector(self.n_key_confs)(self.action_input)
        X_H = Reshape((self.n_key_confs, self.dim_action, 1))(X_H)
        W_H = Reshape((self.n_key_confs, self.dim_state[1], 1))(self.state_input)
        XK_H = Concatenate(axis=2)([X_H, W_H])

        H = Conv2D(filters=n_filters,
                   kernel_size=(1, self.dim_action + self.dim_state[1]),
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
        disc = Model(inputs=[self.action_input, self.state_input, self.tau_input],
                     outputs=disc_output,
                     name='disc_output')
        disc.compile(loss=tau_loss(self.tau_input), optimizer=self.opt_D)
        return disc

    def generate(self, state, n_samples=1):
        if state.shape[0] == 1 and n_samples > 1:
            a_z = noise(n_samples, self.dim_action)
            state = np.tile(state, (n_samples, 1))
            #state = state.reshape((n_samples, self.n_key_confs, self.dim_state[1]))
            #g = self.action_scaler.inverse_transform(self.a_gen.predict([a_z, state]))
            g = self.a_gen.predict([a_z, state])
        elif state.shape[0] == 1 and n_samples == 1:
            a_z = noise(state.shape[0], self.dim_action)
            #state = state.reshape((1, self.n_key_confs, self.dim_state[1]))
            #g = self.action_scaler.inverse_transform(self.a_gen.predict([a_z, state]))
            g = self.a_gen.predict([a_z, state])
        else:
            raise NotImplementedError
        return g

    def predict_Q(self, w):
        a_z = noise(w.shape[0], self.dim_action)
        w = w.reshape((w.shape[0], self.n_key_confs, self.dim_state[1]))
        taus = np.tile(self.tau, (w.shape[0], 1))  # dummy variable when trying to predict
        g = self.a_gen.predict([a_z, w])
        return self.disc.predict([g, w, taus])

    def predict_V(self, w):
        n_samples = 100
        w = np.tile(w, (n_samples, 1, 1, 1))
        w = w.reshape((w.shape[0], self.n_key_confs, self.dim_state[1]))
        qvals = self.predict_Q(w)
        return qvals.mean()

    def record_evaluation(self, states, actions):
        n_data = len(states)
        a_z = noise(n_data, self.dim_noise)
        pred = self.a_gen.predict([a_z, states])
        gen_ir_params = pred[:, 0:3]
        utils.get_absolute_pick_base_pose_from_ir_parameters(gen_ir_params, obj) # which obj...?
        gen_place_base = pred[:, 3:]
        data_ir_params = actions[:, 0:3]
        data_place_base = actions[:, 0:3]
        utils.get_absolute_pick_base_pose_from_ir_parameters(data_ir_params, obj)
        # todo
        #   the trouble here is that the pick parameter consists of:
        #       - how close you are to the object expressed in terms of the proportion of the radius [0.4, 0.9]
        #       - base angle wrt the object [0,2pi)
        #       - angle offset from where it should look [-30,30]
        #   and so it is not really a configuration.
        #   How should we compare the distance in this domain?
        #   Also, how do I make sure it is in the right unit with the place base config?
        #   I guess one natural thing to do is to look convert it to the absolute pick base pose.
        import pdb;pdb.set_trace()

    def train(self, states, actions, sum_rewards, epochs=500, d_lr=1e-3, g_lr=1e-4):

        BATCH_SIZE = np.min([32, int(len(actions) * 0.1)])
        if BATCH_SIZE == 0:
            BATCH_SIZE = 1
        print BATCH_SIZE

        curr_tau = self.tau
        K.set_value(self.opt_G.lr, g_lr)
        K.set_value(self.opt_D.lr, d_lr)

        print self.opt_G.get_config()

        n_score_train = 1
        for i in range(1, epochs):
            stime = time.time()
            tau_values = np.tile(curr_tau, (BATCH_SIZE * 2, 1))
            print "Current tau value", curr_tau
            for idx in range(0, actions.shape[0], BATCH_SIZE):
                for score_train_idx in range(n_score_train):
                    # choose a batch of data
                    indices = np.random.randint(0, actions.shape[0], size=BATCH_SIZE)
                    s_batch = np.array(states[indices, :])  # collision vector
                    a_batch = np.array(actions[indices, :])
                    sum_reward_batch = np.array(sum_rewards[indices, :])

                    # train \hat{S}
                    # make fake and reals
                    a_z = noise(BATCH_SIZE, self.dim_noise)
                    fake = self.a_gen.predict([a_z, s_batch])
                    real = a_batch

                    # make their scores
                    fake_action_q = np.ones((BATCH_SIZE, 1)) * INFEASIBLE_SCORE  # marks fake data
                    real_action_q = sum_reward_batch
                    batch_a = np.vstack([fake, real])
                    batch_s = np.vstack([s_batch, s_batch])

                    batch_scores = np.vstack([fake_action_q, real_action_q])
                    self.disc.fit({'a': batch_a, 's': batch_s, 'tau': tau_values},
                                  batch_scores,
                                  epochs=1,
                                  verbose=False)

                # train G
                # why do i have labels for agen_output?
                a_z = noise(BATCH_SIZE, self.dim_noise)
                y_labels = np.ones((BATCH_SIZE,))  # dummy variable

                before = self.a_gen.get_weights()
                self.DG.fit({'z': a_z, 's': s_batch},
                            {'disc_output': y_labels, 'a_gen_output': y_labels},
                            epochs=1,
                            verbose=0)
                after = self.a_gen.get_weights()
                w_norm = np.linalg.norm(np.hstack([(a-b).flatten() for a, b in zip(before, after)]))
                print "Generator weight norm", w_norm

            print 'Completed: %.2f%%' % (i / float(epochs) * 100)
            curr_tau = np.power(curr_tau, i)
            self.save_weights(additional_name='_epoch_' + str(i))
            print "Epoch took: %.2fs" % (time.time() - stime)

