from keras.layers import *
from keras.layers.merge import Concatenate
from keras.models import Sequential, Model
from keras.optimizers import *
from keras import backend as K
from keras import initializers
from AdversarialPolicy import tau_loss, G_loss, noise, INFEASIBLE_SCORE
from AdversarialPolicy import AdversarialPolicy

import time
import numpy as np
import os


class AdversarialMonteCarlo(AdversarialPolicy):
    def __init__(self, dim_action, dim_state, save_folder, tau, key_configs=None,
                 action_scaler=None):
        AdversarialPolicy.__init__(self, dim_action, dim_state, save_folder, tau, key_configs, action_scaler)
        self.action_input = Input(shape=(dim_action,), name='a', dtype='float32')  # action
        self.state_input = Input(shape=dim_state, name='s', dtype='float32')  # collision vector
        self.a_gen, self.disc, self.DG, = self.createGAN()

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

    def load_weights(self, agen_file):
        self.a_gen.load_weights(self.save_folder + agen_file)

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
        self.disc_output = disc_output
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

    def compare_to_data(self, states, actions):
        n_data = len(states)
        a_z = noise(n_data, self.dim_noise)
        pred = self.a_gen.predict([a_z, states])
        gen_ir_params = pred[:, 0:4]
        data_ir_params = actions[:, 0:4]
        gen_place_base = pred[:, 4:]
        data_place_base = actions[:, 0:4]
        print "IR params", np.mean(np.linalg.norm(gen_ir_params - data_ir_params, axis=-1))
        print "Place params", np.mean(np.linalg.norm(gen_place_base - data_place_base, axis=-1))

        # todo
        #   the trouble here is that the pick parameter consists of:
        #       - how close you are to the object expressed in terms of the proportion of the radius [0.4, 0.9]
        #       - base angle wrt the object [0,2pi)
        #       - angle offset from where it should look [-30,30]
        #   and so it is not really a configuration.
        #   How should we compare the distance in this domain?
        #   Also, how do I make sure it is in the right unit with the place base config?
        #   I guess one natural thing to do is to look convert it to the absolute pick base pose.

    def get_batch(self, states, actions, sum_rewards, batch_size):
        indices = np.random.randint(0, actions.shape[0], size=batch_size)
        s_batch = np.array(states[indices, :])  # collision vector
        a_batch = np.array(actions[indices, :])
        sum_reward_batch = np.array(sum_rewards[indices, :])
        return s_batch, a_batch, sum_reward_batch

    def train(self, states, actions, sum_rewards, epochs=500, d_lr=1e-2, g_lr=1e-3):

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
            self.compare_to_data(states, actions)
            stime = time.time()
            tau_values = np.tile(curr_tau, (batch_size * 2, 1))
            print "Current tau value", curr_tau
            gen_before = self.a_gen.get_weights()
            disc_before = self.disc.get_weights()
            batch_idxs = range(0, actions.shape[0], batch_size)
            for k, idx in enumerate(batch_idxs):
                #print 'Epoch completion: %d / %d' % (k, len(batch_idxs))
                s_batch, a_batch, sum_rewards_batch = self.get_batch(states, actions, sum_rewards, batch_size)

                # train \hat{S}
                # make fake and reals
                a_z = noise(batch_size, self.dim_noise)
                fake = self.a_gen.predict([a_z, s_batch])
                real = a_batch

                # make their scores
                fake_action_q = np.ones((batch_size, 1)) * INFEASIBLE_SCORE  # marks fake data
                real_action_q = sum_rewards_batch.reshape((batch_size,1))
                batch_a = np.vstack([fake, real])
                batch_s = np.vstack([s_batch, s_batch])
                batch_scores = np.vstack([fake_action_q, real_action_q])
                self.disc.fit({'a': batch_a, 's': batch_s, 'tau': tau_values},
                              batch_scores,
                              epochs=1,
                              verbose=False)

                # train G
                a_z = noise(batch_size, self.dim_noise)
                y_labels = np.ones((batch_size,))  # dummy variable
                self.DG.fit({'z': a_z, 's': s_batch},
                            {'disc_output': y_labels, 'a_gen_output': y_labels},
                            epochs=1,
                            verbose=0)
                
                tttau_values = np.tile(curr_tau, (batch_size, 1))
                a_z = noise(batch_size, self.dim_noise)
                s_batch, a_batch, sum_rewards_batch = self.get_batch(states, actions, sum_rewards, batch_size)
                real_score_values = np.mean((self.disc.predict([a_batch, s_batch, tttau_values]).squeeze()))
                fake_score_values = np.mean((self.DG.predict([a_z, s_batch]).squeeze()))
                #print "Real %.4f Gen %.4f" % (real_score_values, fake_score_values)

                if real_score_values <= fake_score_values:
                    g_lr = 1e-4 / (1+1e-1*i)
                    d_lr = 1e-3 / (1+1e-1*i) 
                    K.set_value(self.opt_G.lr, g_lr)
                    K.set_value(self.opt_D.lr, d_lr)    
                else:
                    g_lr = 1e-3 / (1+1e-1*i)
                    d_lr = 1e-4 / (1+1e-1*i)
                    K.set_value(self.opt_G.lr, g_lr)
                    K.set_value(self.opt_D.lr, d_lr)    

            gen_after = self.a_gen.get_weights()
            disc_after = self.disc.get_weights()
            gen_w_norm = np.linalg.norm(np.hstack([(a-b).flatten() for a, b in zip(gen_before, gen_after)]))
            disc_w_norm = np.linalg.norm(np.hstack([(a-b).flatten() for a, b in zip(disc_before, disc_after)]))

            print 'Completed: %d / %d' % (i, float(epochs))
            print "g_lr %.5f d_lr %.5f" %(g_lr, d_lr)
            #curr_tau = curr_tau * 1 /
            curr_tau = self.tau / (1.0 + 1e-1 * i)
            self.save_weights(additional_name='_epoch_' + str(i))
            self.compare_to_data(states, actions)
            a_z = noise(len(states), self.dim_noise)

            tttau_values = np.tile(curr_tau, (len(states), 1))
            print "Real %.4f Gen %.4f" % (real_score_values, fake_score_values)
            print "Discriminiator MSE error", np.mean(np.linalg.norm(np.array(sum_rewards).squeeze() - self.disc.predict([actions, states, tttau_values]).squeeze()))
            print "Epoch took: %.2fs" % (time.time() - stime)
            print "Generator weight norm diff", gen_w_norm
            print "Disc weight norm diff", disc_w_norm
            print "================================"

