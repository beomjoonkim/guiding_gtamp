import matplotlib as mpl
# mpl.use('Agg') # do this before importing plt to run with no DISPLAY
import matplotlib.pyplot as plt

from keras.layers import *
from keras.models import Sequential, Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from PolicySearch import PolicySearch
import tensorflow as tf
import sys
import numpy as np

from data_load_utils import format_RL_data
import scipy.signal

INFEASIBLE_SCORE = -sys.float_info.max


def ppo_loss(sumA_weight, old_pi_a, tau):
    # This is actually PPO loss function
    def loss(actions, pi_pred):
        # log prob term is -K.sum(K.square(old_pi_a - actions),axis=-1,keepdims=True)
        p_old = K.exp(-K.sum(K.square(old_pi_a - actions), axis=-1, keepdims=True))
        p_new = K.exp(-K.sum(K.square(pi_pred - actions), axis=-1, keepdims=True))
        p_ratio = p_new / (p_old + 1e-5)

        L_cpi = tf.multiply(sumA_weight, p_ratio)
        clipped = tf.clip_by_value(p_ratio, 1 - tau[0, 0], 1 + tau[0, 0])
        L_clipped = tf.multiply(sumA_weight, clipped)
        L = tf.minimum(L_cpi, L_clipped)
        return -L

    return loss


class PPO(PolicySearch):
    def __init__(self, sess, dim_action, dim_state, save_folder, tau, explr_const):
        PolicySearch.__init__(self, sess, dim_action, dim_state, save_folder, tau, explr_const)

        # define inputs
        self.a_input = Input(shape=(dim_action,), name='x', dtype='float32')  # action
        self.s_input = Input(shape=(dim_state,), name='w', dtype='float32')  # collision vector
        self.tau_input = Input(shape=(1,), name='tau', dtype='float32')  # collision vector

        self.policy, self.qfcn, _ = self.create_qfcn_and_policy()

    def create_qfcn_and_policy(self):
        qfcn = self.create_qfcn()
        policy, policy_output = self.create_policy()
        return policy, qfcn, None

    def create_policy(self):
        init_ = self.initializer
        dropout_rate = 0.25
        dense_num = 64
        n_filters = 64

        H = Dense(dense_num, activation='relu')(self.s_input)
        H = Dense(dense_num, activation='relu')(H)
        policy_output = Dense(self.dim_action,
                             activation='linear',
                             init=init_,
                             name='policy_output')(H)

        # these two are used for training purposes
        sumAweight_input = Input(shape=(1,), name='sumA', dtype='float32')
        old_pi_a_input = Input(shape=(self.dim_action,), name='old_pi_a', dtype='float32')
        policy = Model(input=[self.s_input, sumAweight_input, old_pi_a_input, self.tau_input], output=[policy_output])
        policy.compile(loss=ppo_loss(sumA_weight=sumAweight_input, old_pi_a=old_pi_a_input, tau=self.tau_input),
                      optimizer=self.opt_G)

        return policy, policy_output

    def create_qfcn(self):
        init_ = self.initializer
        dropout_rate = 0.25
        dense_num = 64

        # K_H = self.k_input
        H = Dense(dense_num, activation='relu')(self.s_input)
        H = Dense(dense_num, activation='relu')(H)
        qfcn_output = Dense(1, activation='linear', init=init_)(H)
        qfcn = Model(input=[self.s_input], output=qfcn_output, name='qfcn_output')
        qfcn.compile(loss='mse', optimizer=self.opt_D)
        return qfcn

    def predict(self, x, n_samples=1):
        dummy_sumA = np.zeros((n_samples, 1))
        dummy_old_pi_a = np.zeros((n_samples, self.dim_action))
        tau = np.tile(self.tau, (n_samples, 1))

        if n_samples == 1:
            n = n_samples
            d = self.dim_action
            pred = self.policy.predict([x, dummy_sumA, dummy_old_pi_a, tau])
            noise = self.explr_const * np.random.randn(n, d)
            return pred + noise
        else:
            n = n_samples
            d = self.dim_action
            pred = self.policy.predict([np.tile(x, (n, 1, 1)), dummy_sumA, dummy_old_pi_a, tau])
            noise = self.explr_const * np.random.randn(n, d)
            return pred + noise

    def compute_advantage_values(self, states, actions, sprimes, rewards, traj_lengths):
        Vsprime = np.array([self.qfcn.predict(s[None, :])[0, 0] \
                                if np.sum(s) != 0 else 0 for s in sprimes])
        n_data = len(Vsprime)
        Vsprime = Vsprime.reshape((n_data, 1))
        Q = rewards + Vsprime
        V = self.qfcn.predict(states)
        A = Q - V
        sumA = []

        for i in range(len(A)):
            try:
                sumA.append(np.sum(A[i:i + traj_lengths[i]]))
            except IndexError:
                break
        return np.array(sumA)[:, None]

    def update_V(self, states, sumR):
        n_data = states.shape[0]
        batch_size = np.min([32, int(len(states) * 0.1)])
        if batch_size == 0:
            batch_size = 1

        checkpointer = ModelCheckpoint(filepath=self.save_folder + '/weights.hdf5',
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True)
        self.qfcn.fit(states, sumR, epochs=20,
                      validation_split=0.1,
                      batch_size=batch_size,
                      verbose=False)
        #self.qfcn.load_weights(self.save_folder + '/weights.hdf5')

    def update_policy(self, states, actions, adv):
        n_data = states.shape[0]
        batch_size = np.min([32, int(len(actions) * 0.1)])
        if batch_size == 0:
            batch_size = 1
        tau = np.tile(self.tau, (n_data, 1))
        old_pi_a = self.policy.predict([states, adv, actions, tau])
        checkpointer = ModelCheckpoint(filepath=self.save_folder + '/pi_weights.hdf5',
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True)
        print "Fitting pi..."
        tau = np.tile(self.tau, (n_data, 1))
        self.policy.fit([states, adv, old_pi_a, tau],
                       actions, epochs=20, validation_split=0.1,
                       batch_size=batch_size,
                       verbose=False)
        print "Done!"
        #self.policy.load_weights(self.save_folder + '/pi_weights.hdf5')

    def train(self, problem, seed, epochs=500, d_lr=1e-3, g_lr=1e-4):
        K.set_value(self.opt_G.lr, g_lr)
        K.set_value(self.opt_D.lr, d_lr)

        print self.opt_G.get_config()

        pfilename = self.save_folder + '/' + str(seed) + '_performance.txt'
        pfile = open(pfilename, 'wb')

        self.n_feasible_trajs = 0
        traj_list = []
        self.pfilename = self.save_folder + '/' + str(seed) + '_performance.txt'
        pfile = open(self.pfilename, 'wb')
        n_data = 0
        n_remains = []
        for i in range(1, epochs):
            self.epoch = i
            print "N simulations %d/%d" % (i, epochs)
            if 'convbelt' in problem.name:
                length_of_rollout = 20
            else:
                length_of_rollout = 10

            for n_iter in range(1):  # N = 5, T = 20, using the notation from PPO paper
                problem.init_saver.Restore()
                problem.objects_currently_not_in_goal = problem.objects
                traj, n_remain = problem.rollout_the_policy(self, length_of_rollout)
                if len(traj['a'])>0:
                    traj_list.append(traj)
                    n_remains.append(n_remain)

            if len(traj['a'])> 0:
                avg_J = self.log_traj_performance([traj_list[-1]], n_remains[-1], i, n_data)
                lowest_possible_reward = -2
                if avg_J > lowest_possible_reward:
                    self.n_feasible_trajs += 1
            else:
                avg_J = self.log_traj_performance(-2.0, 7, i, n_data)

            is_time_to_train = i % 10 == 0
            if is_time_to_train and len(traj_list)>0:
                new_s, new_a, new_r, new_sprime, new_sumR, _, new_traj_lengths = format_RL_data(traj_list)
                n_data += len(new_s)
                self.update_V(new_s, new_sumR)
                new_sumA = self.compute_advantage_values(new_s, new_a, new_sprime, new_r, new_traj_lengths)
                self.update_policy(new_s, new_a, new_sumA)
                traj_list = []
                n_remains = []



