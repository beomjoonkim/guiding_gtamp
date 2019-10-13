from keras.layers import *
from keras.layers.merge import Concatenate
from generators.learning.AdversarialPolicy import AdversarialPolicy
from keras.models import Model
from keras import backend as K

import tensorflow as tf
import time
import os
import socket

if socket.gethostname() == 'lab' or socket.gethostname() == 'phaedra':
    ROOTDIR = './'
else:
    ROOTDIR = '/data/public/rw/pass.port/guiding_gtamp/'


def G_loss(true_actions, pred):
    # pred = Q(G(z))
    # I don't have access to fake and real actions; what to do?
    return -K.mean(pred, axis=-1)




def slice_x(x):
    return x[:, 0:1]


def slice_y(x):
    return x[:, 1:2]


def slice_th(x):
    return x[:, 2:]


class RelKonfMSEPose(AdversarialPolicy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        # todo try different weight initializations
        AdversarialPolicy.__init__(self, dim_action, dim_collision, save_folder, tau)

        self.dim_poses = 8
        self.dim_collision = dim_collision

        self.action_input = Input(shape=(dim_action,), name='a', dtype='float32')  # action
        self.collision_input = Input(shape=dim_collision, name='s', dtype='float32')  # collision vector
        self.pose_input = Input(shape=(self.dim_poses,), name='pose', dtype='float32')  # pose
        self.key_config_input = Input(shape=(615, 4, 1), name='konf', dtype='float32')  # relative key config
        self.goal_flag_input = Input(shape=(615, 4, 1), name='goal_flag',
                                     dtype='float32')  # goal flag (is_goal_r, is_goal_obj)

        # related to detecting whether a key config is relevant
        self.cg_input = Input(shape=(dim_action,), name='cg', dtype='float32')  # action
        self.ck_input = Input(shape=(dim_action,), name='ck', dtype='float32')  # action
        self.collision_at_each_ck = Input(shape=(2,), name='ck', dtype='float32')  # action

        self.weight_file_name = 'admonpose_seed_%d' % config.seed
        self.pretraining_file_name = 'pretrained_%d.h5' % config.seed
        self.seed = config.seed

        self.q_output = self.construct_q_function()
        self.q_mse_model = self.construct_q_mse_model(self.q_output)

        # self.reachability_output = self.construct_reachability_output()
        # self.reachability_model = self.construct_reachability_model()

        self.policy_output = self.construt_self_attention_policy_output()
        # self.policy_output = self.construct_policy_output()
        self.policy_model = self.construct_policy_model()

    def construct_reachability_model(self):
        model = Model(inputs=[self.cg_input, self.ck_input, self.collision_input],
                      outputs=self.reachability_output,
                      name='reachability')
        return model

    def construct_q_mse_model(self, output):
        mse_model = Model(inputs=[self.action_input, self.goal_flag_input, self.pose_input,
                                  self.key_config_input, self.collision_input],
                          outputs=output,
                          name='q_output')
        mse_model.compile(loss='mse', optimizer=self.opt_D)
        return mse_model

    def construct_reachability_output(self):
        # It takes relative key configurations, and the target relative place config wrt to the object as an
        # input, and collisions at each configs, and tells you whether the target relative place config
        # is reachable. How is this related to transformers?

        # In transformer, your output is a function of the current input, and its relevance to all the other
        # inputs; specifically, it is y_i = \sum_{j} w(x_i, x_j) x_i
        dense_num = 64
        x_g = Lambda(slice_x)(self.cg_input)
        y_g = Lambda(slice_y)(self.cg_input)
        th_g = Lambda(slice_th)(self.cg_input)
        x_k = Lambda(slice_x)(self.ck_input)
        y_k = Lambda(slice_y)(self.ck_input)
        th_k = Lambda(slice_th)(self.ck_input)

        Xs = Concatenate(axis=-1)([x_g, x_k])
        Ys = Concatenate(axis=-1)([y_g, y_k])
        Ths = Concatenate(axis=-1)([th_g, th_k])

        H_Xs = Dense(dense_num, activation='relu')(Xs)
        H_Xs = Dense(8, activation='relu')(H_Xs)

        H_Ys = Dense(dense_num, activation='relu')(Ys)
        H_Ys = Dense(8, activation='relu')(H_Ys)

        H_Ths = Dense(dense_num, activation='relu')(Ths)
        H_Ths = Dense(8, activation='relu')(H_Ths)

        H = Concatenate(axis=-1)([H_Xs, H_Ys, H_Ths, self.collision_at_each_ck])
        for i in range(2):
            H = Dense(dense_num, activation='relu')(H)

        H = Dense(1, activation='relu')(H)

        return H

    def construct_policy_output_using_reachability_model(self):
        # input: 1 615 -> relevance
        # I need 615 x 615 x 8 matrix, and the function self.reachability_model applied to each pair of poses
        # The result is 615 x 615 x 1 matrix
        # With this matrix, what do we do?
        # I put an additional layer on each row, so we end up with 615x1 matrix. Call each weight \theta_i
        # Now, I express my output =  \sum_{i=1}^{n} \theta_i * k_i
        # Where does the object pose come into play here? Probably it does not, because it is always the origin
        pass

    def construt_self_attention_policy_output(self):
        tiled_pose = self.get_tiled_input(self.pose_input)
        concat_input = Concatenate(axis=2)(
            [self.key_config_input, self.goal_flag_input, self.collision_input, tiled_pose])
        dim_input = concat_input.shape[2]._value

        # The query matrix
        query = self.create_conv_layers(concat_input, dim_input, use_pooling=False, use_flatten=False)
        query = Conv2D(filters=1,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer)(query)

        # The key matrix
        """
        key = self.create_conv_layers(concat_input, dim_input, use_pooling=False, use_flatten=False)
        key = Conv2D(filters=256,
                     kernel_size=(1, 1),
                     strides=(1, 1),
                     activation='linear',
                     kernel_initializer=self.kernel_initializer,
                     bias_initializer=self.bias_initializer)(key)

        def compute_W(x):
            qvals = x[0]
            kvals = x[1]
            qvals = tf.transpose(qvals, perm=[0, 1, 3, 2])
            dotted = tf.keras.backend.batch_dot(kvals, qvals) /tf.sqrt(tf.dtypes.cast(qvals.shape[2]._value,tf.float32))
            dotted = tf.squeeze(dotted, axis=-1)
            dotted = tf.squeeze(dotted, axis=-1)
            return K.softmax(dotted, axis=-1)

        W = Lambda(compute_W, name='softmax')([query, key])
        """
        def compute_W(x):
            x = K.squeeze(x, axis=-1)
            x = K.squeeze(x, axis=-1)
            return K.softmax(x*100, axis=-1)

        W = Lambda(compute_W, name='softmax')(query)

        self.w_model = Model(
            inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input],
            outputs=W,
            name='w_model')

        # The value matrix
        value = self.create_conv_layers(concat_input, dim_input, use_pooling=False, use_flatten=False)
        value = Conv2D(filters=4,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer,
                       )(value)

        # value = self.key_config_input
        """
        value = Lambda(lambda x: K.squeeze(x, axis=-1))(self.key_config_input)
        """

        value = Lambda(lambda x: K.squeeze(x, axis=2), name='key_config_transformation')(value)
        self.value_model = Model(
            inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input],
            outputs=value,
            name='value_model')
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([W, value])
        return output

    def construct_policy_output(self):
        konf_goal_flag = Concatenate(axis=2)(
            [self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input])
        dim_combined = konf_goal_flag.shape[2]._value
        hidden_relevance = self.create_conv_layers(konf_goal_flag, dim_combined, use_pooling=True,
                                                   use_flatten=True)
        """
        n_conv_filters = 16
        hidden_relevance = Conv2D(filters=n_conv_filters,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  activation='relu',
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer
                                  )(hidden_relevance)
        hidden_relevance = Reshape((615, n_conv_filters, 1))(hidden_relevance)
        #hidden_col_relevance = Concatenate(axis=2)([self.collision_input, hidden_relevance])
        #hidden_col_relevance = self.create_conv_layers(hidden_col_relevance, n_dim=2 + n_conv_filters,
        #                                               use_pooling=False)
        """

        dense_num = 256
        hidden_action = Dense(dense_num, activation='relu',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer)(hidden_relevance)
        hidden_action = Dense(dense_num, activation='relu',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer
                              )(hidden_action)

        action_output = Dense(self.dim_action,
                              activation='linear',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer,
                              name='policy_output')(hidden_action)
        return action_output

    def construct_policy_model(self):
        mse_model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input],
                          outputs=self.policy_output,
                          name='policy_output')
        mse_model.compile(loss='mse', optimizer=self.opt_D)
        return mse_model

    def construct_q_function(self):
        tiled_action = self.get_tiled_input(self.action_input)
        tiled_pose = self.get_tiled_input(self.pose_input)
        hidden_konf_action_goal_flag = Concatenate(axis=2)(
            [self.key_config_input, tiled_pose, tiled_action, self.goal_flag_input])
        dim_combined = hidden_konf_action_goal_flag.shape[2]._value
        hidden_relevance = self.create_conv_layers(hidden_konf_action_goal_flag, dim_combined, use_pooling=False,
                                                   use_flatten=False)
        n_conv_filters = 16
        hidden_relevance = Conv2D(filters=n_conv_filters,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  activation='relu',
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer
                                  )(hidden_relevance)
        hidden_relevance = Reshape((615, n_conv_filters, 1))(hidden_relevance)
        self.relevance_model = Model(inputs=[self.action_input, self.goal_flag_input, self.pose_input,
                                             self.key_config_input, self.collision_input],
                                     outputs=hidden_relevance,
                                     name='q_output')

        hidden_col_relevance = Concatenate(axis=2)([self.collision_input, hidden_relevance])
        hidden_col_relevance = self.create_conv_layers(hidden_col_relevance, n_dim=2 + n_conv_filters,
                                                       use_pooling=False)

        dense_num = 256
        hidden_place = Dense(dense_num, activation='relu',
                             kernel_initializer=self.kernel_initializer,
                             bias_initializer=self.bias_initializer)(hidden_col_relevance)
        hidden_place = Dense(dense_num, activation='relu',
                             kernel_initializer=self.kernel_initializer,
                             bias_initializer=self.bias_initializer
                             )(hidden_place)
        place_value = Dense(1, activation='linear',
                            kernel_initializer=self.kernel_initializer,
                            bias_initializer=self.bias_initializer)(hidden_place)
        q_output = place_value
        return q_output

    def get_train_and_test_data(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards, train_indices,
                                test_indices):
        train = {'states': states[train_indices, :],
                 'poses': poses[train_indices, :],
                 'actions': actions[train_indices, :],
                 'rel_konfs': rel_konfs[train_indices, :],
                 'sum_rewards': sum_rewards[train_indices, :],
                 'goal_flags': goal_flags[train_indices, :]
                 }
        test = {'states': states[test_indices, :],
                'poses': poses[test_indices, :],
                'goal_flags': goal_flags[test_indices, :],
                'actions': actions[test_indices, :],
                'rel_konfs': rel_konfs[test_indices, :],
                'sum_rewards': sum_rewards[test_indices, :]
                }
        return train, test

    def get_batch(self, cols, goal_flags, poses, rel_konfs, actions, sum_rewards, batch_size):
        indices = np.random.randint(0, actions.shape[0], size=batch_size)
        cols_batch = np.array(cols[indices, :])  # collision vector
        goal_flag_batch = np.array(goal_flags[indices, :])  # collision vector
        a_batch = np.array(actions[indices, :])
        pose_batch = np.array(poses[indices, :])
        konf_batch = np.array(rel_konfs[indices, :])
        sum_reward_batch = np.array(sum_rewards[indices, :])
        return cols_batch, goal_flag_batch, pose_batch, konf_batch, a_batch, sum_reward_batch

    def compute_pure_mse(self, data):
        pred = self.q_mse_model.predict(
            [data['actions'], data['goal_flags'], data['poses'], data['rel_konfs'], data['states']])
        return np.mean(np.power(pred - data['sum_rewards'], 2))

    def compute_policy_mse(self, data):
        pred = self.policy_model.predict(
            [data['goal_flags'], data['rel_konfs'], data['states'], data['poses']])
        return np.mean(np.power(pred - data['actions'], 2))

    def train_q_function(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards, epochs=500):
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        self.train_data, self.test_data = self.get_train_and_test_data(states, poses, rel_konfs, goal_flags,
                                                                       actions, sum_rewards,
                                                                       train_idxs, test_idxs)
        callbacks = self.create_callbacks_for_pretraining()
        pre_mse = self.compute_pure_mse(self.test_data)
        self.q_mse_model.fit([self.train_data['actions'], self.train_data['goal_flags'], self.train_data['poses'],
                              self.train_data['rel_konfs'], self.train_data['states']],
                             self.train_data['sum_rewards'], batch_size=32,
                             epochs=epochs,
                             verbose=2,
                             callbacks=callbacks,
                             validation_split=0.1)
        post_mse = self.compute_pure_mse(self.test_data)

        print "Pre-and-post test errors", pre_mse, post_mse

    def train_policy(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards, epochs=500):
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        train_data, test_data = self.get_train_and_test_data(states, poses, rel_konfs, goal_flags,
                                                             actions, sum_rewards,
                                                             train_idxs, test_idxs)
        callbacks = self.create_callbacks_for_pretraining()

        actions = train_data['actions']
        goal_flags = train_data['goal_flags']
        poses = train_data['poses']
        rel_konfs = train_data['rel_konfs']
        collisions = train_data['states']
        pre_mse = self.compute_policy_mse(test_data)
        self.policy_model.fit([goal_flags, rel_konfs, collisions, poses], actions,
                              batch_size=32,
                              epochs=epochs,
                              verbose=2,
                              callbacks=callbacks,
                              validation_split=0.1, shuffle=False)
        post_mse = self.compute_policy_mse(test_data)
        print "Pre-and-post test errors", pre_mse, post_mse
        # wvals = self.W_model.predict([goal_flags, rel_konfs, collisions, poses])[0]
        collision_idxs = collisions[0].squeeze()[:, 0] == True
        import pdb;
        pdb.set_trace()
