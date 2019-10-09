from keras.layers import *
from keras.layers.merge import Concatenate
from generators.learning.AdversarialPolicy import AdversarialPolicy
from keras.models import Model

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


def noise(z_size):
    return np.random.normal(size=z_size).astype('float32')


class RelKonfMSEPose(AdversarialPolicy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        # todo try different weight initializations
        AdversarialPolicy.__init__(self, dim_action, dim_collision, save_folder, tau)

        self.dim_poses = 4
        self.dim_collision = dim_collision

        self.action_input = Input(shape=(dim_action,), name='a', dtype='float32')  # action
        self.collision_input = Input(shape=dim_collision, name='s', dtype='float32')  # collision vector
        self.pose_input = Input(shape=(self.dim_poses,), name='pose', dtype='float32')  # pose
        self.key_config_input = Input(shape=(615, 3, 1), name='konf', dtype='float32')  # relative key config
        self.goal_flag_input = Input(shape=(615, 4, 1), name='goal_flag',
                                     dtype='float32')  # goal flag (is_goal_r, is_goal_obj)

        self.weight_file_name = 'admonpose_seed_%d' % config.seed
        self.pretraining_file_name = 'pretrained_%d.h5' % config.seed
        self.seed = config.seed
        self.q_output = self.construct_relevance_network()
        self.q_mse_model = self.construct_mse_model(self.q_output)

    def construct_mse_model(self, output):
        mse_model = Model(inputs=[self.action_input, self.goal_flag_input, self.pose_input,
                                  self.key_config_input, self.collision_input],
                          outputs=output,
                          name='q_output')
        mse_model.compile(loss='mse', optimizer=self.opt_D)
        return mse_model

    def construct_relevance_network(self):
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

    def train(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards, epochs=500):
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


class RelKonfIMLEPose(RelKonfMSEPose):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        RelKonfMSEPose.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.policy_output = self.construct_policy_output()
        self.policy_model = self.construct_policy_model()
        self.q_on_policy_model = self.create_q_on_policy_model()
        self.weight_file_name = 'imle_pose_seed_%d' % config.seed
        # self.q_mse_model.load_weights(self.save_folder+'pretrained_%d.h5' % config.seed)

    def create_q_on_policy_model(self):
        for l in self.q_mse_model.layers:
            l.trainable = False
            # for some obscure reason, disc weights still get updated when self.disc.fit is called
            # I speculate that this has to do with the status of the layers at the time it was compiled
        q_on_policy_output = self.q_mse_model(
            [self.policy_output, self.goal_flag_input, self.pose_input, self.key_config_input, self.collision_input])
        q_on_policy_model = Model(
            inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                    self.noise_input],
            # outputs=[q_on_policy_output, self.policy_output])
            outputs=[self.policy_output])
        """
        q_on_policy_model.compile(loss={'q_output': G_loss, 'policy_output': 'mse'},
                                  optimizer=self.opt_G,
                                  loss_weights={'q_output': 0, 'policy_output': 1},
                                  metrics=[])
        """
        q_on_policy_model.compile(loss={'policy_output': 'mse'},
                                  optimizer=self.opt_G,
                                  loss_weights={'policy_output': 1},
                                  metrics=[])

        # but when do I train the q_mse_model?
        return q_on_policy_model

    def construct_policy_output(self):
        # todo make this architecture
        tiled_pose = self.get_tiled_input(self.pose_input)
        konf_goal_flag = Concatenate(axis=2)(
            [self.key_config_input, tiled_pose, self.goal_flag_input])
        dim_combined = konf_goal_flag.shape[2]._value
        hidden_relevance = self.create_conv_layers(konf_goal_flag, dim_combined, use_pooling=False,
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
        hidden_col_relevance = Concatenate(axis=2)([self.collision_input, hidden_relevance])
        hidden_col_relevance = self.create_conv_layers(hidden_col_relevance, n_dim=2 + n_conv_filters,
                                                       use_pooling=False)

        dense_num = 256
        hidden_action = Dense(dense_num, activation='relu',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer)(hidden_col_relevance)
        hidden_action = Dense(dense_num, activation='relu',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer
                              )(hidden_action)

        h_noise = Concatenate(axis=-1)([hidden_action, self.noise_input])
        action_output = Dense(self.dim_action,
                              activation='linear',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer,
                              name='policy_output')(h_noise)
        return action_output

    def construct_policy_model(self):
        mse_model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                                  self.noise_input],
                          outputs=self.policy_output,
                          name='q_output')
        mse_model.compile(loss='mse', optimizer=self.opt_D)
        return mse_model

    def generate_k_smples_for_multiple_states(self, states, noise_smpls):
        goal_flags, rel_konfs, collisions, poses = states
        n_data = len(goal_flags)
        k_smpls = []
        k = noise_smpls.shape[1]

        for j in range(k):
            actions = self.policy_model.predict([goal_flags, rel_konfs, collisions, poses, noise_smpls[:, j, :]])
            k_smpls.append(actions)
        new_k_smpls = np.array(k_smpls).swapaxes(0, 1)

        return new_k_smpls

    def find_the_idx_of_closest_point_to_x1(self, x1, database):
        l2_distances = np.linalg.norm(x1 - database, axis=-1)
        return database[np.argmin(l2_distances)], np.argmin(l2_distances)

    def verify_the_noise_generates_the_closest_pt_to_the_true_action(self, closest_noise, noise_smpls, true_action):
        pass

    def create_callbacks_for_pretraining(self):
        fname = self.weight_file_name + '.h5'
        callbacks = [
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=20),
            tf.keras.callbacks.ModelCheckpoint(filepath=self.save_folder + fname,
                                               verbose=False,
                                               save_best_only=True,
                                               save_weights_only=True),
        ]
        return callbacks

    def save_weights(self, additional_name=''):
        fdir = ROOTDIR + '/' + self.save_folder + '/'
        fname = self.weight_file_name + additional_name + '.h5'
        if not os.path.isdir(fdir):
            os.makedirs(fdir)
        self.policy_model.save_weights(fdir + fname)

    def generate(self, goal_flags, rel_konfs, collisions, poses):
        noise_smpls = noise(z_size=(1, self.dim_action))  # n_data by k matrix
        return self.policy_model.predict([goal_flags, rel_konfs, collisions, poses, noise_smpls])

    def get_closest_noise_smpls_for_each_action(self, actions, generated_actions, noise_smpls):
        chosen_noise_smpls = []
        for true_action, generated, noise_smpls_for_action in zip(actions, generated_actions, noise_smpls):
            closest_point, closest_point_idx = self.find_the_idx_of_closest_point_to_x1(true_action, generated)
            noise_that_generates_closest_point_to_true_action = noise_smpls_for_action[closest_point_idx]
            chosen_noise_smpls.append(noise_that_generates_closest_point_to_true_action)
        return np.array(chosen_noise_smpls)

    def train(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards, epochs=1000):
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        train_data, test_data = self.get_train_and_test_data(states, poses, rel_konfs, goal_flags, actions, sum_rewards,
                                                             train_idxs, test_idxs)

        t_actions = test_data['actions']
        t_goal_flags = test_data['goal_flags']
        t_poses = test_data['poses']
        t_rel_konfs = test_data['rel_konfs']
        t_collisions = test_data['states']
        n_test_data = len(t_collisions)

        # generate x_1,...,x_m from the generator
        # pick random batch of size m from the real dataset Y
        # compute the nearest neighbor for each x_i
        n_data = len(train_idxs)
        data_resampling_step = 1
        num_smpl_per_state = 10

        actions = train_data['actions']
        goal_flags = train_data['goal_flags']
        poses = train_data['poses']
        rel_konfs = train_data['rel_konfs']
        collisions = train_data['states']
        callbacks = self.create_callbacks_for_pretraining()
        for epoch in range(epochs):
            is_time_to_smpl_new_data = epoch % data_resampling_step == 0
            batch_size = 160
            col_batch, goal_flag_batch, pose_batch, rel_konf_batch, a_batch, sum_reward_batch = \
                self.get_batch(collisions, goal_flags, poses, rel_konfs, actions, sum_rewards, batch_size=batch_size)
            if is_time_to_smpl_new_data:
                stime = time.time()
                # train data
                world_states = (goal_flag_batch, rel_konf_batch, col_batch, pose_batch)
                noise_smpls = noise(z_size=(batch_size, num_smpl_per_state, self.dim_action))
                generated_actions = self.generate_k_smples_for_multiple_states(world_states, noise_smpls)
                chosen_noise_smpls = self.get_closest_noise_smpls_for_each_action(actions, generated_actions,
                                                                                  noise_smpls)

                # validation data
                t_world_states = (t_goal_flags, t_rel_konfs, t_collisions, t_poses)
                t_noise_smpls = noise(z_size=(n_test_data, num_smpl_per_state, self.dim_action))
                t_generated_actions = self.generate_k_smples_for_multiple_states(t_world_states, t_noise_smpls)
                t_chosen_noise_smpls = self.get_closest_noise_smpls_for_each_action(t_actions, t_generated_actions,
                                                                                    t_noise_smpls)

                print "Data generation time", time.time() - stime

            # I also need to tag on the Q-learning objective
            before = self.policy_model.get_weights()
            # [self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input, self.noise_input]
            self.q_on_policy_model.fit([goal_flag_batch, rel_konf_batch, col_batch, pose_batch, chosen_noise_smpls],
                                       [a_batch],
                                       epochs=1000,
                                       validation_data=(
                                           [t_goal_flags, t_rel_konfs, t_collisions, t_poses, t_chosen_noise_smpls],
                                           [t_actions]),
                                       callbacks=callbacks)
            # I think for this, you want to keep the validation batch, and stop if the validation error is high
            fname = self.weight_file_name + '.h5'
            self.q_on_policy_model.load_weights(self.save_folder + fname)
            after = self.policy_model.get_weights()
            gen_w_norm = np.linalg.norm(np.hstack([(a - b).flatten() for a, b in zip(before, after)]))
            print "Generator weight norm diff", gen_w_norm
