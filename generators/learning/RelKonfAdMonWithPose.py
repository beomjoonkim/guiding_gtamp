from keras.layers import *
from keras.layers.merge import Concatenate
from generators.learning.AdversarialPolicy import AdversarialPolicy
from keras.models import Model


class RelKonfMSEPose(AdversarialPolicy):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        # todo try different weight initializations
        AdversarialPolicy.__init__(self, dim_action, dim_collision, save_folder, tau)

        self.dim_poses = 8
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
        mse_model = Model(inputs=[self.action_input, self.goal_flag_input,
                                  self.key_config_input, self.collision_input],
                          outputs=output,
                          name='q_output')
        mse_model.compile(loss='mse', optimizer=self.opt_D)
        return mse_model

    def construct_relevance_network(self):
        tiled_action = self.get_tiled_input(self.action_input)
        hidden_konf_action_goal_flag = Concatenate(axis=2)([self.key_config_input, tiled_action, self.goal_flag_input])
        dim_combined = hidden_konf_action_goal_flag.shape[2]._value
        hidden_relevance = self.create_conv_layers(hidden_konf_action_goal_flag, dim_combined, use_pooling=False,
                                                   use_flatten=False)
        hidden_relevance = Conv2D(filters=1,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  activation='relu',
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer
                                  )(hidden_relevance)
        hidden_col_relevance = Concatenate(axis=2)([self.collision_input, hidden_relevance])
        hidden_col_relevance = self.create_conv_layers(hidden_col_relevance, n_dim=3, use_pooling=False)

        dense_num = 64
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

    def get_batch(self, states, poses, rel_konfs, actions, sum_rewards, batch_size):
        indices = np.random.randint(0, actions.shape[0], size=batch_size)
        s_batch = np.array(states[indices, :])  # collision vector
        a_batch = np.array(actions[indices, :])
        pose_batch = np.array(poses[indices, :])
        konf_batch = np.array(rel_konfs[indices, :])
        sum_reward_batch = np.array(sum_rewards[indices, :])
        return s_batch, pose_batch, konf_batch, a_batch, sum_reward_batch

    def compute_pure_mse(self, data):
        pred = self.q_mse_model.predict([data['actions'], data['goal_flags'], data['rel_konfs'], data['states']])
        return np.mean(np.power(pred - data['sum_rewards'], 2))

    def train(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards):
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        self.train_data, self.test_data = self.get_train_and_test_data(states, poses, rel_konfs, goal_flags,
                                                                       actions, sum_rewards,
                                                                       train_idxs, test_idxs)
        callbacks = self.create_callbacks_for_pretraining()
        pre_mse = self.compute_pure_mse(self.test_data)
        self.q_mse_model.fit([self.train_data['actions'], self.train_data['goal_flags'],
                              self.train_data['rel_konfs'], self.train_data['states']],
                             self.train_data['sum_rewards'], batch_size=32,
                             epochs=500,
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

    def construct_policy_output(self):
        tiled_obj_pose = self.get_tiled_input(self.pose_input)  # get the object pose
        hidden_konf_action_goal_flag = Concatenate(axis=2)(
            [self.key_config_input, tiled_obj_pose, self.goal_flag_input])
        dim_combined = hidden_konf_action_goal_flag.shape[2]._value
        hidden_relevance = self.create_conv_layers(hidden_konf_action_goal_flag, dim_combined, use_pooling=False,
                                                   use_flatten=False)
        hidden_relevance = Conv2D(filters=1,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  activation='relu',
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer
                                  )(hidden_relevance)

        hidden_col_relevance = Concatenate(axis=2)([self.collision_input, hidden_relevance])
        hidden_col_relevance = self.create_conv_layers(hidden_col_relevance, n_dim=4)

        dense_num = 64
        hidden_place = Dense(dense_num, activation='relu',
                             kernel_initializer=self.kernel_initializer,
                             bias_initializer=self.bias_initializer
                             )(hidden_col_relevance)
        hidden_place = Dense(dense_num, activation='relu',
                             kernel_initializer=self.kernel_initializer,
                             bias_initializer=self.bias_initializer
                             )(hidden_place)
        action_output = Dense(self.dim_action,
                              activation='linear',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.initializer)(hidden_place)
        return action_output

    def construct_policy_model(self):
        mse_model = Model(inputs=[self.action_input, self.goal_flag_input, self.key_config_input, self.collision_input],
                          outputs=self.policy_output,
                          name='q_output')
        mse_model.compile(loss='mse', optimizer=self.opt_D)
        return mse_model

    def train(self, states, poses, rel_konfs, goal_flags, actions, sum_rewards):
        train_idxs, test_idxs = self.get_train_and_test_indices(len(actions))
        train_data, test_data = self.get_train_and_test_data(states, poses, rel_konfs, goal_flags, actions, sum_rewards,
                                                             train_idxs, test_idxs)

        # generate x_1,...,x_m from the generator
        # pick random batch of size m from the real dataset Y
        # compute the nearest neighbor for each x_i

        raise NotImplementedError
