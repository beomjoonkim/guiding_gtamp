from RelKonfMSEWithPose import *
from keras.callbacks import *
from weight_regularizers.gershgorin_regularizer import gershgorin_reg


def noise(z_size):
    return np.random.normal(size=z_size).astype('float32')


class RelKonfIMLEPose(RelKonfMSEPose):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        RelKonfMSEPose.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.policy_output = self.construt_self_attention_policy_output()
        self.policy_model = self.construct_policy_model()
        self.q_on_policy_model = self.create_q_on_policy_model()
        self.weight_file_name = 'imle_pose_seed_%d' % config.seed
        self.z_vals_tried = []
        self.num_generated = 1
        self.kernel_initializer = initializers.glorot_uniform()
        self.bias_initializer = initializers.glorot_uniform()

    def create_q_on_policy_model(self):
        for l in self.q_mse_model.layers:
            l.trainable = False

        q_on_policy_output = self.q_mse_model(
            [self.policy_output, self.goal_flag_input, self.pose_input, self.key_config_input, self.collision_input])
        q_on_policy_model = Model(
            inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                    self.noise_input],
            outputs=[q_on_policy_output, self.policy_output])

        q_on_policy_model.compile(loss={'q_output': G_loss, 'policy_output': 'mse'},
                                  optimizer=self.opt_G,
                                  loss_weights={'q_output': 0, 'policy_output': 1},
                                  metrics=[])
        return q_on_policy_model

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

    def create_regularized_conv_layers(self, input, n_dim, use_pooling=True, use_flatten=True):
        n_filters = 32
        H = Conv2D(filters=n_filters,
                   kernel_size=(1, n_dim),
                   strides=(1, 1),
                   activation='linear',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer,
                   kernel_regularizer=gershgorin_reg
                   )(input)
        H = LeakyReLU()(H)
        for _ in range(2):
            H = Conv2D(filters=n_filters,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer,
                       kernel_regularizer=gershgorin_reg
                       )(H)
            H = LeakyReLU()(H)
        if use_pooling:
            H = MaxPooling2D(pool_size=(2, 1))(H)
        if use_flatten:
            H = Flatten()(H)
        return H

    def create_callbacks_for_pretraining(self):
        callbacks = [
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=20)
        ]
        return callbacks

    def save_weights(self, additional_name=''):
        fdir = ROOTDIR + '/' + self.save_folder + '/'
        fname = self.weight_file_name + additional_name + '.h5'
        if not os.path.isdir(fdir):
            os.makedirs(fdir)
        self.policy_model.save_weights(fdir + fname)

    def generate(self, goal_flags, rel_konfs, collisions, poses, z_vals_tried=None):
        z_vals_tried = []
        noise_smpls = noise(z_size=(1, self.dim_action))  # n_data by k matrix
        print noise_smpls
        pred = self.policy_model.predict([goal_flags, rel_konfs, collisions, poses, noise_smpls])
        return pred, z_vals_tried

    def generate_given_noise(self, goal_flags, rel_konfs, collisions, poses, z_vals):
        pred = self.policy_model.predict([goal_flags, rel_konfs, collisions, poses, z_vals])
        return pred

    def get_closest_noise_smpls_for_each_action(self, actions, generated_actions, noise_smpls):
        chosen_noise_smpls = []
        for true_action, generated, noise_smpls_for_action in zip(actions, generated_actions, noise_smpls):
            closest_point, closest_point_idx = self.find_the_idx_of_closest_point_to_x1(true_action, generated)
            noise_that_generates_closest_point_to_true_action = noise_smpls_for_action[closest_point_idx]
            chosen_noise_smpls.append(noise_that_generates_closest_point_to_true_action)
        return np.array(chosen_noise_smpls)

    def construct_policy_model(self):
        model = Model(inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                              self.noise_input],
                      outputs=[self.policy_output],
                      name='policy_model')
        model.compile(loss='mse', optimizer=self.opt_D)
        return model

    def construct_query_output(self, query_input):
        dim_input = query_input.shape[2]._value
        # query = self.create_conv_layers(query_input, dim_input, use_pooling=False, use_flatten=False)
        query = self.create_regularized_conv_layers(query_input, dim_input, use_pooling=False, use_flatten=False)
        query = Conv2D(filters=1,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer,
                       name='query_output')(query)

        def compute_W(x):
            # I need to modify this - but I cannot do argmax? That leads to undefined gradient
            x = K.squeeze(x, axis=-1)
            x = K.squeeze(x, axis=-1)
            return K.softmax(x * 100, axis=-1)  # perhaps 1000 is better; but we need to test this against planning

        W = Lambda(compute_W, name='softmax')(query)
        self.w_model = Model(
            inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input],
            outputs=W,
            name='W_model')

        return W

    def construct_value_output(self, tiled_pose):
        tiled_noise = self.get_tiled_input(self.noise_input)
        concat_value_input = Concatenate(axis=2)(
            [self.key_config_input, self.goal_flag_input, self.collision_input, tiled_pose, tiled_noise])
        dim_value_input = concat_value_input.shape[2]._value

        value = self.create_conv_layers(concat_value_input, dim_value_input, use_pooling=False, use_flatten=False)
        value = Conv2D(filters=4,
                       kernel_size=(1, 1),
                       strides=(1, 1),
                       activation='linear',
                       kernel_initializer=self.kernel_initializer,
                       bias_initializer=self.bias_initializer,
                       name='value_output'
                       )(value)

        # value = self.key_config_input
        value = Lambda(lambda x: K.squeeze(x, axis=2), name='key_config_transformation')(value)
        self.value_model = Model(
            inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input,
                    self.noise_input],
            outputs=value,
            name='value_model')

        return value

    def construt_self_attention_policy_output(self):
        tiled_pose = self.get_tiled_input(self.pose_input)
        concat_input = Concatenate(axis=2)(
            [self.key_config_input, self.goal_flag_input, self.collision_input, tiled_pose])

        W = self.construct_query_output(concat_input)
        value = self.construct_value_output(tiled_pose)
        self.query_output = W
        self.value_output = value
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='policy_output')([W, value])
        return output

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

        data_resampling_step = 1
        num_smpl_per_state = 10

        actions = train_data['actions']
        goal_flags = train_data['goal_flags']
        poses = train_data['poses']
        rel_konfs = train_data['rel_konfs']
        collisions = train_data['states']
        callbacks = self.create_callbacks_for_pretraining()

        gen_w_norm_patience = 10
        gen_w_norms = [-1] * gen_w_norm_patience
        valid_errs = []
        patience = 0
        for epoch in range(epochs):
            print 'Epoch %d/%d' % (epoch, epochs)
            is_time_to_smpl_new_data = epoch % data_resampling_step == 0
            batch_size = 400
            col_batch, goal_flag_batch, pose_batch, rel_konf_batch, a_batch, sum_reward_batch = \
                self.get_batch(collisions, goal_flags, poses, rel_konfs, actions, sum_rewards, batch_size=batch_size)
            if is_time_to_smpl_new_data:
                stime = time.time()
                # train data
                world_states = (goal_flag_batch, rel_konf_batch, col_batch, pose_batch)
                noise_smpls = noise(z_size=(batch_size, num_smpl_per_state, self.dim_action))
                generated_actions = self.generate_k_smples_for_multiple_states(world_states, noise_smpls)
                chosen_noise_smpls = self.get_closest_noise_smpls_for_each_action(a_batch, generated_actions,
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

            self.policy_model.fit([goal_flag_batch, rel_konf_batch, col_batch, pose_batch, chosen_noise_smpls],
                                  [a_batch],
                                  epochs=100,
                                  validation_data=(
                                      [t_goal_flags, t_rel_konfs, t_collisions, t_poses, t_chosen_noise_smpls],
                                      [t_actions]),
                                  callbacks=callbacks,
                                  verbose=False)
            after = self.policy_model.get_weights()
            gen_w_norm = np.linalg.norm(np.hstack([(a - b).flatten() for a, b in zip(before, after)]))
            print "Generator weight norm diff", gen_w_norm
            gen_w_norms[epoch % gen_w_norm_patience] = gen_w_norm

            pred = self.policy_model.predict([t_goal_flags, t_rel_konfs, t_collisions, t_poses, t_chosen_noise_smpls])
            valid_err = np.mean(np.linalg.norm(pred - t_actions, axis=-1))
            valid_errs.append(valid_err)

            if valid_err <= np.min(valid_errs):
                self.save_weights()
                patience = 0
            else:
                patience += 1

            if patience > 50:
                break

            print "Val error", valid_err
            print np.min(valid_errs)
