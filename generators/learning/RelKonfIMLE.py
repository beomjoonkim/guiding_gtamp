from RelKonfAdMonWithPose import *
from keras.callbacks import *


class RelKonfIMLEPose(RelKonfMSEPose):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        RelKonfMSEPose.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.policy_output = self.construt_self_attention_policy_output()
        self.policy_model = self.construct_policy_model()
        self.q_on_policy_model = self.create_q_on_policy_model()
        self.weight_file_name = 'imle_pose_seed_%d' % config.seed
        self.z_vals_tried = []
        self.num_generated=1
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
            TerminateOnNaN(),
            EarlyStopping(monitor='val_loss', min_delta=1e-2, patience=20)
        ]
        """
                tf.keras.callbacks.ModelCheckpoint(filepath=self.save_folder + fname,
                                                   verbose=False,
                                                   save_best_only=True,
                                                   save_weights_only=True),
        """
        return callbacks

    def save_weights(self, additional_name=''):
        fdir = ROOTDIR + '/' + self.save_folder + '/'
        fname = self.weight_file_name + additional_name + '.h5'
        if not os.path.isdir(fdir):
            os.makedirs(fdir)
        self.policy_model.save_weights(fdir + fname)

    def generate(self, goal_flags, rel_konfs, collisions, poses, z_vals_tried=None):
        stime = time.time()
        if z_vals_tried is None or len(z_vals_tried) == 0:
            noise_smpls = self.num_generated/10.0*noise(z_size=(1, self.dim_action))  # n_data by k matrix
            z_vals_tried.append(noise_smpls.squeeze())
            self.num_generated+=1
        else:
            noise_smpls = ((20) * np.random.uniform(size=self.dim_action) - 10)[None, :]
            z_vals_tried = np.array(z_vals_tried)
            min_dist = np.min(np.linalg.norm(noise_smpls - z_vals_tried, axis=-1))
            i = len(z_vals_tried)
            while min_dist < 10.:
                noise_smpls = ((20 + i * 10) * np.random.uniform(size=self.dim_action) - (10 + i * 10))[None, :]
                min_dist = np.min(np.linalg.norm(noise_smpls - z_vals_tried, axis=-1))
                i += 1
                # print i
            z_vals_tried = z_vals_tried.tolist()
            z_vals_tried.append(noise_smpls.squeeze())
        # print "Z sampling time", time.time()-stime
        # stime=time.time()
        pred = self.policy_model.predict([goal_flags, rel_konfs, collisions, poses, noise_smpls])
        # print "Prediction time", time.time() - stime
        return pred, z_vals_tried

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
                      outputs=self.policy_output,
                      name='policy_model')
        model.compile(loss='mse', optimizer=self.opt_D)
        return model

    def construt_self_attention_policy_output(self):
        tiled_pose = self.get_tiled_input(self.pose_input)
        concat_input = Concatenate(axis=2)(
            [self.key_config_input, self.goal_flag_input, self.collision_input, tiled_pose])
        dim_input = concat_input.shape[2]._value

        # This transforms the entire key configurations. We have 615 x n_feature, an embeeding matrix which we call E
        # Computation of E:
        hidden_relevance = self.create_conv_layers(concat_input, dim_input, use_pooling=False, use_flatten=False)
        hidden_relevance = Conv2D(filters=1,
                                  kernel_size=(1, 1),
                                  strides=(1, 1),
                                  activation='linear',
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer)(hidden_relevance)

        self.relevance_model = Model(
            inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input],
            outputs=hidden_relevance,
            name='relevance_model')

        def compute_W(x):
            x = K.squeeze(x, axis=-1)
            x = K.squeeze(x, axis=-1)
            return K.softmax(x, axis=-1)

        W = Lambda(compute_W)(hidden_relevance)
        self.W_model = Model(
            inputs=[self.goal_flag_input, self.key_config_input, self.collision_input, self.pose_input],
            outputs=W,
            name='w_model')

        ###### Key config transformation
        tiled_noise = self.get_tiled_input(self.noise_input)
        concat_konf_noise = Concatenate(axis=-1)([self.key_config_input, tiled_noise])
        dim_input = concat_konf_noise.shape[2]._value
        n_filters = 64
        H = Conv2D(filters=n_filters,
                   kernel_size=(1, dim_input),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(concat_konf_noise)
        H = Conv2D(filters=n_filters,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation='relu',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        H = Conv2D(filters=4,
                   kernel_size=(1, 1),
                   strides=(1, 1),
                   activation='linear',
                   kernel_initializer=self.kernel_initializer,
                   bias_initializer=self.bias_initializer)(H)
        key_configs = Lambda(lambda x: K.squeeze(x, axis=2))(H)
        #################################################################

        output = Lambda(lambda x: K.batch_dot(x[0], x[1]), name='policy_output')([W, key_configs])
        return output

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
        h_noise = Concatenate(axis=-1)([hidden_col_relevance, self.noise_input])
        hidden_action = Dense(dense_num, activation='relu',
                              kernel_initializer=self.kernel_initializer,
                              bias_initializer=self.bias_initializer)(h_noise)
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

        gen_w_norm_patience = 10
        gen_w_norms = [-1] * gen_w_norm_patience
        valid_errs = []
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
                                       epochs=100,
                                       validation_data=(
                                           [t_goal_flags, t_rel_konfs, t_collisions, t_poses, t_chosen_noise_smpls],
                                           [t_actions]),
                                       callbacks=callbacks,
                                       verbose=False)
            # I think for this, you want to keep the validation batch, and stop if the validation error is high
            # fname = self.weight_file_name + '.h5'
            # self.q_on_policy_model.load_weights(self.save_folder + fname)
            after = self.policy_model.get_weights()
            gen_w_norm = np.linalg.norm(np.hstack([(a - b).flatten() for a, b in zip(before, after)]))
            print "Generator weight norm diff", gen_w_norm
            gen_w_norms[epoch % gen_w_norm_patience] = gen_w_norm

            pred = self.policy_model.predict([t_goal_flags, t_rel_konfs, t_collisions, t_poses, t_chosen_noise_smpls])
            valid_err = np.mean(np.linalg.norm(pred - t_actions, axis=-1))
            valid_errs.append(valid_err)

            if valid_err <= np.min(valid_errs):
                self.save_weights()
            print "Val error", valid_err
            print np.min(valid_errs)
            # if np.all(np.array(gen_w_norms) == 0):
            #    break
