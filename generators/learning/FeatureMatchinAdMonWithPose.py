from AdMonWithPose import AdversarialMonteCarloWithPose


class FeatureMatchingAdMonWithPose(AdversarialMonteCarloWithPose):
    def __init__(self, dim_action, dim_collision, save_folder, tau, config):
        AdversarialMonteCarloWithPose.__init__(self, dim_action, dim_collision, save_folder, tau, config)
        self.target_feature_match_input = Input(shape=(64,), name='feature', dtype='float32')  # action

    def create_discriminator(self):
        disc_output = self.get_disc_output_with_preprocessing_layers()
        self.disc_output = disc_output
        disc = Model(inputs=[self.action_input, self.collision_input, self.pose_input, self.tau_input],
                     outputs=disc_output,
                     name='disc_output')
        disc.compile(loss=tau_loss(self.tau_input), optimizer=self.opt_D)

        self.discriminator_feature_matching_model = Model(
            inputs=[self.action_input, self.collision_input, self.pose_input],
            outputs=self.discriminator_feature_matching_layer,
            name='feature_matching_model')

        self.discriminator_feature_matching_model.compile(loss='mse', optimizer=self.opt_D)
        return disc

    def createGAN(self):
        disc = self.create_discriminator()
        a_gen, a_gen_output = self.create_generator()
        for l in disc.layers:
            l.trainable = False
        DG_output = self.discriminator_feature_matching_model([a_gen_output, self.collision_input, self.pose_input])
        DG = Model(inputs=[self.noise_input, self.collision_input, self.pose_input], outputs=[DG_output])
        DG.compile(loss='mse', optimizer=self.opt_G)

        return a_gen, disc, DG

    def train(self, states, poses, actions, sum_rewards, epochs=500, d_lr=1e-3, g_lr=1e-4):
        batch_size = np.min([32, int(len(actions) * 0.1)])
        if batch_size == 0:
            batch_size = 1
        print batch_size

        curr_tau = self.tau
        K.set_value(self.opt_G.lr, g_lr)
        K.set_value(self.opt_D.lr, d_lr)
        print self.opt_G.get_config()

        for i in range(1, epochs):
            self.compare_to_data(states, poses, actions)
            stime = time.time()
            tau_values = np.tile(curr_tau, (batch_size * 2, 1))
            print "Current tau value", curr_tau
            gen_before = self.a_gen.get_weights()
            disc_before = self.disc.get_weights()
            batch_idxs = range(0, actions.shape[0], batch_size)
            for k, idx in enumerate(batch_idxs):
                s_batch, pose_batch, a_batch, sum_rewards_batch = self.get_batch(states, poses, actions,
                                                                                 sum_rewards,
                                                                                 batch_size)

                # train \hat{S}
                # make fake and reals
                a_z = noise(batch_size, self.dim_noise)
                fake = self.a_gen.predict([a_z, s_batch, pose_batch])
                real = a_batch

                # make their scores
                fake_action_q = np.ones((batch_size, 1)) * INFEASIBLE_SCORE  # marks fake data
                real_action_q = sum_rewards_batch.reshape((batch_size, 1))
                batch_a_for_disc = np.vstack([fake, real])
                batch_s_for_disc = np.vstack([s_batch, s_batch])
                batch_rp_for_disc = np.vstack([pose_batch, pose_batch])
                batch_scores = np.vstack([fake_action_q, real_action_q])
                self.disc.fit(
                    {'a': batch_a_for_disc, 's': batch_s_for_disc, 'pose': batch_rp_for_disc, 'tau': tau_values},
                    batch_scores,
                    epochs=1,
                    verbose=False)

                # train G
                a_z = noise(batch_size, self.dim_noise)
                feature_to_match = self.discriminator_feature_matching_model.predict([a_batch, s_batch, pose_batch])
                self.DG.fit({'z': a_z, 's': s_batch, 'pose': pose_batch}, feature_to_match, epochs=1, verbose=0)

                tttau_values = np.tile(curr_tau, (batch_size, 1))
                a_z = noise(batch_size, self.dim_noise)
                s_batch, pose_batch, a_batch, sum_rewards_batch = self.get_batch(states, poses, actions, sum_rewards,
                                                                                 batch_size)
                real_score_values = np.mean((self.disc.predict([a_batch, s_batch, pose_batch, tttau_values]).squeeze()))
                fake_score_values = np.mean((self.disc.predict([fake, s_batch, pose_batch, tttau_values]).squeeze()))
                # print "Real %.4f Gen %.4f" % (real_score_values, fake_score_values)

                if real_score_values <= fake_score_values:
                    g_lr = 1e-4 / (1 + 1e-1 * i)
                    d_lr = 1e-3 / (1 + 1e-1 * i)
                    K.set_value(self.opt_G.lr, g_lr)
                    K.set_value(self.opt_D.lr, d_lr)
                else:
                    g_lr = 1e-3 / (1 + 1e-1 * i)
                    d_lr = 1e-4 / (1 + 1e-1 * i)
                    K.set_value(self.opt_G.lr, g_lr)
                    K.set_value(self.opt_D.lr, d_lr)

            gen_after = self.a_gen.get_weights()
            disc_after = self.disc.get_weights()
            gen_w_norm = np.linalg.norm(np.hstack([(a - b).flatten() for a, b in zip(gen_before, gen_after)]))
            disc_w_norm = np.linalg.norm(np.hstack([(a - b).flatten() for a, b in zip(disc_before, disc_after)]))

            print 'Completed: %d / %d' % (i, float(epochs))
            print "g_lr %.5f d_lr %.5f" % (g_lr, d_lr)
            # curr_tau = curr_tau * 1 /
            curr_tau = self.tau / (1.0 + 1e-1 * i)
            # curr_tau = self.tau / (1.0 + 1e-1 * i)
            if i > 20:
                self.save_weights(additional_name='_epoch_' + str(i))
            self.compare_to_data(states, poses, actions)
            a_z = noise(len(states), self.dim_noise)

            tttau_values = np.tile(curr_tau, (len(states), 1))
            print "Real %.4f Gen %.4f" % (real_score_values, fake_score_values)
            print "Discriminiator MSE error", np.mean(np.linalg.norm(
                np.array(sum_rewards).squeeze() - self.disc.predict([actions, states, poses, tttau_values]).squeeze()))
            print "Epoch took: %.2fs" % (time.time() - stime)
            print "Generator weight norm diff", gen_w_norm
            print "Disc weight norm diff", disc_w_norm
            print "================================"
