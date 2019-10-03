import tensorflow as tf
import openravepy
from manipulation.regions import AARegion
import pickle
import numpy as np
from data_traj import make_one_hot_encoded_edge, make_one_hot_encoded_node


class GNN(object):
    def __init__(self, num_entities, num_node_features, num_edge_features, config=None, entity_names=None):
        self.num_entities = num_entities
        self.activation = 'relu'
        self.config = config
        self.top_k = config.top_k
        self.weight_file_name = self.create_weight_file_name()
        self.optimizer = self.create_optimizer(config.optimizer, config.lr)
        self.num_node_features = num_node_features
        self.node_input, self.edge_input, self.action_input, self.op_input, self.cost_input = \
            self.create_inputs(num_entities, num_node_features, num_edge_features, dim_action=[1])
        self.n_node_features = num_node_features

        q_layer, value_layer = self.create_value_function_layers(config)
        self.loss_model = self.create_loss_model(q_layer, value_layer)
        self.q_model = self.make_model([self.node_input, self.edge_input, self.action_input], q_layer, 'qmodel')
        if entity_names is not None:
            self.entity_names = entity_names
            # idxs = np.random.randint(0,len(entity_names), size=(len(entity_names),1))
            # self.entity_name_to_idx = {name: idx[0] for name, idx in zip(entity_names, idxs)}
            self.entity_name_to_idx = {name: idx for idx, name in enumerate(entity_names)}
            print self.entity_name_to_idx

    def load_weights(self):
        """
        self.weight_file_name ='/home/beomjoon//Documents/github/qqq/learn/q-function-weights/' \
                               'Q_weight_n_msg_passing_1_mse_weight_1.0_optimizer_' \
                               'adam_seed_0_lr_0.0001_operator_two_arm_pick_two_arm_place_n_layers_2_n_hidden_32' \
                               '_top_k_1_num_train_5000_loss_largemargin.hdf5'
        """
        self.weight_file_name = './learn/q-function-weights/' \
                                'Q_weight_n_msg_passing_1_mse_weight_1.0_optimizer_' \
                                'adam_seed_%d_lr_0.0001_operator_two_arm_pick_two_arm_place_n_layers_2_n_hidden_32' \
                                '_top_k_1_num_train_5000_loss_%s.hdf5' % (self.config.seed, self.config.loss)
        print "Loading weight", self.weight_file_name
        self.loss_model.load_weights(self.weight_file_name)

    def __deepcopy__(self, _):
        return self

    def create_inputs(self, num_entities, num_node_features, num_edge_features, dim_action):
        node_shape = (num_entities, num_node_features)
        edge_shape = (num_entities, num_entities, num_edge_features)

        knodes = tf.keras.Input(shape=list(node_shape), name='nodes')
        kedges = tf.keras.Input(shape=list(edge_shape), name='edges')
        kactions = tf.keras.Input(shape=list(dim_action), dtype=tf.int32, name='actions')
        koperators = tf.keras.Input(shape=[1], dtype=tf.int32)
        kcosts = tf.keras.Input(shape=[1], dtype=tf.int32)

        return knodes, kedges, kactions, koperators, kcosts

    def get_config(self):
        return {
            'name': 'Model',
            'config': self.config.__dict__,
        }

    def eval(self, nodes, edges, actions):
        pass

    def create_sender_dest_edge_concatenation_lambda_layer(self):
        def concatenate_src_dest_edge_fcn(src_node_tensor, dest_node_tensor, edge_tensor,
                                          transposed_edge_node_tensor):
            n_entities = self.num_entities
            src_repetitons = [1, 1, n_entities, 1]  # repeat in the columns
            repeated_srcs = tf.tile(tf.expand_dims(src_node_tensor, -2), src_repetitons)
            dest_repetitons = [1, n_entities, 1, 1]  # repeat in the rows
            repeated_dests = tf.tile(tf.expand_dims(dest_node_tensor, 1), dest_repetitons)
            src_dest_edge_feature = tf.concat(
                [repeated_srcs, repeated_dests, edge_tensor, transposed_edge_node_tensor], axis=-1)
            return src_dest_edge_feature

        concat_layer = tf.keras.layers.Lambda(lambda args: concatenate_src_dest_edge_fcn(*args), name='concat')
        return concat_layer

    @staticmethod
    def create_aggregation_lambda_layer():
        def aggregate_msg(msgs):
            return tf.reduce_mean(msgs, axis=1)

        aggregation_layer = tf.keras.layers.Lambda(lambda msgs: aggregate_msg(msgs), name='msg_aggregation')
        return aggregation_layer

    @staticmethod
    def make_model(input, network, name):
        return tf.keras.Model(input, network, name=name)

    def create_msg_computaton_layers(self, input, num_latent_features, name, n_layers):
        n_node_features = self.num_node_features

        n_dim_last_layer = num_latent_features if self.config.n_msg_passing == 0 else n_node_features
        # n_dim_last_layer = n_node_features
        if n_layers == 1:
            h = tf.keras.layers.Conv2D(n_dim_last_layer, kernel_size=(1, 1),
                                       kernel_initializer=self.config.weight_initializer,
                                       bias_initializer=self.config.weight_initializer,
                                       name=name + "_0",
                                       activation='linear')(input)
            # since this represents, roughly, the value of an entity,
            # it should range from -inf to inf
            # Perhaps I could use tanh?
        else:
            h = tf.keras.layers.Conv2D(num_latent_features, kernel_size=(1, 1),
                                       kernel_initializer=self.config.weight_initializer,
                                       bias_initializer=self.config.weight_initializer,
                                       name=name + "_0",
                                       activation=self.activation)(input)
            idx = -1
            for idx in range(n_layers - 2):
                h = tf.keras.layers.Conv2D(num_latent_features, kernel_size=(1, 1),
                                           kernel_initializer=self.config.weight_initializer,
                                           bias_initializer=self.config.weight_initializer,
                                           name=name + "_" + str(idx + 1),
                                           activation=self.activation)(h)
            h = tf.keras.layers.Conv2D(n_dim_last_layer, kernel_size=(1, 1),
                                       kernel_initializer=self.config.weight_initializer,
                                       bias_initializer=self.config.weight_initializer,
                                       name=name + "_" + str(idx + 2),
                                       activation='linear')(h)
        return h

    def create_msg_computation_model(self, num_latent_features, name, n_layers):
        n_entities = self.num_entities
        place_holder_input = tf.keras.Input(shape=(n_entities, n_entities, num_latent_features * 4))
        h = self.create_msg_computaton_layers(place_holder_input, num_latent_features, name, n_layers)
        h_model = self.make_model(place_holder_input, h, 'msg_computation')
        return h_model

    def create_graph_network_layers_with_different_msg_passing_network(self, config):
        # this implements different weights at each msg passing
        num_latent_features = config.n_hidden
        num_layers = config.n_layers

        sender_network = self.create_multilayers(self.node_input, num_latent_features, 'src', num_layers)
        dest_network = self.create_multilayers(self.node_input, num_latent_features, 'dest', num_layers)
        edge_network = self.create_multilayers(self.edge_input, num_latent_features, 'edge', num_layers)
        # wait, who is deciding that I am feeding the destination node value? That is, how do I know I am applying
        # dest_network to the neighbor nodes? I tile everything according to rows (sources) and columns (dests)

        concat_lambda_layer = self.create_sender_dest_edge_concatenation_lambda_layer()
        concat_layer = concat_lambda_layer([sender_network, dest_network, edge_network])
        msg_network = self.create_msg_computaton_layers(concat_layer, num_latent_features, 'msgs', num_layers)

        aggregation_lambda_layer = self.create_aggregation_lambda_layer()
        msg_aggregation_layer = aggregation_lambda_layer(msg_network)  # aggregates msgs from neighbors

        # rounds of msg passing
        for i in range(config.n_msg_passing):
            sender_network = self.create_multilayers(msg_aggregation_layer, num_latent_features, 'src' + str(i),
                                                     num_layers)
            dest_network = self.create_multilayers(msg_aggregation_layer, num_latent_features, 'dest' + str(i),
                                                   num_layers)
            edge_network = self.create_multilayers(self.edge_input, num_latent_features, 'edge' + str(i), num_layers)
            concat_layer = concat_lambda_layer([sender_network, dest_network, edge_network])
            msg_network = self.create_msg_computaton_layers(concat_layer, num_latent_features, 'msgs' + str(i),
                                                            num_layers)
            msg_aggregation_layer = aggregation_lambda_layer(msg_network)  # aggregates msgs from neighbors

        return msg_aggregation_layer

    def create_graph_network_layers_with_same_msg_passing_network(self, config):
        num_latent_features = config.n_hidden
        num_layers = config.n_layers
        same_model_for_sender_and_dest = config.same_vertex_model

        # create networks for vertices
        if same_model_for_sender_and_dest:
            vertex_model = self.create_multilayer_model(self.node_input, num_latent_features, 'vertex', num_layers)
            vertex_network = vertex_model(self.node_input)
        else:
            sender_model = self.create_multilayer_model(self.node_input, num_latent_features, 'src', num_layers)
            dest_model = self.create_multilayer_model(self.node_input, num_latent_features, 'dest', num_layers)
            sender_network = sender_model(self.node_input)
            dest_network = dest_model(self.node_input)

        # create edge networks
        edge_model = self.create_multilayer_model(self.edge_input, num_latent_features, 'edge', num_layers)
        edge_network = edge_model(self.edge_input)

        def transpose(inp):
            return tf.transpose(inp, perm=[0, 2, 1, 3])

        transpose_layer = tf.keras.layers.Lambda(lambda inp: transpose(inp), name='test')
        transpose_layer = transpose_layer(self.edge_input)
        transposed_edge_network = edge_model(transpose_layer)  # for reversing the roles of src and dests

        # create concatenation, msg computation, and aggregation layers
        concat_lambda_layer = self.create_sender_dest_edge_concatenation_lambda_layer()
        msg_model = self.create_msg_computation_model(num_latent_features, 'msgs', num_layers)
        aggregation_lambda_layer = self.create_aggregation_lambda_layer()

        if same_model_for_sender_and_dest:
            concat_layer = concat_lambda_layer([vertex_network, vertex_network, edge_network, transposed_edge_network])
        else:
            concat_layer = concat_lambda_layer([sender_network, dest_network, edge_network, transposed_edge_network])

        msg_network = msg_model(concat_layer)
        msg_aggregation_layer = aggregation_lambda_layer(msg_network)  # aggregates msgs from neighbors

        # rounds of msg passing
        for i in range(config.n_msg_passing):
            if same_model_for_sender_and_dest:
                vertex_network = vertex_model(msg_aggregation_layer)
                concat_layer = concat_lambda_layer(
                    [vertex_network, vertex_network, edge_network, transposed_edge_network])
            else:
                sender_network = sender_model(msg_aggregation_layer)
                dest_network = dest_model(msg_aggregation_layer)
                concat_layer = concat_lambda_layer(
                    [sender_network, dest_network, edge_network, transposed_edge_network])
            msg_network = msg_model(concat_layer)
            msg_aggregation_layer = aggregation_lambda_layer(msg_network)  # aggregates msgs from neighbors
            # todo it is saturating here when I use relu?

        # for testing purpose; delete it later
        inputs = [self.node_input, self.edge_input, self.action_input]

        self.msg_model = self.make_model(inputs, msg_network, 'msg_model')
        self.concat_model = self.make_model(inputs, concat_layer, 'concat_model')
        if same_model_for_sender_and_dest:
            self.sender_model = vertex_model  # sender_model
            self.dest_model = vertex_model  # dest_model
        else:
            self.sender_model = sender_model  # sender_model
            self.dest_model = dest_model  # dest_model

        self.edge_model = edge_model
        return msg_aggregation_layer

    def create_value_function_layers(self, config):
        if config.diff_weight_msg_passing:
            msg_aggregation_layer = self.create_graph_network_layers_with_different_msg_passing_network(config)
        else:
            msg_aggregation_layer = self.create_graph_network_layers_with_same_msg_passing_network(config)

        value_layer = tf.keras.layers.Conv1D(1, kernel_size=1,
                                             kernel_initializer=self.config.weight_initializer,
                                             bias_initializer=self.config.weight_initializer,
                                             name='value_layer',
                                             activation='linear')(msg_aggregation_layer)

        def compute_q(values, action_idxs):
            values = tf.squeeze(values)
            num_entities = self.num_entities
            attention = tf.squeeze(tf.one_hot(action_idxs, num_entities, dtype=tf.float32))
            q_value = tf.reduce_sum(attention * values, axis=-1)
            return q_value


        q_layer = tf.keras.layers.Lambda(lambda args: compute_q(*args))
        q_layer = q_layer([value_layer, self.action_input])

        # testing purpose
        inputs = [self.node_input, self.edge_input, self.action_input]
        self.value_model = self.make_model(inputs, value_layer, 'value_model')
        self.aggregation_model = self.make_model(inputs, msg_aggregation_layer, 'value_aggregation')
        return q_layer, value_layer

    def create_loss_model(self, q_layer, value_layer):
        def compute_rank_loss(alt_msgs, target_msg, costs):
            alt_msgs = tf.squeeze(alt_msgs)
            sorted_top_k_q_vals = tf.nn.top_k(alt_msgs, self.top_k, sorted=True)[0]  # sorts it in descending order
            min_of_top_k_q_val = sorted_top_k_q_vals[:, -1]
            q_delta = target_msg - min_of_top_k_q_val
            action_ranking_cost = 1.0 - q_delta
            hinge_loss = tf.reduce_mean(tf.maximum(tf.cast(0., tf.float32), action_ranking_cost))
            return hinge_loss

        def compute_mse_loss(msgs, target_msg, costs):
            return tf.losses.mean_squared_error(target_msg, -costs)

        # https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf
        def compute_dql_loss(msgs, target_msg, costs):
            msgs = tf.squeeze(msgs)
            target_msg = tf.squeeze(target_msg)
            costs = tf.squeeze(costs)

            gamma = .99
            final = tf.cast(costs[:-1] <= 2, tf.float32)
            reward = -2. + 101. * final
            q_s_a = target_msg[:-1]
            q_sprime_aprime = tf.reduce_max(msgs, -1)[1:]
            y = reward + gamma * q_sprime_aprime * (1 - final)
            return tf.losses.mean_squared_error(q_s_a, y)

        def get_alt_msgs(msgs, action_idxs):
            values = tf.squeeze(msgs)
            num_entities = self.num_entities
            attention = tf.squeeze(1 - tf.one_hot(action_idxs, num_entities, dtype=tf.float32))
            alt_msgs = tf.squeeze(attention * values)

            # For some reason, I cannot use boolean_mask in the loss functions.
            # I am speculating that it has to do with the fact that the loss functions receive a batch of data,
            # instead of a single data point
            # This is a workaround for making tf.math.top_k to not choose the action seen in the planning experience
            neg_inf_attention = tf.squeeze(tf.one_hot(action_idxs, num_entities, dtype=tf.float32)) * -2e32
            alt_msgs = alt_msgs + neg_inf_attention

            return alt_msgs

        alt_msgs = tf.keras.layers.Lambda(lambda args: get_alt_msgs(*args))
        alt_msg_layer = alt_msgs([value_layer, self.action_input])

        loss_layer = tf.keras.layers.Lambda(
            lambda args: compute_dql_loss(*args) if self.config.loss == 'dql' else (
                        compute_rank_loss(*args) + self.config.mse_weight * compute_mse_loss(*args)))
        loss_layer = loss_layer([alt_msg_layer, q_layer, self.cost_input])
        loss_inputs = [self.node_input, self.edge_input, self.action_input, self.cost_input]
        loss_model = tf.keras.Model(loss_inputs, loss_layer)
        loss_model.compile(loss=lambda _, loss_as_pred: loss_as_pred, optimizer=self.optimizer)

        # testing purpose
        inputs = [self.node_input, self.edge_input, self.action_input, self.cost_input]
        self.alt_msg_layer = self.make_model(inputs, alt_msg_layer, 'alt_msg')
        return loss_model

    def create_multilayers(self, input, dim_hidden, name, n_layers):
        h = tf.keras.layers.Dense(dim_hidden, activation=self.activation,
                                  kernel_initializer=self.config.weight_initializer,
                                  bias_initializer=self.config.weight_initializer,
                                  name=name + "_0")(input)
        for idx in range(n_layers - 1):
            h = tf.keras.layers.Dense(dim_hidden, activation=self.activation,
                                      kernel_initializer=self.config.weight_initializer,
                                      bias_initializer=self.config.weight_initializer,
                                      name=name + "_" + str(idx + 1))(h)
        return h

    def create_multilayer_model(self, input, dim_hidden, name, n_layers):
        h = self.create_multilayers(input, dim_hidden, name, n_layers)
        h_model = self.make_model(input, h, name + '_model')
        return h_model

    @staticmethod
    def create_optimizer(opt_name, lr):
        if opt_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr)
        elif opt_name == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(lr)
        else:
            raise NotImplementedError
        return optimizer

    def create_weight_file_name(self):
        filedir = './learn/q-function-weights/'
        filename = "Q_weight_"
        filename += '_'.join(arg + "_" + str(getattr(self.config, arg)) for arg in [
            'n_msg_passing',
            # 'diff_weight_msg_passing',
            'mse_weight',
            'optimizer',
            # 'batch_size',
            'seed',
            'lr',
            'operator',
            'n_layers',
            'n_hidden',
            'top_k',
            'num_train',
            # 'num_test',
            # 'val_portion',
            'use_region_agnostic',
            'loss',
        ])
        filename += '.hdf5'
        print "Config:", filename
        return filedir + filename

    def predict_with_raw_input_format(self, nodes, edges, actions):
        nodes = np.array(nodes)
        edges = np.array(edges)
        actions = np.array(actions)
        if len(nodes.shape) == 2:
            nodes = nodes[None, :]
        if len(edges.shape) == 3:
            edges = edges[None, :]
        if len(actions.shape) == 0:
            actions = actions[None]
        return self.q_model.predict([nodes[:, :, :], edges, actions])
        # return self.q_model.predict([nodes, edges, actions])

    def make_raw_format(self, state, op_skeleton):
        nodes = np.array([make_one_hot_encoded_node(state.nodes[name]) for name in self.entity_names])
        edges = np.array(
            [[make_one_hot_encoded_edge(state.edges[(a, b)]) for b in self.entity_names] for a in self.entity_names])
        # nodes = np.array([state.nodes[name] for name in self.entity_names])
        # edges = np.array(
        #    [[state.edges[(a, b)] for b in self.entity_names] for a in self.entity_names])
        action = np.array(self.get_idx_based_representation_of_op_skeleton(op_skeleton))
        return nodes[:, 6:], edges, action

    def predict(self, state, op_skeleton):
        """
        idx = 0
        for name in self.entity_name_to_idx:
            self.entity_name_to_idx[name] = idx
            idx += 1
        """
        nodes, edges, action = self.make_raw_format(state, op_skeleton)
        nodes = nodes[None, :]
        edges = edges[None, :]
        val = self.predict_with_raw_input_format(nodes, edges, action)
        return val

    def get_idx_based_representation_of_op_skeleton(self, op_skeleton):
        # todo write
        is_place = op_skeleton.type.find('place') != -1
        discrete_parameters = []
        for d in op_skeleton.discrete_parameters.values():
            is_d_obj = isinstance(d, openravepy.KinBody)
            is_d_region = isinstance(d, AARegion)
            is_d_str = isinstance(d, str) or isinstance(d, unicode)

            if is_d_str:
                is_d_obj = d.find('region') == -1
                is_d_region = not is_d_obj

            if is_d_region and is_place:
                if is_d_str:
                    discrete_parameters.append(d)
                else:
                    discrete_parameters.append(d.name)
            elif is_d_obj and not is_place:
                if is_d_str:
                    discrete_parameters.append(d)
                else:
                    discrete_parameters.append(d.GetName())

        idx_based = [self.entity_name_to_idx[param] for param in discrete_parameters]
        return idx_based
