import pickle

import tensorflow as tf


class Model(object):
    def __init__(self, num_entities, num_node_features, num_edge_features, num_operators, config):
        self.sess = tf.Session()
        tf.keras.backend.set_session(self.sess)
        self.config = config
        self.top_k = config.top_k
        self.num_operators = num_operators
        self.weight_file_name = self.create_weight_file_name()
        self.weight_initailizer = self.create_initializer()
        self.optimizer = self.create_optimizer(config.optimizer, config.lr)
        self.weights = self.create_model(num_node_features, num_edge_features, num_operators, config,
                                         self.weight_initailizer)
        self.kmodel = self.create_compiled_keras_model((num_entities, num_node_features),
                                                       (num_entities, num_entities, num_edge_features),
                                                       [1])
        self.sess.run(tf.initialize_all_variables())

    def __deepcopy__(self, _):
        return self

    def get_config(self):
        return {
            'name': 'Model',
            'config': self.config.__dict__,
        }

    def create_model(self, num_node_features, num_edge_features, num_operators, config, weight_initailizer):
        num_latent_features = config.n_hidden
        num_layers = config.n_layers
        with tf.variable_scope("model"):
            # todo what are "graph weights"? Every weight in our GNN is a graph weight. Don't use a generic variable name.
            #   For example, source_weights, dest_weights, and edge_weights are good names.
            graph_weights = [[
                tf.get_variable('graph_weights' + str(pass_idx) + 'x' + str(layer_idx),
                                (num_latent_features, num_latent_features), tf.float32, weight_initailizer)
                for layer_idx in range(num_layers - 1)
            ] for pass_idx in range(config.num_passes) # todo what is num_passes?
            ]
            graph_bias = [[
                tf.get_variable('graph_bias' + str(pass_idx) + 'x' + str(layer_idx), num_latent_features, tf.float32,
                                weight_initailizer)
                for layer_idx in range(num_layers)
            ] for pass_idx in range(config.num_passes)
            ]

            source_weights = self.create_layers(num_node_features, num_latent_features,
                                                weight_initailizer, 'source', config.num_passes)
            dest_weights = self.create_layers(num_node_features, num_latent_features,
                                              weight_initailizer, 'dest', config.num_passes)
            edge_weights = self.create_layers(num_edge_features, num_latent_features,
                                              weight_initailizer, 'edge', config.num_passes, False)

            # todo there should be layers for taking the attentioned-aggregated-msg, and passing it through
            #   multiple layers. Is fc_weights responsible for doing that?
            #   If so, give it an informative name. Any layer can be fully connected; this has a specific
            #   responsibility
            fc_shapes = [num_latent_features * (2 if not config.no_goal_nodes else 1)] + [num_latent_features] * (
                    config.num_fc_layers - 1) + [num_operators]
            fc_weights = [
                tf.get_variable('fc' + str(i), (outshape, inshape), tf.float32, weight_initailizer)
                for i, (inshape, outshape) in enumerate(zip(fc_shapes[:-1], fc_shapes[1:]))
            ]
            fc_bias = [
                tf.get_variable('output_bias' + str(i), outshape, tf.float32, weight_initailizer)
                for i, (inshape, outshape) in enumerate(zip(fc_shapes[:-1], fc_shapes[1:]))
            ]
        weights = {
            'source': source_weights,
            'dest': dest_weights,
            'edge': edge_weights,
            'graph_weights': graph_weights,
            'graph_bias': graph_bias,
            'fc_weights': fc_weights,
            'fc_bias': fc_bias
        }
        return weights

    @staticmethod
    def create_initializer():
        return tf.random_uniform_initializer(-.1, .1)

    @staticmethod
    def create_optimizer(opt_name, lr):
        if opt_name == 'adam':
            optimizer = tf.keras.optimizers.Adam(lr)
        elif opt_name == 'adadelta':
            optimizer = tf.keras.optimizers.Adadelta(lr)
        else:
            raise NotImplementedError
        return optimizer

    def create_compiled_keras_model(self, dim_node, dim_edge, dim_action):
        knodes = tf.keras.Input(shape=list(dim_node))
        kedges = tf.keras.Input(shape=list(dim_edge))
        kactions = tf.keras.Input(shape=list(dim_action), dtype=tf.int32)
        koperators = tf.keras.Input(shape=[1], dtype=tf.int32)
        kcosts = tf.keras.Input(shape=[1], dtype=tf.int32)

        inputs = [knodes, kedges, kactions, koperators, kcosts]
        if self.config.use_mse:
            klayer = tf.keras.layers.Lambda(lambda args: self.rankloss(*args) + .2 * self.mseloss(*args))
            # klayer = tf.keras.layers.Lambda(lambda args: self.mseloss(*args))
        else:
            klayer = tf.keras.layers.Lambda(lambda args: self.rankloss(*args))
        outputs = klayer((knodes, kedges, kactions, koperators, kcosts))

        klayer._trainable_weights = self.params()
        kmodel = tf.keras.Model(inputs, outputs)

        kmodel.compile(loss=lambda _, loss: loss, optimizer=self.optimizer)
        return kmodel

    def mseloss(self, nodes, edges, actions, operators, costs):
        operators = operators[..., 0]
        costs = costs[..., 0]

        pred = self.eval(nodes, edges, actions)
        mask = tf.one_hot(operators, self.num_operators, axis=-1, dtype=tf.float32)
        return tf.losses.mean_squared_error(tf.reduce_sum(pred * mask, axis=-1), -costs)

    def rankloss(self, nodes, edges, actions, operators, costs):
        operators = operators[..., 0]
        costs = costs[..., 0]

        mask = tf.one_hot(operators, self.num_operators, axis=-1)
        pred = tf.reduce_sum(self.eval(nodes, edges, actions) * mask, -1)
        # alt_pred = [
        #    self.eval(nodes, edges, tf.ones((tf.shape(nodes)[0], 2), dtype=tf.int32) * [[i, j]]) * mask
        #    for i in range(8)
        #    for j in range(8, 11)
        # ]
        alt_pred = [
            tf.reduce_sum(self.eval(nodes, edges,
                                    tf.ones((tf.shape(nodes)[0], actions.shape[-1]), dtype=tf.int32) * [[i, ]]) * mask,
                          -1)
            for i in range(11)
        ]

        k = self.top_k
        # todo Here, a wrong variable name is used that would confuse the reader. For example,
        #  top_k_q_val = tf.nn.top_k(tf.transpose(alt_pred), k + 1)[0][..., -1] is not actually top-k Q values.
        #  It is min_of_top_k_q_vals, unless I misunderstood the operations.
        #  In general, readability is preferred over shorter code, at least in a research environment.
        top_k_q_val = tf.nn.top_k(tf.transpose(alt_pred), k + 1)[0][..., -1]
        q_delta = pred - top_k_q_val
        action_ranking_cost = 1 - q_delta
        hinge_loss_on_action_ranking = tf.reduce_mean(tf.maximum(tf.cast(0., tf.float32), action_ranking_cost))

        return hinge_loss_on_action_ranking

    @staticmethod
    def create_variables(dim_input, dim_output, initializer, name):
        return tf.get_variable(name, (dim_output, dim_input), tf.float32, initializer)

    def create_layers(self, dim_input, dim_hidden, initializer, name, n_layers, replace=True):
        layers = [self.create_variables(dim_input, dim_hidden, initializer, name + "0")]

        for i in range(1, n_layers):
            new_var = self.create_variables(dim_hidden if replace else dim_input, dim_hidden, initializer,
                                            name + str(i))
            layers.append(new_var)

        return layers

    def multi_layer_prediction(self, layers, input):
        # todo
        pass

    def eval(self, nodes, edges, actions):
        tnodes = tf.cast(nodes, tf.float32)
        tedges = tf.cast(edges, tf.float32)

        for i in range(self.config.num_passes):
            if i == 0:
                lnodes = tnodes
            else:
                lnodes = tf.transpose(latent, (1, 2, 0))

            # todo what are lnodes and tedges? Use more meaningful names. Don't be afraid to use long variable names.
            sources = tf.tensordot(self.weights['source'][i], lnodes, [[-1], [-1]])
            dests = tf.tensordot(self.weights['dest'][i], lnodes, [[-1], [-1]])
            edges = tf.tensordot(self.weights['edge'][i], tedges, [[-1], [-1]])

            # sources/dests: num_latent x batch x num_node
            # edges: num_latent x batchxnum_node x num_node

            # todo
            #  If you are performing an operation with constants, then they should be named, not left with constants.
            #  For example, it is difficult to interpret this: [-1] + [1] * (len(edges.shape) - 1).
            #  Give it a name. For instance, it can be named bias_shape.
            # todo why the sudden change to leaky_relu?
            messages = tf.nn.leaky_relu(tf.expand_dims(sources, -2) + tf.expand_dims(dests, -1) + edges
                                        + tf.reshape(self.weights['graph_bias'][i][0],
                                                     [-1] + [1] * (len(edges.shape) - 1)))
            for j in range(1, self.config.n_layers):
                messages = tf.nn.leaky_relu(
                    tf.tensordot(self.weights['graph_weights'][i][j - 1], messages, [[-1], [0]]) + tf.reshape(
                        self.weights['graph_bias'][i][j], (-1, 1, 1, 1)))

            # todo latent is too generic of a variable name. this can be called aggregated_message, if I am not mistaken.
            latent = tf.reduce_sum(messages, -1)

        # todo again, if you are performing an operation with constants, then they should be named, not left with constants.
        #   for example, what is aggregated_msg[8:9, :, :] supposed to represent?
        #                what is tf.reduce_sum(tf.one_hot(actions, aggregated_msg.shape[-1], dtype=tf.float32), -2)?
        attention = tf.concat(
            [tf.reduce_sum(tf.reduce_sum(tf.one_hot(actions, latent.shape[-1], dtype=tf.float32), -2) * latent, -1)] + (
                [tf.reduce_sum(latent * latent[8:9, :, :], -1)] if not self.config.no_goal_nodes else []), 0)

        # todo this code has too much coupling; it can be significantly refactored if you
        #   write a separate function for multi-layer prediction
        output = tf.transpose(attention)
        for i in range(self.config.num_fc_layers):
            output = tf.nn.leaky_relu(tf.tensordot(output, tf.transpose(self.weights['fc_weights'][i]), [[-1], [0]]) +
                                      self.weights['fc_bias'][i])

        return output

    def params(self):
        params = [l for k in self.weights.values() for v in (k if isinstance(k, list) else [k]) for l in
                  (v if isinstance(v, list) else [v])]
        return params

    def create_weight_file_name(self):
        filedir = './learn/'
        filename = "Q_weight_"
        print "Config:"
        for arg in vars(self.config):
            print arg, getattr(self.config, arg)
        #filename += '_'.join(arg + "_" + str(getattr(self.config, arg)) for arg in vars(self.config))

        filename += '_'.join(arg + "_" + str(getattr(self.config, arg)) for arg in [
            'optimizer',
            'batch_size',
            'seed',
            'num_train',
            'lr',
            'val_portion',
            'num_test',
            'operator',
            'num_passes',
            'n_layers',
            'num_fc_layers',
            'n_hidden',
            'no_goal_nodes',
        ])

        filename += '.hdf5'
        return filedir + filename

    def get_idx_based_representation_of_op_skeleton(self, op):
        # todo write
        discrete_parameters = []
        for d in op.discrete_parameters.values():
            if type(d) == str or type(d) == unicode:
                discrete_parameters.append(str(d))
            elif hasattr(d, 'name'):
                discrete_parameters.append(d.name)
            else:
                discrete_parameters.append(str(d.GetName()))

        idx_based = [self.entity_idx[param] for param in discrete_parameters]
        return idx_based


