import tensorflow as tf

class Baseline(object):
	def __init__(self, num_node_features, num_edge_features, num_nodes):
		with tf.variable_scope("model"):
			num_latent = 1
			self.node_weights = []
			self.edge_weights = []
			self.action_weights = []
			for i in range(num_latent):
				self.node_weights.append(tf.get_variable('node_weights{}'.format(i), (num_nodes, num_node_features), tf.float64, tf.random_uniform_initializer(-.1,.1)))
				self.edge_weights.append(tf.get_variable('edge_weights{}'.format(i), (num_nodes, num_nodes, num_edge_features), tf.float64, tf.random_uniform_initializer(-.1,.1)))
				self.action_weights.append(tf.get_variable('action_weights{}'.format(i), num_node_features, tf.float64, tf.random_uniform_initializer(-.1,.1)))
			self.output_weights = tf.get_variable('output_weights', num_latent, tf.float64, tf.random_uniform_initializer(-.1,.1))

	def eval(self, nodes, edges, actions):
		tnodes = tf.cast(nodes, tf.float64)
		tedges = tf.cast(edges, tf.float64)

		latent = tf.stack([
			tf.reduce_sum(node_weights * tnodes, [-2,-1]) + tf.reduce_sum(edge_weights * tedges, [-3,-2,-1]) + tf.reduce_sum(tf.tensordot(tf.reduce_sum(tf.one_hot(actions, tnodes.shape[-2], dtype=tf.float64), -2), action_weights, [[],[]]) * tnodes, [-2,-1])
			for node_weights, edge_weights, action_weights in zip(self.node_weights, self.edge_weights, self.action_weights)
		])

		return latent

		latent = tf.nn.relu(latent)

		output = tf.reduce_sum(self.output_weights * tf.transpose(latent), -1)

		return output

	def params(self):
		return self.node_weights + self.edge_weights + self.action_weights + [self.output_weights]

