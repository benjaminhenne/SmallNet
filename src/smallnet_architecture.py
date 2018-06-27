import tensorflow as tf

class Smallnet(object):
	def __init__(self, settings):
		self.settings = settings

		# inputs
		with tf.name_scope('inputs'):
			self.inputs = tf.placeholder(tf.float32, [None, 32,32,3], name='inputs')
			self.labels = tf.placeholder(tf.int64, [None], name='labels')
			self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')
			self.global_step = tf.Variable(0, trainable=False, name='global_step')

		if self.settings.network_layout == 'default':
			self.logits, layer_names = self.build_network('default', 'smallnet')
		elif self.settings.network_layout == 'default_fixed':
			self.logits, layer_names = self.build_network('default_fixed', 'smallnet')
		else:
			raise Exception('Specified network type not yet implemented!')

		# objective
		self.penalty = tf.constant(0)
		with tf.name_scope('objective'):
			# cross entroy
			self.xentropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))

			if self.settings.l1_regularize and self.settings.l2_regularize:
				raise Exception('[FATAL ERROR] Cannot L1 and L2 regularise at the same time!')

			elif self.settings.l1_regularize:
				# collect all activation weights
				weight_sets = [[var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if l in var.name and '/activation_weights' in var.name] for l in layer_names]
				# init l1 regulariser and apply regularisation to collected weights
				l1_reg = tf.contrib.layers.l1_regularizer(scale=self.settings.l1_scale, scope="l1_regularisation")
				penalties = [tf.contrib.layers.apply_regularization(l1_reg, weight) for weight in weight_sets]
				# add penalties to loss and apply summaries
				self.penalty = tf.add_n(penalties)
				self.loss = self.xentropy + self.penalty
				self.add_summaries(self.penalty, 'penalty')
			elif self.settings.l2_regularize:
				self.l2 = [tf.nn.l2_loss(v) for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'weights' and not 'activation_weight' in v.name]
				self.weights_norm = tf.reduce_sum(input_tensor=self.settings.l2_lambda*tf.stack(self.l2), name='weights_norm')
				self.loss = self.xentropy + self.weights_norm
			else:
				self.loss = self.xentropy

			# accuracy
			self.accuracy = tf.equal(tf.argmax(tf.nn.softmax(self.logits), 1), self.labels)
			self.accuracy = tf.reduce_mean(tf.cast(self.accuracy, tf.float32))

			self.add_summaries(self.accuracy, 'accuracy')
			self.add_summaries(self.accuracy, 'validation_accuracy')
			self.add_summaries(self.loss, 'loss')
			self.add_summaries(self.xentropy, 'cross_entropy')

		with tf.name_scope('optimisation'):
			if self.settings.optimiser == 'Adam':
				self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
			else:
				self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

			self.minimize = self.optimizer.minimize(self.loss)
			varlist = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
			self.gradients = self.optimizer.compute_gradients(self.loss, var_list=varlist)
			self.update = self.optimizer.apply_gradients(grads_and_vars=self.gradients, global_step=self.global_step)

		sums = 		[s for s in tf.get_collection(tf.GraphKeys.SUMMARIES) if 'validation' not in s.name]
		val_sums =	[s for s in tf.get_collection(tf.GraphKeys.SUMMARIES) if 'validation' in s.name]
		self.summaries = 			tf.summary.merge(sums)
		self.validation_summaries = tf.summary.merge(val_sums)


	# adds a range of summaries to a node
	def add_summaries(self, arg, label):
		with tf.name_scope(label):
			mean = tf.reduce_mean(arg)
			stddev = tf.sqrt(tf.reduce_mean(tf.square(arg - mean)))
			tf.summary.scalar('mean', mean)
			tf.summary.histogram('histogram', arg)
			if self.settings.verbose_summaries:
				tf.summary.scalar('stddev', stddev)
				tf.summary.scalar('max', tf.reduce_max(arg))
				tf.summary.scalar('min', tf.reduce_min(arg))

	def build_network(self, net_type, name_scope):
		if net_type == 'default':
			return self.build_default_network(name_scope)
		elif net_type == 'default_fixed':
			#build_default_fixed_network(name_scope)
			raise Exception('Specified network type not yet implemented!')
		else:
			raise Exception('Specified network type not yet implemented!')

	def build_default_network(self, name_scope='None'):
		# conv1
		state	= self.conv2d_multiple_act(layer_input=self.inputs, weights_shape=[5,5,3,64], scope_name='conv01')
		state 	= tf.nn.dropout(state, keep_prob=self.settings.dropout_rate, name='dropout01')
		state 	= self.conv2d_multiple_act(layer_input=state, weights_shape=[3,3,64,64], scope_name='conv02')
		# pooling 1
		state	= tf.nn.max_pool(state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='max_pool01')
		# conv2
		state	= self.conv2d_multiple_act(layer_input=state, weights_shape=[1,1,64,64], scope_name='conv03')
		state	= tf.nn.dropout(state, keep_prob=self.settings.dropout_rate, name='dropout02')
		state	= self.conv2d_multiple_act(layer_input=state, weights_shape=[5,5,64,64], scope_name='conv04')
		# pooling 2
		state	= tf.nn.max_pool(state, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME', name='max_pool02')
		# dense layer 1
		state	= self.dense_multiple_act(state, weights_shape=[384], bias_shape=[-1], bias_init=0.1, scope_name='dense01')
		state	= tf.nn.dropout(state, keep_prob=self.settings.dropout_rate, name='dropout03')
		state 	= self.dense_multiple_act(state, weights_shape=[192], bias_shape=[-1], bias_init=0.1, scope_name='dense02')
		# output layer
		logits = self.dense_multiple_act(state, weights_shape=[10], bias_shape=[-1], bias_init=0.1, logits=True, scope_name='output01')
		return logits, ['conv01', 'conv02', 'conv03', 'conv04', 'dense01', 'dense02']


	# standard conv2 with addition of multiple activations (return statement)
	def conv2d_multiple_act(self, layer_input, weights_shape, bias_shape=[-1], stride=[1,1,1,1], padding='SAME', bias_init=0.1, scope_name=None):
		with tf.variable_scope(scope_name):
			# calc bias sizes on the fly
			if bias_shape == [-1]:
				bias_shape = [weights_shape[-1]]

			weight_init = tf.truncated_normal_initializer(stddev=tf.sqrt(2./(weights_shape[0] * weights_shape[1] * weights_shape[2])))
			weights	= tf.get_variable('layer_weights', weights_shape, initializer=weight_init)
			biases = tf.get_variable('layer_biases', bias_shape, initializer=tf.constant_initializer(bias_init))
			layer = tf.nn.conv2d(layer_input, weights, stride, padding)
			self.add_summaries(weights, 'layer_weights')
			self.add_summaries(biases, 'layer_biases')
			if bias_shape != [0]: # manually add biases now before we mess with activations
				layer += biases
			return self.add_activations(layer)

	# standard dense layer setup with addition of multiple activations (return statement) and possibly logits for the output
	def dense_multiple_act(self, layer_input, weights_shape, bias_shape, bias_init, logits=False, scope_name=None):
		with tf.variable_scope(scope_name):
			# calc bias sizes on the fly
			flat_layer_input = tf.layers.flatten(layer_input)
			dims = flat_layer_input.get_shape().as_list()[1]

			if bias_shape == [-1]:
				bias_shape = [weights_shape[-1]]

			weights_shape = [dims, weights_shape[0]]
			weights = tf.get_variable('layer_weights', weights_shape, initializer=tf.truncated_normal_initializer(stddev=tf.sqrt(2./(weights_shape[0] * weights_shape[1]))))
			biases = tf.get_variable('layer_biases', bias_shape, initializer=tf.constant_initializer(bias_init))
			self.add_summaries(weights, 'layer_weights')
			self.add_summaries(biases, 'layer_biases')
			layer = tf.matmul(flat_layer_input, weights)
			if bias_shape != [0]: # manually add biases now before we mess with activations
				layer += biases
			return layer if logits else self.add_activations(layer)

	# adds all available activation functions to a layer after custom conv2d/dense_layer call
	def add_activations(self, layer):
		out_list = []
		# go through list of activations
		for i in range(len(self.settings.activations)):
			with tf.variable_scope(self.settings.activations[i].__name__):
				# get layer dimensions as list so they can be fed into the init_w lambda functions
				dims = layer.get_shape().as_list()
				# TODO correct shapes for weights; do we even need biases?; optimal init values?
				weights = tf.get_variable('activation_weights', shape=[], initializer=tf.truncated_normal_initializer(
					stddev=self.settings.act_inits[i](dims)))
				self.add_summaries(weights, 'activation_weights')
				# swish and identity don't want to behave like API activation functions, so they get special treatment
				if self.settings.activations[i].__name__ == 'swish':
					beta = tf.get_variable('act_swish_beta', initializer=self.settings.act_inits[i](layer))
					self.add_summaries(beta, 'act_swish_beta')
					act = self.settings.activations[i]()
					out_list.append(act(layer, beta) * weights)
				elif self.settings.activations[i].__name__ == 'identity':
					act = self.settings.activations[i]()
					out_list.append(act(layer) * weights)
				else:
					out_list.append(self.settings.activations[i](layer) * weights)
		# adds all pre-built activation functions to this layer in parallel
		output = tf.add_n(out_list)
		return output
