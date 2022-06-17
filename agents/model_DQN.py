import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from collections import defaultdict
from absl import logging
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

class DQN:
	def __init__(self,
				n_actions,
				n_features,
				learning_rate=0.0015,
				discount_factor=0.75,
				e_greedy=0.95,
				replace_target_iter=500,
				memory_size=15000,
				batch_size=1024,
				e_greedy_increment=None,
				output_graph=False,
				upset_memory=False,
				assist_learn=None,
				normalize_reward=True,
				layers=None
	):
		# DQN parameter
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = discount_factor
		self.epsilon_max = e_greedy
		self.replace_target_iter = replace_target_iter
		self.memory_size = memory_size
		self.batch_size = batch_size
		self.epsilon_increment = e_greedy_increment
		self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
		self.upset_memory = upset_memory
		self.assist_learn = assist_learn
		self.normalize_reward = normalize_reward
		self.layers = layers
		# total learning step
		self.learn_step_counter = 0

		# initialize zero memory [s, a, r, s_]
		self.memory = np.zeros((self.memory_size, n_features * 2 + 2))
		# self.temp_sa['request']['victim'] = [[index, time], ...]
		# self.temp_sa = defaultdict(dict)

		# self.s_dict = dict()
		# self.a_dict = dict()

		self.t_len_hit_list = []
		self.t_len_miss_list = []
		# consist of [target_net, evaluate_net]
		self._build_net()

		t_params = tf.get_collection(
			tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
		e_params = tf.get_collection(
			tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

		with tf.variable_scope('hard_replacement'):
			self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

		self.sess = tf.Session()

		if output_graph:
			# $ tensorboard --logdir=logs
			tf.summary.FileWriter("logs/", self.sess.graph)

		self.sess.run(tf.global_variables_initializer())
		self.cost_his = []
		self.reward_his = []
	
	def mem_reset(self):
		self.memory_counter = 0
		del self.memory
		self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

	def create_net(self, layers, net_name):
		if net_name != 'eval_net' and net_name != 'target_net':
			raise "create_net error: net_name error"
		w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
		with tf.variable_scope(net_name):
			pre_input_n = self.n_features
			pre_input = self.s if net_name == 'eval_net' else self.s_
			for i in range(len(layers)):
				with tf.variable_scope('l'+str(i)):
					w = tf.get_variable('w'+str(i), [pre_input_n, layers[i]], initializer=w_initializer)
					b = tf.get_variable('b'+str(i), [1, layers[i]], initializer=b_initializer)
					l = tf.nn.relu(tf.matmul(pre_input, w) + b)
					pre_input_n = layers[i]
					pre_input = l
			with tf.variable_scope('last_layer'):
				Lw = tf.get_variable('Lw', [pre_input_n, self.n_actions], initializer=w_initializer)
				Lb = tf.get_variable('Lb', [1, self.n_actions], initializer=b_initializer)
			if net_name == 'eval_net':
				self.q_eval = tf.matmul(pre_input, Lw) + Lb
			else:
				self.q_next = tf.matmul(pre_input, Lw) + Lb


	def _build_net(self):
		# ------------------ all inputs ------------------------
		self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
		self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
		self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
		self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

		# w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
		# l1_n, l2_n, l3_n, l4_n = 64, 64, 512, 512
		self.create_net(self.layers, "eval_net")
		self.create_net(self.layers, "target_net")

		with tf.variable_scope('q_target'):
			q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')    # shape=(None, )
			self.q_target = tf.stop_gradient(q_target)
		with tf.variable_scope('q_eval'):
			a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
			self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)    # shape=(None, )
		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))
		with tf.variable_scope('train'):
			self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
			# AdamOptimizer RMSPropOptimizer

	def get_key_if_len_gt(self, now_time, len_limit):
		## get "temp_sa"'s key if len greater than "len_limit"
		for key, value in self.temp_sa.items():
			for val in value:
				print(key, val, self.temp_sa[key][val], now_time)
 
	def store_transition(self, s, a, r, s_):
		if not hasattr(self, 'memory_counter'):
			self.memory_counter = 0
		# replace the old memory with new memory
		for _ in range(1):
			index = self.memory_counter % self.memory_size
			if self.upset_memory: s, a, r, s_ = self.get_upset_state(s, a, r, s_)
			transition = np.hstack((s, [a, r], s_))
			self.memory[index, :] = transition
			self.memory_counter += 1

	def get_upset_state(self, s, a, r, s_, feature_num=3):
		while True:
			rand_x, rand_y = random.randint(0, 5), random.randint(0, 5)
			if rand_x != a and rand_y != a: break
		for i in range(feature_num):
			s[rand_x+i], s[rand_y+i] = s[rand_y+i], s[rand_x+i]
			s_[rand_x+i], s_[rand_y+i] = s_[rand_y+i], s_[rand_x+i]
		return s, a, r, s_

	def choose_action(self, observation):
		self.c_method = ""
		if np.random.uniform() < self.epsilon:
			# to have batch dimension when feed into tf placeholder
			# print(observation['feature'])
			observation = observation['feature'][np.newaxis, :]
			# forward feed the observation and get q value for every actions
			actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
			# if self.learn_step_counter % 100 == 0:
			# logging.info("Qvalue =%s", actions_value)
			action = np.argmax(actions_value)
			self.reward_his.append(actions_value[0][action])
			self.c_method = "DQN"
		elif self.assist_learn != None and np.random.uniform() < 0.8:
			ass_me = random.randint(0, len(self.assist_learn)-1)
			action = self.assist_learn[ass_me].choose_action(observation)
			# logging.info("Assist Method =%s", self.assist_learn[ass_me])
			self.c_method = str(self.assist_learn[ass_me])
		else:
			action = np.random.randint(0, self.n_actions)
			self.c_method = "random"
		# logging.info("Current Action Method: %s", self.c_method)
		return action

	def choose_action_value(self, observation):
		# to have batch dimension when feed into tf placeholder
		observation = observation[np.newaxis, :]
		return self.sess.run(self.q_eval, feed_dict={self.s: observation})[0]

	def learn(self):
		# check to replace target parameters
		if self.learn_step_counter % self.replace_target_iter == 0:
			# print('update target', self.learn_step_counter)
			self.sess.run(self.target_replace_op)
			# print('\ntarget_params_replaced\n')
			# print(self.epsilon)

		# sample batch memory from all memory
		# if not hasattr(self, 'memory_counter'):
		# 	self.memory_counter = 1
		if self.memory_counter > self.memory_size:
			sample_index = np.random.choice(self.memory_size, size=self.batch_size)
		else:
			sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
		batch_memory = self.memory[sample_index, :]

		_, cost = self.sess.run(
			[self._train_op, self.loss],
			feed_dict={
				self.s: batch_memory[:, :self.n_features],
				self.a: batch_memory[:, self.n_features],
				self.r: batch_memory[:, self.n_features + 1],
				self.s_: batch_memory[:, -self.n_features:],
			})

		self.cost_his.append(cost)

		# increasing epsilon
		self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
		# logging.info("epsilon:%s", self.epsilon)
		# if self.epsilon < self.epsilon_max: print(self.epsilon)
		self.learn_step_counter += 1

	# def plot_cost(self):
	# 	plt.plot(np.arange(len(self.cost_his)), self.cost_his)
	# 	plt.ylabel('Cost')
	# 	plt.xlabel('training steps')
	# 	plt.show()
