# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import math
np.random.seed(1234)
# tf.set_random_seed(1234)

class Qlearning:
	def __init__(self,
				n_actions,
				n_features,
				learning_rate=0.8,
				reward_decay=0.9,
				e_greedy=0.9,
				memory_size=20
	):
		# QL parameter
		self.n_actions = n_actions
		self.n_features = n_features
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon_max = e_greedy
		self.memory_size = memory_size
		# total learning step
		self.learn_step_counter = 0

		# self.s_dict = dict()
		# self.a_dict = dict()

		self.t_len_hit_list = []
		self.t_len_miss_list = []
		# consist of [target_net, evaluate_net]
		self.qtable = self.build_q_table(self.n_features, self.n_actions)

		self.cost_his = []

	def build_q_table(self, n_states, actions):
		table = pd.DataFrame(
			np.zeros((n_states, len(actions))),     # q_table 全 0 初始
			columns=actions,                        # columns 对应的是行为名称
	        # index=actions
		)
		return table

	def choose_action(self, state, actions):
		if np.random.uniform() > self.epsilon:
			# random
			action = random.choice(actions)
		else:
			for index in range(len(actions)):
				
		return action

	def learn(self):
		# check to replace target parameters
		if self.learn_step_counter % self.replace_target_iter == 0:
			self.sess.run(self.target_replace_op)
			print('\ntarget_params_replaced\n')
			print(self.epsilon)

		# sample batch memory from all memory
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
		self.learn_step_counter += 1

	def plot_cost(self):
		plt.plot(np.arange(len(self.cost_his)), self.cost_his)
		plt.ylabel('Cost')
		plt.xlabel('training steps')
		plt.show()
