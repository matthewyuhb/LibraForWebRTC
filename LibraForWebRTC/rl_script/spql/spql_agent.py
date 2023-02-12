import numpy as np
from .node import Node
from sklearn.cluster import KMeans
import random
import multiprocessing
from multiprocessing import Process
from multiprocessing import Pool


class Agent():

    def __init__(self, state_dim, action_dim, state_bounds, action_bounds, gamma, epsilon=0.4,
                 max_sample_num_for_node=800, max_node_num=160, rounds_per_backward=3,
                 increase_factor=1.2, alpha=0.8, threshold_num=160, degree=2, fit_ratio_threshold=0.1,
                 ridge_alpha=1, is_sample_weighted=False, discounted_factor=0.98, ep_decrease_factor=0.000002,
                 split_ratio_limit=5, discarding=False, discard_thr_ratio=0.01, lowest_ep=0.08, is_logging=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state_bounds = state_bounds  # 这里对bound应当做出调整进行稍微的拓张
        self.action_bounds = action_bounds
        self.gamma = gamma
        self.epsilon = epsilon
        self.node_list = []
        self.threshold_num = threshold_num
        self.increase_factor = increase_factor
        self.rounds_per_backward = rounds_per_backward
        self.alpha = alpha
        self.max_node_num = max_node_num
        self.degree = degree #控制拟合函数是二次函数还是3次
        self.fit_ratio_threshold = fit_ratio_threshold
        self.ep_decrease_factor = ep_decrease_factor
        self.lowest_ep = lowest_ep
        self.is_logging = is_logging
        self.last_node_bias = None
        self.node_num = 2


        # discard setting
        self.discarding = discarding
        self.last_state = np.array([])
        self.discard_abs = []
        for bound in self.state_bounds:
            self.discard_abs.append(abs(bound[1] - bound[0]) * discard_thr_ratio)

        self.is_sample_weighted = is_sample_weighted
        self.discounted_factor = discounted_factor

        self.max_sample_num_for_node = max_sample_num_for_node
        self.ridge_alpha = ridge_alpha
        first_node = Node(self,state_dim, action_dim, state_bounds, action_bounds, node_index=0, degree=degree,
                          max_sample_num=self.max_sample_num_for_node, can_split_threshold=self.threshold_num,
                          ridge_alpha=ridge_alpha, is_sample_weighted=is_sample_weighted,
                          discounted_factor=self.discounted_factor)
        first_node.learning_rate = self.alpha
        self.done_node = Node(self,state_dim, action_dim, state_bounds, action_bounds, node_index=-1, degree=degree,
                              max_sample_num=self.max_sample_num_for_node, can_split_threshold=self.threshold_num,
                              ridge_alpha=ridge_alpha, is_sample_weighted=is_sample_weighted,
                              discounted_factor=self.discounted_factor)
        self.node_list.append(first_node)
        self.node_dic = {}
        self.node_dic[0] = first_node
        self.last_node_index = 0
        self.restart_num = 0
        self.split_ratio_limit = split_ratio_limit
        # increase state bounds sightly
        for i in range(self.state_dim):
            self.state_bounds[i][1] += 0.01

    def __discarding(self, state):
        if self.discarding:
            if self.last_state.shape[0] == 0:
                self.last_state = state
                return False
            else:
                for i in range(len(state)):
                    if abs(self.last_state[i] - state[i]) > self.discard_abs[i]:
                        self.last_state = state
                        return False
                return True
        self.last_state = state
        return False

    def get_record(self, state, action, next_state, reward, episode_index, sample_weight=1, done=False):  # 这里计算q值的方式仍有问题，拟合之后应当重新算所有的q值

        if self.__discarding(state):
            return

        if done:
            next_state = [float('inf') for i in range(len(state))]
            next_node = self.done_node
            node = self.find_node(state, 0)
            q_value = reward
        else:
            node = self.find_node(state, 0)
            next_node = self.find_node(next_state, 0)  # 在这个地方就可以记录哪些node被反向查找，从而建立反向的链路
            _, next_q_value = next_node.get_max_q_value(next_state)
            q_value = reward + self.gamma * next_q_value
        sample = np.concatenate([state, action, next_state, np.array([reward]), np.array([q_value]),
                                 np.array([next_node.node_index]), np.array([episode_index]), np.array([sample_weight])],
                                axis=0)  # 也即sample内的储存顺序为：state，action，next_state, reward, q_value, next_state_node_index, episode_index
        node.insert_sample(sample)
        self.epsilon = max(self.lowest_ep, self.epsilon - self.ep_decrease_factor)
        if self.is_logging:
            print('epsilon: {}'.format(self.epsilon))
            print("node num:{}".format(len(self.node_list)))
            print("current node's sample num:{}".format(node.history_data.shape[0]))
            print("node bias: {}".format(node.bias))
        # assert not isinstance(node.bias, int) and node.bias[0] > 500

        if node.is_fit:
            if node.is_need_split():
                self.split_node(node, episode_index)
            elif node.update_ratio() > self.fit_ratio_threshold: #self.fit_ratio_threshold = 0.1
                node.fit_q_function(current_episode=episode_index)
                self.update_samples_in_back_node(node)
        elif node.history_data.shape[0] > self.state_dim * 5:
            node.fit_q_function(current_episode=episode_index)

    def get_action(self, state):
        r = random.uniform(0, 1)
        if r < self.epsilon:
            action = np.empty([self.action_dim], dtype=float)
            for i in range(self.action_dim):
                action[i] = random.uniform(self.action_bounds[i][0], self.action_bounds[i][1])
            return action
        else:
            node = self.find_node(state, 0)
            self.last_node_bias = node.bias
            if not node.is_state_in(state):
                # print(state)
                raise ValueError('state not in node range')
            return node.get_action(state)

    def get_test_action(self, state):
        node = self.find_node(state, 0)
        if not node.is_state_in(state):
            # print(state)
            raise ValueError('state not in node range')
        return node.get_action(state)

    def find_node(self, state, index):
        node = self.node_dic[index]
        if node.high_tree_node == None:
            return node

        temp_node = node
        while temp_node.high_tree_node != None:
            if temp_node.high_tree_node.is_state_in(state):
                temp_node = temp_node.high_tree_node
            else:
                temp_node = temp_node.low_tree_node
        return temp_node

    def _insert_node(self, node):
        self.node_list.append(node)

    def split_node(self, node, current_episode_index):
        bias = 0
        min_bias = float('inf')
        dim = 0
        max_dim_length_ratio = -1
        mid = 0
        for i in range(self.state_dim):
            current_dim_len_ratio = (node.state_bounds[i][1] - node.state_bounds[i][0]) / (
                        self.state_bounds[i][1] - self.state_bounds[i][0])
            if current_dim_len_ratio > max_dim_length_ratio:
                dim = i
                mid = np.mean(node.history_data[:, i])
                max_dim_length_ratio = current_dim_len_ratio

        for i in range(self.state_dim):
            # v = np.var(node.history_data[:,i]/(self.state_bounds[i][1] - self.state_bounds[i][0]))
            # if v > vari:
            #     dim = i
            #     vari = v
            temp_mid = np.mean(node.history_data[:, i])
            current_dim_len_ratio_1 = (node.state_bounds[i][1] - temp_mid) / (
                        self.state_bounds[i][1] - self.state_bounds[i][0])
            current_dim_len_ratio_2 = (temp_mid - node.state_bounds[i][0]) / (
                    self.state_bounds[i][1] - self.state_bounds[i][0])
            if current_dim_len_ratio_1 < max_dim_length_ratio / (self.split_ratio_limit) or \
                    current_dim_len_ratio_2 < max_dim_length_ratio / (self.split_ratio_limit) or \
                    np.var(node.history_data[:, i]) == 0:
                continue

            samples_low = np.array([])
            samples_high = np.array([])
            for j in range(node.history_data.shape[0]):
                sample = node.history_data[j]
                if sample[i] < temp_mid:
                    count = samples_low.shape[0]
                    if count == 0:
                        samples_low = np.array([sample])
                    else:
                        samples_low = np.insert(samples_low, count, sample, axis=0)
                else:
                    count = samples_high.shape[0]
                    if count == 0:
                        samples_high = np.array([sample])
                    else:
                        samples_high = np.insert(samples_high, count, sample, axis=0)

            temp_bias = node.temp_fit_q_function(samples_high, current_episode_index) + node.temp_fit_q_function(
                samples_low, current_episode_index)
            if temp_bias < min_bias:
                min_bias = temp_bias
                dim = i
                mid = temp_mid

        low_bound = node.state_bounds.copy()
        low_bound[dim][1] = mid
        high_bound = node.state_bounds.copy()
        high_bound[dim][0] = mid
        self.last_node_index += 1
        node_low = Node(self, self.state_dim, self.action_dim, low_bound, self.action_bounds,
                        node_index=self.last_node_index, degree=self.degree,
                        max_sample_num=self.max_sample_num_for_node, can_split_threshold=self.threshold_num,
                        ridge_alpha=self.ridge_alpha, is_sample_weighted=self.is_sample_weighted,
                        discounted_factor=self.discounted_factor)
        node_low.can_split_threshold = (int)(node.can_split_threshold * self.increase_factor)
        self.last_node_index += 1
        node_high = Node(self, self.state_dim, self.action_dim, high_bound, self.action_bounds,
                         node_index=self.last_node_index, degree=self.degree,
                         max_sample_num=self.max_sample_num_for_node, can_split_threshold=self.threshold_num,
                         ridge_alpha=self.ridge_alpha, is_sample_weighted=self.is_sample_weighted,
                         discounted_factor=self.discounted_factor)
        # self.node_num += 1
        
        node_high.can_split_threshold = (int)(node.can_split_threshold * self.increase_factor)

        node_low.level = node.level + 1
        node_high.level = node.level + 1
        node_low.learning_rate = self.alpha * pow(1, node_low.level)
        node_high.learning_rate = self.alpha * pow(1, node_high.level)

        for i in range(node.history_data.shape[0]):
            sample = node.history_data[i]
            if sample[dim] < mid:
                node_low.insert_sample(sample)
            else:
                node_high.insert_sample(sample)

        node_low.fit_q_function(current_episode_index)
        node_high.fit_q_function(current_episode_index)

        # self.backward_calculate(node)

        node.low_tree_node = node_low
        node.high_tree_node = node_high
        self.node_dic[node_low.node_index] = node_low
        self.node_dic[node_high.node_index] = node_high

        self.node_list.remove(node)
        self._insert_node(node_low)
        self._insert_node(node_high)
        print("--------Remove node:{},current we have {} nodes.-----".format(node.node_index,len(self.node_list)))

        self.replace_next_node_index(node, low_node=node_low, high_node=node_high)
        self.update_samples_in_back_node(node_low)
        self.update_samples_in_back_node(node_high)

        # self.backward_calculate(node_low, self.rounds_per_backward)
        # self.backward_calculate(node_high, self.rounds_per_backward)

    def replace_next_node_index(self, old_node, low_node, high_node):
        # begin and end refer to the positions of the next_state in samples
        begin = self.state_dim + self.action_dim
        end = begin + self.state_dim
        for node in self.node_list:
            if old_node.node_index not in node.next_nodes.keys():
                continue
            for i in range(node.history_data.shape[0]):
                if node.history_data[i][node.next_index_pos] == old_node.node_index:
                    next_state = node.history_data[i][begin:end]
                    if low_node.is_state_in(next_state):
                        node.update_sample(i, None, low_node.node_index)
                    else:
                        node.update_sample(i, None, high_node.node_index)

    def get_q_value(self, state):
        node = self.find_node(state, 0)
        _, q_value = node.get_max_q_value(state)
        return q_value

    # this function is designd to boost the training process, but useless
    def backward_calculate(self, node,
                           round=6):  # 这个地方很关键，需要解决长程依赖问题，因此需要反向计算，但是这样可能会引入巨量的误差，反向计算时可以考虑不用拟合的函数而是直接寻找数据，或者直接在计算q值时采用反向计算的方法

        def find_max_change_node(update_nodes):
            max_rate = 0
            target_node = None
            for temp_node in update_nodes:
                rate = temp_node.update_ratio()
                if rate > max_rate:
                    max_rate = rate
                    target_node = temp_node
            if max_rate < self.fit_ratio_threshold: # self.fit_ratio_threshold = 0.1
                target_node = None
            return target_node

        # begin and end refer to the positions of the next_state in samples
        begin = self.state_dim + self.action_dim
        end = begin + self.state_dim

        for _ in range(round):
            update_nodes = self.node_list.copy()
            if _ == 0:
                next_node = node
            else:
                next_node = find_max_change_node(update_nodes)
            while next_node != None and len(update_nodes) != 0:
                update_nodes.remove(next_node)
                next_node.fit_q_function()
                self.update_samples_in_back_node(next_node)
                next_node = find_max_change_node(update_nodes)

    def update_samples_in_back_node(self, next_node):
        # begin and end refer to the positions of the next_state in samples
        begin = self.state_dim + self.action_dim
        end = begin + self.state_dim

        def update_one_back_node(next_node, need_update_node):
            for i in range(need_update_node.history_data.shape[0]):
                if need_update_node.history_data[i][need_update_node.next_index_pos] == next_node.node_index:
                    next_state = need_update_node.history_data[i][begin: end]
                    _, next_q_value = next_node.get_max_q_value(next_state)
                    new_q_value = need_update_node.history_data[i][need_update_node.q_value_pos] * (
                                1 - need_update_node.learning_rate) \
                                  + need_update_node.learning_rate * (
                                              need_update_node.history_data[i][need_update_node.reward_value_pos]
                                              + self.gamma * next_q_value)
                    need_update_node.update_sample(i, new_q_value, next_node.node_index)

        for need_update_node in self.node_list:
            if next_node.node_index in need_update_node.next_nodes.keys():
                update_one_back_node(next_node, need_update_node)
