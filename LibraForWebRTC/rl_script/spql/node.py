import abc
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from scipy import optimize
import random
from sklearn.metrics import mean_squared_error


class Node():
    def __init__(self, agent, state_dim, action_dim, state_bounds, action_bounds, node_index, degree = 2,
                 can_split_threshold=100, max_sample_num = 800, batch=50, ridge_alpha=1, importance_ratio=0.001
                 , is_sample_weighted=False, discounted_factor = 0.98):
        # self.node_id = node_id
        self.agent = agent
        self.state_bounds = state_bounds
        self.action_bounds = action_bounds
        self.history_data = np.array([])
        self.updated_sample_tag = []
        self.q_value = {}
        self.max_sample_num = max_sample_num
        self.bias = 0
        self.last_bias = self.bias
        self.last_sample_num = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_value_begin = state_dim * 2 + action_dim + 1
        self.parameters = []
        self.poly_reg = PolynomialFeatures(degree = degree)
        self.linear_model = linear_model.Ridge(alpha=ridge_alpha)
        self.temp_linear_model = linear_model.Ridge(alpha=ridge_alpha)

        self.is_fit = False
        self.random_action = np.array([(bound[0] + bound[1]) / 2 for bound in self.action_bounds])
        self.batch = batch
        self.count = self.batch
        self.close_sample_num = 0
        self.can_split_threshold = can_split_threshold
        self.check_count = 0
        self.enter_count = 0 #record the times of the states which the agent came across
        self.level = 0
        self.learning_rate = 0.8
        self.score = 0
        self.updated_sample_num = 0
        self.last_fit_num = 0
        self.q_value_pos = -4
        self.next_index_pos = -3
        self.episode_index = -2
        self.reward_value_pos = self.q_value_pos - 1
        self.sample_weight_pos = -1

        #for search
        self.low_tree_node = None
        self.high_tree_node = None
        self.split_dim = -1
        self.split_threshold = None
        self.node_index = node_index

        #for backforward calculate
        self.next_nodes = {}

        #for sample weight
        self.is_sample_weighted = is_sample_weighted
        self.importance_ratio = importance_ratio
        self.discounted_factor = discounted_factor

    def fit_q_function(self, current_episode):
        # print("num of samples before fit: {}".format(self.history_data.shape[0]))
        q_value_pos = -3
        if self.history_data.shape[0] == 0:
            return
        X_data = self.history_data[:,0:self.state_dim + self.action_dim]
        Y_data = self.history_data[:, self.q_value_pos]
        X_data = self.poly_reg.fit_transform(X_data)
        is_sample_weight = True
        if self.is_sample_weighted:
            weights = [sample[self.sample_weight_pos] * pow(self.discounted_factor, current_episode -
                                                            sample[self.episode_index]) for sample in self.history_data]
            a = self.linear_model.fit(X_data, Y_data, sample_weight=weights)
            self.bias = mean_squared_error(Y_data, self.linear_model.predict(X_data), sample_weight=weights) / self.history_data.shape[0]
        else:
            a = self.linear_model.fit(X_data, Y_data)
            self.bias = mean_squared_error(Y_data, self.linear_model.predict(X_data)) / self.history_data.shape[0]

        self.last_bias = self.bias

        self.last_fit_num = int(self.history_data.shape[0])
        self.is_fit = True
        self.count = self.batch
        self.updated_sample_num = 0
        self.updated_sample_tag = [False for i in self.updated_sample_tag]

    def temp_fit_q_function(self, samples, current_episode):
        X_data = samples[:, 0:self.state_dim + self.action_dim]
        Y_data = samples[:, self.q_value_pos]
        X_data = self.poly_reg.fit_transform(X_data)
        if self.is_sample_weighted:
            # weights = np.linspace(max(0.001, 1 - self.importance_ratio * samples.shape[0] / 2),
            #                       1 + self.importance_ratio * samples.shape[0] / 2,
            #                       samples.shape[0])
            weights = [sample[self.sample_weight_pos] * pow(self.discounted_factor, current_episode -
                                                             sample[self.episode_index]) for sample in samples]
            a = self.temp_linear_model.fit(X_data, Y_data, sample_weight=weights)
            bias = mean_squared_error(Y_data, self.temp_linear_model.predict(X_data), sample_weight=weights)
        else:
            a = self.temp_linear_model.fit(X_data, Y_data)
            bias = mean_squared_error(Y_data, self.temp_linear_model.predict(X_data))
        # self.score = self.linear_model.score(X_data, Y_data)

        return bias

    def get_max_q_value(self, state):
        def calculate_q_value(state):
            return lambda action : self.linear_model.predict(self.poly_reg.fit_transform(np.array([np.concatenate((state, action), axis=0)])))[0]
        if not self.is_fit:
            return None, 0
        # print(state)
        a = optimize.minimize(calculate_q_value(state), x0=self.random_action, bounds=self.action_bounds, method='SLSQP')
        return a.x, float(a.fun)

    def get_action(self, state):
        if self.is_fit:
            action, _ = self.get_max_q_value(state)
            return action
        else:
            action = np.empty([self.action_dim], dtype=float)
            for i in range(self.action_dim):
                print(self.action_bounds)
                action[i] = random.uniform(self.action_bounds[i][0], self.action_bounds[i][1])
            return action

    def is_state_in(self, state):
        for i in range(self.state_dim):
            if state[i] >= self.state_bounds[i][1] or state[i] < self.state_bounds[i][0]:
                return False
        return True


    def insert_sample(self, sample):
        def compare(sample1, sample2):
            for i in range(self.state_dim):
                if sample1[i] > sample2[i]:
                    return True
                elif sample1[i] < sample2[i]:
                    return False
            return True



        if int(sample[self.next_index_pos]) not in self.next_nodes.keys():
            self.next_nodes[int(sample[self.next_index_pos])] = 1
        else:
            self.next_nodes[int(sample[self.next_index_pos])] += 1

        next_state = sample[self.state_dim + self.action_dim: self.state_dim * 2 + self.action_dim]
        if self.is_state_in(next_state):
            self.close_sample_num += 1

        self.count = self.count - 1
        # sample = np.concatenate([state, action, next_state, reward, q_value], axis=0)
        j = self.history_data.shape[0]
        self.updated_sample_tag.append(True)
        self.updated_sample_num += 1
        if j == 0:
            self.history_data = np.array([sample])

            return


        self.history_data = np.insert(self.history_data, j, sample, axis=0)
        print("node id:{}, level:{}, history_data:{}, state_bounds:{}".format(self.node_index, self.level, self.history_data.shape[0],self.state_bounds))


        if self.is_fit:
            X_data = np.array([sample[0:self.state_dim + self.action_dim]])
            predict_q = self.linear_model.predict(self.poly_reg.fit_transform(X_data))

            self.bias = (self.bias * (self.history_data.shape[0] - 1) + abs(sample[self.q_value_pos] - predict_q)) / \
                        self.history_data.shape[0]

        if self.history_data.shape[0] > self.max_sample_num:
            delete_sample = self.history_data[0]
            next_state = delete_sample[self.state_dim + self.action_dim: self.state_dim * 2 + self.action_dim]
            if self.is_state_in(next_state):
                self.close_sample_num -= 1
            del_X_data = np.array([delete_sample[0:self.state_dim+self.action_dim]])
            del_q_value = self.linear_model.predict(self.poly_reg.fit_transform(del_X_data))
            self.bias = (self.bias * (self.history_data.shape[0]) - abs(delete_sample[self.q_value_pos] - del_q_value)) / \
                        (self.history_data.shape[0] - 1)
            self.history_data = np.delete(self.history_data, 0, axis=0)
            if self.updated_sample_tag:
                self.updated_sample_num -= 1
            self.updated_sample_tag.pop(0)
            self.next_nodes[int(delete_sample[self.next_index_pos])] -= 1



    def update_sample(self, index, q_value, next_node_index):

        if self.history_data[index][self.next_index_pos] != next_node_index:
            if next_node_index not in self.next_nodes.keys():
                self.next_nodes[next_node_index] = 1
            else:
                self.next_nodes[next_node_index] += 1
        self.history_data[index][self.next_index_pos] = next_node_index
        if q_value != None:
            if next_node_index != self.node_index and not self.updated_sample_tag[index]:
                self.updated_sample_num += 1
                self.updated_sample_tag[index] = True
            self.history_data[index][self.q_value_pos] = q_value


    def update_ratio(self):
        print("node id:{}, level:{},update_ratio:updated_sample_num{},last_fit_num:{},state_bounds:{}".format(self.node_index, self.level,self.updated_sample_num,self.last_fit_num,self.state_bounds))
        return self.updated_sample_num / self.last_fit_num

    def is_need_split(self):
        if self.enter_count > self.can_split_threshold:
            return True
        self.enter_count += 1
        return False

