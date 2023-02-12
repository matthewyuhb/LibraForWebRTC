#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from struct import pack
import sys
import os
from typing import Tuple
from gym import ActionWrapper
import numpy as np

import pandas as pd  # 保存数据用
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gym"))

import alphartc_gym
# from alphartc_gym.utils.trendline import TrendlineEstimator
# from alphartc_gym.utils.packet_record import RateControler

# from oc_svm import OC_SVM
UNIT_M = 1000000
MAX_BANDWIDTH_MBPS = 6  # 新版本模型需要修改
MIN_BANDWIDTH_MBPS = 0.7
LOG_MAX_BANDWIDTH_MBPS = np.log(MAX_BANDWIDTH_MBPS)
LOG_MIN_BANDWIDTH_MBPS = np.log(MIN_BANDWIDTH_MBPS)


class Delay_gradient_calculator:
    def __init__(self, path: str, size: int) -> None:
        self.timestamps = []
        self.size = size
        self.path = path

    # data = (send_timestamp, receive_timestamp)

    def push(self, data: Tuple[int, int]) -> None:
        # if len(self.timestamps) > self.size:
        #     self.timestamps = self.timestamps[1:]
        self.timestamps.append(data)

    def get_gradient(self) -> np.float64:
        if len(self.timestamps) < 2:
            return 0.0
        ts = self.timestamps[:self.size]
        gradients = []
        for i in range(1, len(ts)):
            s = ts[i][0] - ts[i - 1][0]
            r = ts[i][1] - ts[i - 1][1]
            gradients.append(r - s)
        return np.mean(gradients)

    def export(self, path: str):
        df = pd.DataFrame(self.timestamps)
        df.to_csv(path, index=False)

    def to_pic(self, episode: int) -> None:
        ts = self.timestamps
        gradients = []
        for i in range(1, len(ts)):
            s = ts[i][0] - ts[i - 1][0]
            r = ts[i][1] - ts[i - 1][1]
            gradients.append(r - s)
        x = [ts[1] for ts in self.timestamps[1:]]
        y = gradients
        plt.plot(x, y)
        plt.savefig(os.path.join(self.path, 'dgc-%d' % episode))
        plt.close()

    def reset(self):
        self.timestamps = []


class Packet_logger:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.packet_records = []

    def export(self, path: str):
        pass


class RL_logger:
    def __init__(self, path: str) -> None:
        self.base_path = path
        self.states = []
        self.actions = []
        self.rewards = []
        self.times = []
        self.state_names = [
            'log(receiving_rate)', 'min(delay / 1000, 1)', 'loss_ratio',
            'log(latest_prediction)', 'log(delta_prediction)', 'trendline'
        ]

    def push(self, state: np.ndarray, action: np.double, reward: np.double,
             time: int) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.times.append(time)

    def export(self, episode: int) -> None:
        fig, ax = plt.subplots(3, 3)
        # fig.subplots_adjust(hspace=0.5, wspace=0.5)
        fig.set_figwidth(16)
        fig.set_figheight(16)
        state_dim = len(self.states[0])

        def index2location(index: int) -> Tuple[int, int]:
            return index // 3, index % 3

        for i in range(state_dim):
            ax[index2location(i)].plot(self.times, [s[i] for s in self.states])
            ax[index2location(i)].set_title(self.state_names[i])
        ax[index2location(state_dim)].plot(self.times, self.actions)
        ax[index2location(state_dim)].set_title('action')
        ax[index2location(state_dim + 1)].plot(self.times, self.rewards)
        ax[index2location(state_dim + 1)].set_title('reward')
        fig.savefig(f"{self.base_path}/episode_{episode}.png")
        plt.close()
        # plt.savefig(f"{self.base_path}/episode_{episode}.png")

    def reset(self) -> None:
        self.states = []
        self.actions = []
        self.rewards = []
        self.times = []


class GymEnv:
    def __init__(self,
                 aw: ActionWrapper,
                 path: str,
                 report_interval_ms=0,
                 train_mode='gcc',
                 isLibra=False,
                 port_num = 0):
        self.gym_env = None
        self.step_time = report_interval_ms
        self.train_mode = train_mode
        self.aw = aw
        self.isLibra = isLibra
        self.port_num = port_num
        # self.last_interval_rtime = None

    def reset(
        self,
        trace_paths: str,
        episode_: int,
        logger_enabled: bool = False,
        duration_time_ms_: int = 30000,
        loss_rate_: float = 0.0
    ):
        self.gym_env = alphartc_gym.Gym()
        self.gym_env.reset(
            trace_path=trace_paths,
            # self.gym_env.reset(trace_path="rl_script/traces/trace_300k.json",#5G_12mbps.json",#"_WIRED_200kbps.json"4G_500kbps.json
            report_interval_ms=self.step_time,
            train_mode=self.train_mode,
            duration_time_ms = duration_time_ms_,
            loss_rate = loss_rate_,
            port_num = self.port_num,
            episode = episode_)
        return self.aw.reset(logger_enabled, episode_)

    def export_log(self, episode: int):
        # self.rl_logger.export(episode)
        pass


    def step(self, action, w_state):
        print("action:"+str(action))
        if self.isLibra:
            # print("is Libra")
            bandwidth_prediction = action
        else:
            # print("is not Libra")
            bandwidth_prediction = self.aw.action(action)

        msg = f'{int(bandwidth_prediction)}, {int(w_state.value)}'

        packet_list, done = self.gym_env.step(msg)
        # if self.last_interval_rtime:
        #     packet_list.last_interval_rtime = self.last_interval_rtime
        # logging.warning("step:packet_list"+str(packet_list))
        is_valid, states, reward = self.aw.update(packet_list)
        # packet_list.clear()
        # if len(packet_list)>0:
        #     self.last_interval_rtime  = packet_list[-1]['timestamp']

        return is_valid, states, reward, done, {}
