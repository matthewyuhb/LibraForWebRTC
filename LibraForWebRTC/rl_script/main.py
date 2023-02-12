#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################ 该版本是吴泓宇实现的trendline的移植版本请关闭训练模块也就是将max_num_episodes设置为0;有一个虚假的预训练模型即可，实际上不会调用
import os

import torch

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import draw
from rtc_env import GymEnv
from deep_rl.storage import Storage
from deep_rl.ppo_agent import PPO
from deep_rl.actor_critic import ActorCritic
from enum import Enum
import numpy as np
import pandas as pd  # 保存数据用
from pandas import DataFrame as df
from optparse import OptionParser

#PPO 算法 基于Actor-Critic算法实现，Critic相当于价值网络，Actor相当于策略网络

class WLibraState(Enum):
    Exploration = 0
    Evaluation = 1
    Exploitation = 2


def main():
    parser = OptionParser()

    parser.add_option("-d", "--is_test_mode", dest="TestMode", default=1,
                      help="1:test_mode 0:train_mode")
    (options, args) = parser.parse_args()
    # duration=int(options.Duration)
    is_test_mode = int(options.TestMode)

    w_state = WLibraState.Exploration
    Test =is_test_mode  ## 0 表示训练不利用特定model进行测试;1表示利用特定model测试但是不训练; 1的时候会将max_num_episodess
    print("Test:"+str(Test))
    ############## Hyperparameters for the experiments ##############
    env_name = "AlphaRTC"
    max_num_episodes = 2000  # maximal episodes
    if Test == 1:
        max_num_episodes = 1

    update_interval = 90000  # update policy every update_interval timesteps
    save_interval = 5  # save model every save_interval episode

    exploration_param = 0.05  # the std var of action distribution
    K_epochs = 32  # update policy for K_epochs
    ppo_clip = 0.2  # clip parameter of PPO
    gamma = 0.95  # discount factor
    lr = 5e-3  # Adam parameters
    betas = (0.9, 0.999)
    state_dim = 5
    action_dim = 1

    data_path = f'.rl_script/data/'  # Save model and reward curve here
    #############################################

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    print("Starting gym...")
    env = GymEnv()
    env.set_state_dim(state_dim)
    storage = Storage()  # used for storing data
    model_path="rl_script/data/ppo_2021_07_20_09_15_37.pth"
    ppo = PPO(state_dim, action_dim, exploration_param, lr, betas, gamma,
              K_epochs, ppo_clip,model_path)
    #ppo.load_state_dict(torch.load('{}pretrained_model.pth'.format(data_path)))
    record_episode_reward = []  #记录每一轮的reward
    episode_reward = 0
    time_step = 0
    

    print("Starting training loop...")
    # training loop
    for episode in range(max_num_episodes):
        print("Episode:"+str(episode))
        state_dim_list_init = env.reset();
        done = False
        while not done and time_step < update_interval:
            print("Time:"+str(time_step))
            state = torch.Tensor(state_dim_list_init)
            while not done and time_step < update_interval:
                action = ppo.select_action(state, storage)
                print("matthew:stepping...")
                state, reward, done, _ = env.step(action,w_state)
                print("matthew:recv...")
                # print("matthew:isDone:"+str(done))
                state = torch.Tensor(state)
                # Collect data for update
                storage.rewards.append(reward)
                storage.is_terminals.append(done)
                time_step += 1
                episode_reward += reward

        next_value = ppo.get_value(state)
        storage.compute_returns(next_value, gamma)

        # update
        policy_loss, val_loss = ppo.update(storage, state)
        storage.clear_storage()
        episode_reward /= time_step
        print("time_step", time_step)
        record_episode_reward.append(episode_reward)
        print(
            'Episode {} \t Average policy loss, value loss, reward {}, {}, {}'.
            format(episode, policy_loss, val_loss, episode_reward))

        if (episode > 0 and not (episode % save_interval)) or episode>=max_num_episodes-10: # 最后10次
            ppo.save_model(data_path)
            plt.plot(range(len(record_episode_reward)), record_episode_reward)
            plt.xlabel('Episode')
            plt.ylabel('Averaged episode reward')
            plt.savefig('%sreward_record.jpg' % (data_path))

        episode_reward = 0
        time_step = 0
    print('max_num_episodes:', max_num_episodes)
    print("max_time_step", time_step)
    if Test == 0:
        # 训练画图 和下面的不兼容
        draw.draw_module(ppo.policy, data_path)
        env.record_processing()
        env.draw_processing()

    else:
        ## 使用预训练模型画图
        #model_path = "./data/ppo_2021_07_13_14_36_14.pth"
        # model_path = "./data/ppo_2021_07_14_03_47_19.pth"
        # model_path= "./data/ppo_2021_07_14_22_13_17.pth"
        #model_path="./data/ppo_2021_07_15_17_19_38.pth"
        # model_path="./data/ppo_2021_07_17_09_37_58.pth" # data005
        model_path="./rl_script/data/ppo_2021_07_20_09_15_37.pth" # gym-c2 data007
        model = ActorCritic(state_dim, action_dim, exploration_param=0.05)
        model.load_state_dict(torch.load(model_path))
        draw.draw_module(model, data_path)
    #导出reward数据
    output_report = np.array(record_episode_reward)
    output_report = pd.DataFrame(output_report)
    output_report.to_csv("{}record_reward.csv".format(data_path))


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('time cost:', time_end - time_start, 's')

