import gym
import logging
import numpy as np
from ns3_env import NS3Env, CCType
# from gcc_obs_module import Estimator
import os
logging.basicConfig(level=logging.INFO)

class LokiRLEnv(gym.Env):
    '''
    LokiGCCEnv 是在模仿学习框架下训练 Loki 的黑盒 GCC 模型使用的环境。黑盒 GCC 的 Action
    空间是从一个固定的10维向量中选出的值。向量为从0.85到1.0025的10个数字。观测的结果是一个向
    量，包括
    '''

    def __init__(self, traces_path: str, cc_type: CCType, duration: int, time_interval: int):
        # 读取所有trace
        if not os.path.exists(traces_path):
            raise Exception(f"traces_path {traces_path} not exists")
        if not os.path.isdir(traces_path):
            self.traces = [traces_path]
        else:
            self.traces = os.listdir(traces_path)
            # 过滤json文件
            self.traces = [x for x in self.traces if x.endswith('.json')]
            self.traces = [os.path.join(traces_path, t) for t in self.traces]

        self.cc_type = cc_type
        self.duration = duration
        # 动作空间，从一个固定的10维向量中选出的值。向量为从0.85到1.0025的10个数字。
        self.action_space = gym.spaces.Discrete(10)  # type: ignore
        # 观测空间，2*5的一维向量，其中每一维是分别表示 loss_ratio 和 delay_jitter的两个
        # 值
        self.observation_space = gym.spaces.Box(     # type: ignore
            low=np.array([0, -1000, 0, 0] * 5),
            high=np.array([1, 1000, 1000, 10] * 5),
            dtype=np.float32)
        # 动作备选值
        self.action_list = np.array([
            0.7, 0.83, 0.96, 1.09, 1.22, 1.35, 1.48, 1.61, 1.74, 1.87])
        self.STEP_TIME = time_interval
        self.env = NS3Env()

    def reset(self):
        # 选取一个trace
        trace = np.random.choice(self.traces)
        logging.info(f"reset with trace {trace}")
        # 重启NS3环境
        self.env.reset(
            trace,
            self.cc_type,
            self.duration)
        # 上次选定的带宽
        self.last_bw_action = 0.7
        # 观测历史
        self.obs_history = np.zeros(20)
        self.lastest_receiving_rate = 0
        self.loss_ratio = 0
        # 重置时间
        self.first_time = 0
        self.time_now = 0
        # GCC OBS 模块
        # self.estimator = Estimator()
        return self.obs_history

    def step(self, action):
        # action 是一个整数，表示从0.85到1.0025的10个数字中选出的index
        # 通过 action_list 将 action 转换为实际的值
        act = self.action_list[action]
        # 通过环境执行动作
        self.env.send(f"{int(act * 1e6)}, 0")
        resp = self.env.recv()
        if resp is None:
            return self.obs_history, 0, True, {}

        # 通过GCC计算Trendline
        trendline = None
        if len(resp) > 0:
            self.time_now = resp[-1]['time_now']
        receiving_rate, loss_ratio, trendline, delay = self.estimator.parse_packets(
            resp)
        recv_mpbs = receiving_rate / 1e6

        self.obs_history = np.roll(self.obs_history, -4)
        self.obs_history[-4:] = [loss_ratio, trendline, delay, recv_mpbs]
        # 计算奖励

        # 系数
        k_a = 10
        k_b = 100
        k_c = 0.02
        k_d = 10
        k_e = 5
        k_f = 0.1
        k_g = 10
        k_h = 5
        k_i = 0.01

        # 根据当前环境的表现按分类给予奖励
        reward = 0
        # 1. 丢包率过高
        if loss_ratio > 0.1:
            reward = (recv_mpbs - act) * k_a - loss_ratio * k_b - delay * k_c
        # 2. 延迟过高
        elif delay > 75:
            reward = (recv_mpbs - act) * k_d - loss_ratio * k_e - delay * k_f
        # 3. 丢包率和延迟都不高
        else:
            reward = (act - recv_mpbs) * k_g - loss_ratio * k_h - delay * k_i
        
        # reward = alpha * (recv_mpbs - loss_percent)/delay - beta * (act - self.last_bw_action)
        # gain = alpha * (recv_mpbs - loss_ratio)/delay
        # punish = beta * (act - self.last_bw_action)
        self.last_bw_action = act

        
        logging.info(
            f'env_time={self.time_now}, bitrate={self.last_bw_action}, action={action}, recv_mpbs={recv_mpbs}, loss_ratio={loss_ratio}, reward={reward}, trendline={trendline}, delay={delay}')

        return self.obs_history, reward, False, {}

    def render(self, mode='human'):
        return None

    def close(self):
        del self.env
