import gym
import logging
import numpy as np
from ns3_env import NS3Env, CCType
from gcc_obs_module import Estimator
from gcc_action_module.gcc_actor import GCCActor
import os
import uuid


class LokiRLEnv(gym.Env):
    '''
    LokiGCCEnv 是在模仿学习框架下训练 Loki 的黑盒 GCC 模型使用的环境。黑盒 GCC 的 Action
    空间是从一个固定的10维向量中选出的值。向量为从0.85到1.0025的10个数字。观测的结果是一个向
    量，包括
    '''

    def __init__(self, traces_path: str, cc_type: CCType, duration: int, time_interval: int, is_training: bool):
        # 设置logger
        self.logger = logging.getLogger("loki")
        if is_training:
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger.setLevel(logging.INFO)
        # 读取所有trace
        if not os.path.exists(traces_path):
            raise Exception(f"traces_path {traces_path} not exists")
        self.logger.info(f"traces_path is {traces_path}")
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
            low=np.array([0, -1024, 0, 0, 0] * 1),
            high=np.array([1, 1024, 1024, 10, 10] * 1),
            dtype=np.float32)
        # 动作备选值
        # self.action_list = np.array([
        #     0.7, 0.83, 0.96, 1.09, 1.22, 1.35, 1.48, 1.61, 1.74, 1.87])
        self.action_list = np.array([
            0.5, 0.7, 0.85, 0.95, 1, 1.05, 1.2, 1.5, 1.8, 2])
        self.STEP_TIME = time_interval
        zmq_path = str(uuid.uuid4())
        self.env = NS3Env(f'loki-{zmq_path}')

    def reset(self):
        # 选取一个trace
        trace = np.random.choice(self.traces)
        self.logger.info(f"reset with trace {trace}")
        # 重启NS3环境
        self.env.reset(
            trace,
            self.cc_type,
            self.duration)
        # 上次选定的带宽
        self.last_bw_action = 0.3  # Mbps
        self.max_bw = self.last_bw_action
        self.min_delay = 50
        # 观测历史
        self.obs_history = np.zeros(1*5)
        self.lastest_receiving_rate = 0
        self.loss_ratio = 0
        # 重置时间
        self.first_time = 0
        self.time_now = 0
        # GCC OBS 模块
        self.estimator = Estimator()
        self.gcc_actor = GCCActor()
        # 慢启动
        self.inited = False
        return self.obs_history

    def step(self, action):
        # action 是一个整数，表示从0.85到1.0025的10个数字中选出的index
        # 通过 action_list 将 action 转换为实际的值
        if not self.inited:
            self.last_bw_action *= 2 * np.log(2)
        else:
            self.last_bw_action *= self.action_list[action]
        hitwall = False
        if self.last_bw_action > 6 or self.last_bw_action < 0.1:
            hitwall = True
        self.last_bw_action = np.clip(self.last_bw_action, 0.1, 6)
        # 通过环境执行动作
        self.env.send(f"{int(self.last_bw_action * 1e6)}, 0")
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
        self.max_bw = max(self.max_bw, recv_mpbs)
        if delay > 2 * self.min_delay:
            self.inited = True

        self.obs_history = np.roll(self.obs_history, -5)
        self.obs_history[-5:] = [loss_ratio, trendline, self.min_delay / delay,
                                 self.last_bw_action / self.max_bw, self.gcc_actor.get_state(trendline)]
        # 计算奖励

        # 系数
        # k_a = 10
        # k_b = 100
        # k_c = 0.02
        # k_d = 10
        # k_e = 5
        # k_f = 0.1
        # k_g = 10
        # k_h = 5
        # k_i = 0.01

        # 根据当前环境的表现按分类给予奖励
        reward = 0

        # 1. 丢包率过高
        # if loss_ratio > 0.1:
        #     reward = (self.last_bw_action - act) * k_a - loss_ratio * k_b - delay * k_c
        # # 2. 延迟过高
        # elif delay > 75:
        #     reward = (self.last_bw_action - act) * k_d - loss_ratio * k_e - delay * k_f
        # # 3. 丢包率和延迟都不高
        # else:
        #     reward = (act - self.max_bw) * k_g + 2

        delay_metric = self.min_delay / (delay * 2)
        # reward = (recv_mpbs - 20 * loss_ratio * recv_mpbs) / self.max_bw * \
        #     delay_metric - delay_metric - 3 if hitwall else 0

        # reward = alpha * (recv_mpbs - loss_percent)/delay - beta * (act - self.last_bw_action)
        # gain = alpha * (recv_mpbs - loss_ratio)/delay
        # punish = beta * (act - self.last_bw_action)

        reward_orca = 7 * (recv_mpbs - 0.7 * loss_ratio *
                           recv_mpbs) / self.max_bw + delay_metric

        reward_bw = 3 * recv_mpbs / 2
        reward = reward_orca + reward_bw - (20 if hitwall else 0)

        self.logger.info(
            f'env_time={self.time_now}, bitrate={self.last_bw_action}, action={action}, recv_mpbs={recv_mpbs:.2f}, loss_ratio={loss_ratio}, reward={reward:.4f}, trendline={trendline:.4f}, delay={delay:.4f}')
        self.logger.info(
            f'state: {self.obs_history}'
        )

        return self.obs_history, reward, False, {}

    def render(self, mode='human'):
        return None

    def close(self):
        del self.env
