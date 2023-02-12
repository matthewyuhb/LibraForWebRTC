from abc import ABC, abstractmethod
from asyncio.log import logger
from cmath import log
import imp
import logging
import numpy as np
import tensorflow.compat.v1 as tf
import sys
import os
import random
from enum import Enum, auto
# from libra_pygcc import WLibraState
from gcc.delaybasedbwe import delay_base_bwe, kBwOverusing, \
    kBwNormal, kBwUnderusing
from gcc.ack_bitrate_estimator import Ack_bitrate_estimator
from rl_logger import MLogger, Graph, MultiGraph

from deep_rl.utils import Params
from deep_rl.agent import Agent
from loki.fusion_agent import LokiFusionAgent

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "gym"))
from alphartc_gym.utils.packet_record import PacketRecord
from alphartc_gym.utils.packet_info import PacketInfo


class AgentWrapper(ABC):
    def __init__(self, log_base: str) -> None:
        self.STATE_DIM = 0

    @abstractmethod
    def action(self, a: np.float64) -> np.float64:
        pass

    @abstractmethod
    def update(self, packet_list: list) -> tuple:
        pass

    def get_state_dim(self) -> int:
        return self.STATE_DIM

    @abstractmethod
    def reset(self, logger_enabled: bool) -> list:
        pass


class GCCAgent(AgentWrapper):
    def __init__(self, log_base) -> None:
        # Constants
        self.STATE_DIM = 1
        # Fields
        self.first_time = 0
        self.time_now = 0
        self.gcc_bitrate = 30000
        self.logger_enabled = False
        # Objects
        self.packet_record = PacketRecord()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.gcc_rate_controller = delay_base_bwe()
        self.gcc_rate_controller.set_time(self.first_time)
        self.gcc_rate_controller.set_start_bitrate(self.gcc_bitrate)
        # Logger
        self.logger = MLogger('GCC', log_base)

    def reset(self, logger_enabled: bool, episode: int) -> list:
        # Fields
        self.first_time = 0
        self.time_now = 0
        self.gcc_bitrate = 30000
        self.logger_enabled = logger_enabled
        # Objects
        self.packet_record.reset()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.gcc_rate_controller = delay_base_bwe()
        self.gcc_rate_controller.set_time(self.first_time)
        self.gcc_rate_controller.set_start_bitrate(self.gcc_bitrate)
        # Logger
        super().reset(logger_enabled)
        return [0.0 for _ in range(self.STATE_DIM)]

    def action(self, a: np.float64) -> np.float64:
        return self.gcc_bitrate

    def update(self, packet_list: list) -> tuple:
        # 从数据包中提取信息
        for pkt in packet_list:
            packet_info = PacketInfo()
            packet_info.payload_type = pkt["payload_type"]
            packet_info.ssrc = pkt["ssrc"]
            packet_info.sequence_number = pkt["sequence_number"]
            packet_info.send_timestamp = pkt["send_time_ms"]
            packet_info.receive_timestamp = pkt["arrival_time_ms"]
            packet_info.padding_length = pkt["padding_length"]
            packet_info.header_length = pkt["header_length"]
            packet_info.payload_size = pkt["payload_size"]
            self.time_now = pkt["time_now"]
            self.packet_record.on_receive(packet_info)
        states = []
        reward = 0.0

        # 计算 GCC 速率
        gcc_bitrate = 0
        if len(packet_list) > 0:
            now_ts = self.time_now - self.first_time
            self.gcc_ack_bitrate.ack_estimator_incoming(packet_list)
            result = self.gcc_rate_controller.delay_bwe_incoming(
                packet_list, self.gcc_ack_bitrate.ack_estimator_bitrate_bps(),
                now_ts)
            # if self.gcc_rate_controller.rate_control.state == 2:
            # if self.gcc_rate_controller.detector.state == 2:
            #     logging.info("Overusing!")
            # else:
            #     logging.info("Underusing!")

            gcc_bitrate = result.bitrate
            self.gcc_bitrate = gcc_bitrate
        states.append(gcc_bitrate)

        return False,states, reward


class OrcaAgent(AgentWrapper):
    def __init__(self, step_time: int, log_base: str) -> None:
        # Constants
        self.STATE_DIM = 5
        self.UNIT_M = 1000000
        self.MAX_BANDWIDTH_MBPS = 2
        self.MIN_BANDWIDTH_MBPS = 0.7
        self.LOG_MAX_BANDWIDTH_MBPS = np.log(self.MAX_BANDWIDTH_MBPS)
        self.LOG_MIN_BANDWIDTH_MBPS = np.log(self.MIN_BANDWIDTH_MBPS)
        self.STEP_TIME = step_time
        # Fields
        self.first_time = 0
        self.time_now = 0
        self.current_bandwidth_est = 30000
        self.latest_bandwidth_est = 30000
        self.max_bandwidth = 30000
        self.min_delay = 999999
        self.logger_enabled = False
        # Objects
        self.packet_record = PacketRecord()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.dbb = delay_base_bwe()
        self.dbb.set_time(self.first_time)
        self.dbb.set_start_bitrate(self.latest_bandwidth_est)
        # Logging
        self.mlogger = MLogger('Orca', log_base)
        self.log = self.mlogger.logger
        self.log_times = []
        self.log_states = []
        self.log_rewards = []
        self.log_actions = []

    def reset(self, logger_enabled: bool, episode: int) -> list:
        # Fields
        self.first_time = 0
        self.time_now = 0
        self.current_bandwidth_est = 30000
        self.latest_bandwidth_est = 30000
        self.max_bandwidth = 30000
        self.min_delay = 999999
        self.episode = episode
        # Objects
        self.packet_record.reset()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.dbb.reset()
        self.dbb.set_time(self.first_time)
        self.dbb.set_start_bitrate(self.latest_bandwidth_est)
        # Logger
        if self.logger_enabled:
            m = MultiGraph('Overview')
            for i in range(self.STATE_DIM):
                m.new_graph(
                    Graph(x=self.log_times,
                          y=[s[i] for s in self.log_states],
                          title=f'state {i}'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_rewards, title='rewards'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_actions, title='actions'))
            self.mlogger.new_multigraph(m)
            self.mlogger.export(episode)
        self.mlogger.reset()
        self.log_times = []
        self.log_states = []
        self.log_rewards = []
        self.log_actions = []
        self.logger_enabled = logger_enabled
        return [0.0 for _ in range(self.STATE_DIM)]

    def __linear_to_log(self, value):
        # from 80kbps~20Mbps to 0~1 如果小于80k为0
        value = np.clip(value / self.UNIT_M, self.MIN_BANDWIDTH_MBPS,
                        self.MAX_BANDWIDTH_MBPS)
        log_value = np.log(value)
        return (log_value - self.LOG_MIN_BANDWIDTH_MBPS) / (
            self.LOG_MAX_BANDWIDTH_MBPS - self.LOG_MIN_BANDWIDTH_MBPS)

    def __log_to_linear(self, value):
        # from 0~1 to 10kbps to 8Mbps
        value = np.clip(value, 0, 1)
        log_bwe = value * (
            self.LOG_MAX_BANDWIDTH_MBPS -
            self.LOG_MIN_BANDWIDTH_MBPS) + self.LOG_MIN_BANDWIDTH_MBPS
        return np.exp(log_bwe) * self.UNIT_M

    def action(self, a: np.float64) -> np.float64:
        self.log_actions.append(np.squeeze(a))
        a = np.squeeze(a)
        a = np.power(2, a) * self.current_bandwidth_est
        self.latest_bandwidth_est = self.current_bandwidth_est
        self.current_bandwidth_est = a
        return np.clip(a, self.MIN_BANDWIDTH_MBPS * self.UNIT_M,
                       self.MAX_BANDWIDTH_MBPS * self.UNIT_M)

    def update(self, packet_list: list) -> tuple:
        for pkt in packet_list:
            packet_info = PacketInfo()
            packet_info.payload_type = pkt["payload_type"]
            packet_info.ssrc = pkt["ssrc"]
            packet_info.sequence_number = pkt["sequence_number"]
            packet_info.send_timestamp = pkt["send_time_ms"]
            packet_info.receive_timestamp = pkt["arrival_time_ms"]
            packet_info.padding_length = pkt["padding_length"]
            packet_info.header_length = pkt["header_length"]
            packet_info.payload_size = pkt["payload_size"]
            self.time_now = pkt["time_now"]
            self.packet_record.on_receive(packet_info)

        receiving_rate = self.packet_record.calculate_receiving_rate(
            interval=self.STEP_TIME)

        self.max_bandwidth = max(self.max_bandwidth, receiving_rate)

        delay = self.packet_record.calculate_average_delay(
            interval=self.STEP_TIME)
        self.min_delay = min(
            self.min_delay,
            self.packet_record.min_seen_delay)  # min(self.min_delay, delay)

        loss_ratio = self.packet_record.calculate_loss_ratio(
            interval=self.STEP_TIME)

        # latest_prediction = self.packet_record.calculate_latest_prediction()

        delay = max(1, delay)
        delay_metric = self.min_delay / delay

        # ------------------------- Trendline ---------------------------------
        length_packet = len(packet_list)
        if length_packet > 0:
            now_ts = self.time_now - self.first_time
            self.gcc_ack_bitrate.ack_estimator_incoming(packet_list)
            # print("ack",self.gcc_ack_bitrate.ack_estimator_bitrate_bps())
            self.dbb.delay_bwe_incoming(
                packet_list, self.gcc_ack_bitrate.ack_estimator_bitrate_bps(),
                now_ts)
        trendline = self.dbb.get_trendline_slope()
        # ---------------------------------------------------------------------

        states = []
        states.append(self.__linear_to_log(receiving_rate))
        states.append(loss_ratio)
        states.append(min(delay / 1000, 1))
        states.append(self.__linear_to_log(self.latest_bandwidth_est))
        states.append(trendline)

        reward = (
            receiving_rate - 5 * loss_ratio * receiving_rate
        ) / self.max_bandwidth * delay_metric + 0.5 * receiving_rate / 2000000

        if self.latest_bandwidth_est:
            reward = reward - 0.5 * abs(
                self.current_bandwidth_est -
                self.latest_bandwidth_est) / self.latest_bandwidth_est

        reward = np.squeeze(reward)

        if self.logger_enabled:
            self.log_times.append(self.time_now)
            self.log_states.append(states)
            self.log_rewards.append(reward)

        return states, reward


class FusionAgent(AgentWrapper):
    class GCCState(Enum):
        UNDERUSING = auto()
        OVERUSING = auto()
    class WLibraState(Enum):
        # Exploration = 0
        # Evaluation = 1
        # Exploitation = 2
        Ordinary = 0
        EI_c_1 = 1 #
        EI_r_1 = 2 #
        EI_c_2 = 3 #receive ack of the class action and generate the reward 
        EI_r_2 = 4 #receive ack of the rl action and generate the reward
    def __init__(self, time_interval, base_dir) -> None:
        # Constants
        self.STATE_DIM = 5
        self.UNIT_M = 1000000
        self.MAX_BANDWIDTH_MBPS = 6
        self.MIN_BANDWIDTH_MBPS = 0.1
        self.LOG_MAX_BANDWIDTH_MBPS = np.log(self.MAX_BANDWIDTH_MBPS)
        self.LOG_MIN_BANDWIDTH_MBPS = np.log(self.MIN_BANDWIDTH_MBPS)
        self.STEP_TIME = time_interval
        self.HITWALL_THRESHOLD = 2
        # Fields
        self.gcc_state = self.GCCState.UNDERUSING
        self.latest_bandwidth = 30000
        self.current_bandwidth = self.latest_bandwidth
        self.time_now = 0
        self.first_time = 0
        self.gcc_bitrate = self.latest_bandwidth
        self.max_bandwidth = self.latest_bandwidth
        self.min_delay = 999999
        self.start_up = True
        self.hitwall = False
        self.latest_action = self.latest_bandwidth
        self.loss_rate = 0.0
        # Objects
        self.packet_record = PacketRecord()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.gcc_rate_controller = delay_base_bwe()
        self.gcc_rate_controller.set_time(self.first_time)
        self.gcc_rate_controller.set_start_bitrate(self.gcc_bitrate)
        # Loggers
        self.logger_enabled = False
        self.episode = 0
        self.mlogger = MLogger('Fusion', base_dir)
        self.log_times = []
        self.log_rewards = []
        self.log_states = []
        self.log_actions = []
        self.log_start = []
        self.log = self.mlogger.logger
        
        #--------------RL init------------------------
        global params
        params = Params(os.path.join(os.path.dirname(__file__), 'deep_rl/params.json'))

        self.single_s_dim, a_dim = self.STATE_DIM, params.dict['action_dim']
        rec_s_dim = self.single_s_dim * params.dict['rec_dim']

        
        params.dict['train_dir'] = os.path.join(os.path.dirname(__file__), params.dict['logdir'])
        
        tf.set_random_seed(1234)
        random.seed(1234)
        np,random.seed(1234)
        tf.compat.v1.disable_eager_execution()
        summary_writer = tf.summary.FileWriterCache.get(params.dict['logdir'])
        self.agent = Agent(rec_s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer,h1_shape=params.dict['h1_shape'],
                            h2_shape=params.dict['h2_shape'],stddev=params.dict['stddev'],mem_size=params.dict['memsize'],gamma=params.dict['gamma'],
                            lr_c=params.dict['lr_c'],lr_a=params.dict['lr_a'],tau=params.dict['tau'],PER=params.dict['PER'],CDQ=params.dict['CDQ'],
                            LOSS_TYPE=params.dict['LOSS_TYPE'],noise_type=params.dict['noise_type'],noise_exp=params.dict['noise_exp'])
        self.s1 = np.zeros([self.single_s_dim])
        self.s0_rec_buffer = np.zeros([rec_s_dim])
        self.s1_rec_buffer = np.zeros([rec_s_dim])
        # self.s0_rec_buffer[-1*single_s_dim:] = s0
        tfconfig = tf.ConfigProto(allow_soft_placement=True)

        self.sess = tf.Session(config=tfconfig)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        saver = tf.train.Saver()
        self.agent.assign_sess(self.sess)

        # learner部分
        self.agent.init_target()

        # if not Train:
        # max_episodes = 1
        logging.info(os.path.join(params.dict['ckptdir']))
        
        params.dict['ckptdir'] = os.path.join(os.path.dirname(__file__), params.dict['ckptdir'])
        # model_name = 'model-0714_112251-2970' #在fixed/step 场景下训练的不错
        # model_name = 'model-0720_095709-21630' #在2970基础上，在波动很大的trace下也进行了训练
        model_name = 'model-0721_164644-12570' 
        
        
        saver.restore(self.sess, os.path.join(params.dict['ckptdir'], model_name))
        
    def __linear_to_log(self, value):
        # from 80kbps~20Mbps to 0~1 如果小于80k为0
        value = np.clip(value / self.UNIT_M, self.MIN_BANDWIDTH_MBPS,
                        self.MAX_BANDWIDTH_MBPS)
        log_value = np.log(value)
        return (log_value - self.LOG_MIN_BANDWIDTH_MBPS) / (
            self.LOG_MAX_BANDWIDTH_MBPS - self.LOG_MIN_BANDWIDTH_MBPS)

    def action(self, a: np.float64) -> np.float64:
        # print("doing action...")
        # #----------- Loki ACTION -------------
        # if self.start_up:
        #     act_raw = 2 * np.log(2) * self.latest_action
        #     # print("start up:{}".format(act_raw))
        #     # act_raw = 200000
        # else:
        #     self.hitwall = False
        #     act = int(a*10)
        #     if act == 10 :
        #         act = 9

        #     a_list = [0.5,0.67,0.84,0.95,1,1.05,1.2,1.5,1.8,2]
        #     act_raw = self.latest_action * a_list[act]
        # self.latest_action = act_raw #act_raw-->act

        # self.log_actions.append(np.squeeze(a))

        # act = np.clip(act_raw, self.MIN_BANDWIDTH_MBPS * self.UNIT_M,
        #               self.MAX_BANDWIDTH_MBPS * self.UNIT_M)

        # return act
        # #--------- Loki ACTION END -----------

        if self.start_up:
            act_raw = 2 * np.log(2) * self.latest_action
            # print("start up:".format(act_raw))
            # act_raw = 200000
        else:
            exp = a
            # if (self.gcc_state is self.GCCState.UNDERUSING) and (self.loss_rate<0.1):
            #     logging.info("UNDERUSING!")
            #     exp = 0 + (a + 1) / 2
            # elif self.gcc_state is self.GCCState.OVERUSING or self.loss_rate>0.1:
            #     logging.info("OVERUSING!")
            #     exp = -1 + (a + 1) / 2
            act_raw = np.power(2, exp) * self.latest_action
            # act_raw = np.power(1.5, a) * self.latest_action
            
            if act_raw / self.UNIT_M >= self.MAX_BANDWIDTH_MBPS or \
                    act_raw / self.UNIT_M <= self.MIN_BANDWIDTH_MBPS:
                self.hitwall = True
            else:
                self.hitwall = False
            # logging.info("a:"+str(a) + " latest_action:" + str(self.latest_action)+ " act_raw:"+str(act_raw))
            # logging.info("a:"+str(a)+" exp:"+str(exp) + " latest_action:" + str(self.latest_action)+ " act_raw:"+str(act_raw))
            act_raw = np.squeeze(act_raw)
            # print("No start up:".format(act_raw))

        act = np.clip(act_raw, self.MIN_BANDWIDTH_MBPS * self.UNIT_M,
                      self.MAX_BANDWIDTH_MBPS * self.UNIT_M)
        self.latest_action = act #act_raw-->act
        self.log_actions.append(np.squeeze(a))
        # print("final action:{}".format(act))
        return act

    def reset(self, logger_enabled: bool, episode: int) -> list:
        # Fields
        self.gcc_state = self.GCCState.UNDERUSING
        self.latest_bandwidth = 30000
        self.current_bandwidth = self.latest_bandwidth
        self.time_now = 0
        self.first_time = 0
        self.gcc_bitrate = self.latest_bandwidth
        self.max_bandwidth = self.latest_bandwidth
        self.min_delay = 999999
        self.start_up = True
        self.hitwall = False
        self.latest_action = self.latest_bandwidth
        # Objects
        self.packet_record.reset()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.gcc_rate_controller = delay_base_bwe()
        self.gcc_rate_controller.set_time(self.first_time)
        self.gcc_rate_controller.set_start_bitrate(self.gcc_bitrate)
        # Loggers
        if self.logger_enabled:
            print("Draw Overview!")
            print("---------------------------")
            print(self.log_times)
            print(self.log_rewards)
            print(self.log_actions)
            print(self.log_start)
            print("---------------------------")
            print("---------------------------")
            print("---------------------------")

            m = MultiGraph('Overview')
            for i in range(self.STATE_DIM):
                m.new_graph(
                    Graph(x=self.log_times,
                          y=[s[i] for s in self.log_states],
                          title=f'state {i}'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_rewards, title='rewards'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_actions, title='actions'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_start, title='start up'))
            self.mlogger.new_multigraph(m)
            self.mlogger.export(episode)
        self.logger_enabled = logger_enabled
        
        self.episode = episode
        self.mlogger.reset()
        self.log_times = []
        self.log_rewards = []
        self.log_states = []
        self.log_actions = []
        self.log_start = []
        return [0.0 for _ in range(self.STATE_DIM)]

    def update(self,
               packet_list: list) -> tuple:
        # print("updating...")
        # ------------从数据包中提取信息---------------
        for pkt in packet_list:
            packet_info = PacketInfo()
            packet_info.payload_type = pkt["payload_type"]
            packet_info.ssrc = pkt["ssrc"]
            packet_info.sequence_number = pkt["sequence_number"]
            packet_info.send_timestamp = pkt["send_time_ms"]
            packet_info.receive_timestamp = pkt["arrival_time_ms"]
            packet_info.padding_length = pkt["padding_length"]
            packet_info.header_length = pkt["header_length"]
            packet_info.payload_size = pkt["payload_size"]
            self.time_now = pkt["time_now"]
            # print("updating...time:{}".format(self.time_now))

            self.packet_record.on_receive(packet_info)
            # print("pkt:"+str(pkt))
            
        #---------------计算RL的关键信息----------------------
        receiving_rate = self.packet_record.calculate_receiving_rate(
            interval=self.STEP_TIME)

        self.max_bandwidth = max(self.max_bandwidth, receiving_rate)

        delay = self.packet_record.calculate_average_delay(
            interval=self.STEP_TIME)
        self.min_delay = min(
            self.min_delay,
            self.packet_record.min_seen_delay)  # min(self.min_delay, delay)

        loss_ratio = self.packet_record.calculate_loss_ratio(
            interval=self.STEP_TIME)
        
        self.loss_rate = loss_ratio
        
        self.packet_record.clear()
        
        # --------------计算 GCC 速率-----------------
        gcc_bitrate = 0
        trendline = 0
        if len(packet_list) > 0:
            now_ts = self.time_now - self.first_time
            self.gcc_ack_bitrate.ack_estimator_incoming(packet_list)
            result = self.gcc_rate_controller.delay_bwe_incoming(
                packet_list, self.gcc_ack_bitrate.ack_estimator_bitrate_bps(),
                now_ts)
            gcc_bitrate = result.bitrate
            self.gcc_bitrate = gcc_bitrate
            trendline = self.gcc_rate_controller.get_trendline_slope()
            if self.gcc_rate_controller.detector.state == 2:
                # logging.info("OVERUSING")
                self.gcc_state = self.GCCState.OVERUSING
            else:
                # logging.info("UNDERUSINg")
                self.gcc_state = self.GCCState.UNDERUSING
            if loss_ratio>0.1:
                self.gcc_state = self.GCCState.OVERUSING
            # if self.rate_control.state == kBwOverusing:
            #     self.gcc_state = self.GCCState.OVERUSING
            # else:
            #     self.gcc_state = self.GCCState.UNDERUSING


        
        

        # latest_prediction = self.packet_record.calculate_latest_prediction()

        delay = max(1, delay)
        delay = self.min_delay if self.min_delay*1.3>delay else delay
        delay_metric = self.min_delay / (delay * 0.5)

        if delay > self.min_delay * 1.5:
            self.start_up = False
            
        
        #-----------------append states---------------------------
        
        # print("loss rate:"+str(loss_ratio))
        states = []
        states.append(loss_ratio)
        states.append(trendline)
        states.append(self.latest_action/self.max_bandwidth)
        states.append(self.min_delay/delay)
        states.append(self.gcc_rate_controller.detector.state-1.0)
        
        


 
        reward_orca = 7*(receiving_rate - 0.7 * loss_ratio *
                       receiving_rate) / self.max_bandwidth + delay_metric
                    
                    #    receiving_rate) / self.max_bandwidth * delay_metric

        reward_bw = 3 * (receiving_rate / self.UNIT_M -
                           self.MIN_BANDWIDTH_MBPS) / self.MAX_BANDWIDTH_MBPS

        reward = reward_orca + reward_bw - 3 * self.hitwall
        self.latest_bandwidth = receiving_rate

        if self.logger_enabled:
            self.log_times.append(self.time_now)
            self.log_states.append(states)
            self.log_rewards.append(np.squeeze(reward))
            self.log_start.append(self.start_up)
            
        self.s1 = states
        self.r = reward_orca
        
        self.s1_rec_buffer = np.concatenate((self.s0_rec_buffer[self.single_s_dim:], self.s1) )
        
        self.s0_rec_buffer = self.s1_rec_buffer

        return not self.start_up, states, reward
    
    def get_estimated_bandwidth(self)->tuple:
        if self.start_up:
            logging.info("Start up!")
            a_rl = self.agent.get_action(self.s1_rec_buffer,False) #if i%200!=1 else agent.get_action(s0_rec_buffer,False);
            self.a_final = self.action(np.squeeze(a_rl))
            self.prev_sending_rate = self.a_final
            self.latest_action = self.a_final
            self.gcc_rate_controller.rate_control.curr_rate = self.a_final
        else:
            # print("state:{}".format(self.s1))
            # self.s1_rec_buffer = np.concatenate((self.s0_rec_buffer[self.single_s_dim:], self.s1) )
            a_rl = self.agent.get_action(self.s1_rec_buffer,False) #if i%200!=1 else agent.get_action(s0_rec_buffer,False);
            a_rl = np.squeeze(a_rl)
            self.a_final = self.action(a_rl)#按照rl得出的对应的cwnd的值
        # print("get_estimated_bandwidth:{}".format(self.a_final))
        return self.a_final,self.WLibraState.Ordinary #int(20000000) # 1Mbps
        
class SPQL_FusionAgent(AgentWrapper):
    class GCCState(Enum):
        UNDERUSING = auto()
        OVERUSING = auto()
        
    def __init__(self, time_interval, base_dir) -> None:
        # Constants
        self.STATE_DIM = 3
        self.UNIT_M = 1000000
        self.MAX_BANDWIDTH_MBPS = 6
        self.MIN_BANDWIDTH_MBPS = 0.1
        self.LOG_MAX_BANDWIDTH_MBPS = np.log(self.MAX_BANDWIDTH_MBPS)
        self.LOG_MIN_BANDWIDTH_MBPS = np.log(self.MIN_BANDWIDTH_MBPS)
        self.STEP_TIME = time_interval
        self.HITWALL_THRESHOLD = 2
        # Fields
        self.gcc_state = self.GCCState.UNDERUSING
        self.latest_bandwidth = 30000
        self.current_bandwidth = self.latest_bandwidth
        self.time_now = 0
        self.first_time = 0
        self.gcc_bitrate = self.latest_bandwidth
        self.max_bandwidth = self.latest_bandwidth
        self.min_delay = 999999
        self.start_up = True
        self.hitwall = False
        self.latest_action = self.latest_bandwidth
        self.loss_rate = 0.0
        # Objects
        self.packet_record = PacketRecord()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.gcc_rate_controller = delay_base_bwe()
        self.gcc_rate_controller.set_time(self.first_time)
        self.gcc_rate_controller.set_start_bitrate(self.gcc_bitrate)
        # Loggers
        self.logger_enabled = False
        self.episode = 0
        self.mlogger = MLogger('Fusion', base_dir)
        self.log_times = []
        self.log_rewards = []
        self.log_states = []
        self.log_actions = []
        self.log_start = []
        self.log = self.mlogger.logger
        
    def __linear_to_log(self, value):
        # from 80kbps~20Mbps to 0~1 如果小于80k为0
        value = np.clip(value / self.UNIT_M, self.MIN_BANDWIDTH_MBPS,
                        self.MAX_BANDWIDTH_MBPS)
        log_value = np.log(value)
        return (log_value - self.LOG_MIN_BANDWIDTH_MBPS) / (
            self.LOG_MAX_BANDWIDTH_MBPS - self.LOG_MIN_BANDWIDTH_MBPS)

    def action(self, a: np.float64) -> np.float64:
        # print("doing action...")
        # #----------- Loki ACTION -------------
        # if self.start_up:
        #     act_raw = 2 * np.log(2) * self.latest_action
        #     # print("start up:{}".format(act_raw))
        #     # act_raw = 200000
        # else:
        #     self.hitwall = False
        #     act = int(a*10)
        #     if act == 10 :
        #         act = 9

        #     a_list = [0.5,0.67,0.84,0.95,1,1.05,1.2,1.5,1.8,2]
        #     act_raw = self.latest_action * a_list[act]
        # self.latest_action = act_raw #act_raw-->act

        # self.log_actions.append(np.squeeze(a))

        # act = np.clip(act_raw, self.MIN_BANDWIDTH_MBPS * self.UNIT_M,
        #               self.MAX_BANDWIDTH_MBPS * self.UNIT_M)

        # return act
        # #--------- Loki ACTION END -----------

        if self.start_up:
            act_raw = 2 * np.log(2) * self.latest_action
            # print("start up:".format(act_raw))
            # act_raw = 200000
        else:
            exp = a
            # if (self.gcc_state is self.GCCState.UNDERUSING) and (self.loss_rate<0.1):
            #     logging.info("UNDERUSING!")
            #     exp = 0 + (a + 1) / 2
            # elif self.gcc_state is self.GCCState.OVERUSING or self.loss_rate>0.1:
            #     logging.info("OVERUSING!")
            #     exp = -1 + (a + 1) / 2
            act_raw = np.power(2, exp) * self.latest_action
            # act_raw = np.power(1.5, a) * self.latest_action
            
            if act_raw / self.UNIT_M >= self.MAX_BANDWIDTH_MBPS or \
                    act_raw / self.UNIT_M <= self.MIN_BANDWIDTH_MBPS:
                self.hitwall = True
            else:
                self.hitwall = False
            # logging.info("a:"+str(a) + " latest_action:" + str(self.latest_action)+ " act_raw:"+str(act_raw))
            # logging.info("a:"+str(a)+" exp:"+str(exp) + " latest_action:" + str(self.latest_action)+ " act_raw:"+str(act_raw))
            act_raw = np.squeeze(act_raw)
            # print("No start up:".format(act_raw))

        act = np.clip(act_raw, self.MIN_BANDWIDTH_MBPS * self.UNIT_M,
                      self.MAX_BANDWIDTH_MBPS * self.UNIT_M)
        self.latest_action = act #act_raw-->act
        self.log_actions.append(np.squeeze(a))
        # print("final action:{}".format(act))
        return act

    def reset(self, logger_enabled: bool, episode: int) -> list:
        # Fields
        self.gcc_state = self.GCCState.UNDERUSING
        self.latest_bandwidth = 30000
        self.current_bandwidth = self.latest_bandwidth
        self.time_now = 0
        self.first_time = 0
        self.gcc_bitrate = self.latest_bandwidth
        self.max_bandwidth = self.latest_bandwidth
        self.min_delay = 999999
        self.start_up = True
        self.hitwall = False
        self.latest_action = self.latest_bandwidth
        # Objects
        self.packet_record.reset()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.gcc_rate_controller = delay_base_bwe()
        self.gcc_rate_controller.set_time(self.first_time)
        self.gcc_rate_controller.set_start_bitrate(self.gcc_bitrate)
        # Loggers
        if self.logger_enabled:
            print("Draw Overview!")
            print("---------------------------")
            print(self.log_times)
            print(self.log_rewards)
            print(self.log_actions)
            print(self.log_start)
            print("---------------------------")
            print("---------------------------")
            print("---------------------------")

            m = MultiGraph('Overview')
            for i in range(self.STATE_DIM):
                m.new_graph(
                    Graph(x=self.log_times,
                          y=[s[i] for s in self.log_states],
                          title=f'state {i}'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_rewards, title='rewards'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_actions, title='actions'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_start, title='start up'))
            self.mlogger.new_multigraph(m)
            self.mlogger.export(episode)
        self.logger_enabled = logger_enabled
        
        self.episode = episode
        self.mlogger.reset()
        self.log_times = []
        self.log_rewards = []
        self.log_states = []
        self.log_actions = []
        self.log_start = []
        return [0.0 for _ in range(self.STATE_DIM)]

    def update(self,
               packet_list: list) -> tuple:
        # print("updating...")
        # ------------从数据包中提取信息---------------
        for pkt in packet_list:
            packet_info = PacketInfo()
            packet_info.payload_type = pkt["payload_type"]
            packet_info.ssrc = pkt["ssrc"]
            packet_info.sequence_number = pkt["sequence_number"]
            packet_info.send_timestamp = pkt["send_time_ms"]
            packet_info.receive_timestamp = pkt["arrival_time_ms"]
            packet_info.padding_length = pkt["padding_length"]
            packet_info.header_length = pkt["header_length"]
            packet_info.payload_size = pkt["payload_size"]
            self.time_now = pkt["time_now"]
            # print("updating...time:{}".format(self.time_now))

            self.packet_record.on_receive(packet_info)
            # print("pkt:"+str(pkt))
            
        #---------------计算RL的关键信息----------------------
        receiving_rate = self.packet_record.calculate_receiving_rate(
            interval=self.STEP_TIME)

        self.max_bandwidth = max(self.max_bandwidth, receiving_rate)

        delay = self.packet_record.calculate_average_delay(
            interval=self.STEP_TIME)
        self.min_delay = min(
            self.min_delay,
            self.packet_record.min_seen_delay)  # min(self.min_delay, delay)

        loss_ratio = self.packet_record.calculate_loss_ratio(
            interval=self.STEP_TIME)
        
        self.loss_rate = loss_ratio
        
        self.packet_record.clear()
        
        # --------------计算 GCC 速率-----------------
        gcc_bitrate = 0
        trendline = 0
        if len(packet_list) > 0:
            now_ts = self.time_now - self.first_time
            self.gcc_ack_bitrate.ack_estimator_incoming(packet_list)
            result = self.gcc_rate_controller.delay_bwe_incoming(
                packet_list, self.gcc_ack_bitrate.ack_estimator_bitrate_bps(),
                now_ts)
            gcc_bitrate = result.bitrate
            self.gcc_bitrate = gcc_bitrate
            trendline = self.gcc_rate_controller.get_trendline_slope()
            if self.gcc_rate_controller.detector.state == 2:
                # logging.info("OVERUSING")
                self.gcc_state = self.GCCState.OVERUSING
            else:
                # logging.info("UNDERUSINg")
                self.gcc_state = self.GCCState.UNDERUSING
            if loss_ratio>0.1:
                self.gcc_state = self.GCCState.OVERUSING
            # if self.rate_control.state == kBwOverusing:
            #     self.gcc_state = self.GCCState.OVERUSING
            # else:
            #     self.gcc_state = self.GCCState.UNDERUSING


        
        

        # latest_prediction = self.packet_record.calculate_latest_prediction()

        delay = max(1, delay)
        delay = self.min_delay if self.min_delay*1.3>delay else delay
        delay_metric = self.min_delay / (delay * 0.5)

        if delay > self.min_delay * 1.5:
            self.start_up = False
            
        
        #-----------------append states---------------------------
        
        # print("loss rate:"+str(loss_ratio))
        states = []
        # states.append(loss_ratio)
        states.append(trendline)
        states.append(self.latest_action/self.max_bandwidth)
        states.append(self.min_delay/delay)
        # states.append(self.gcc_rate_controller.detector.state-1.0)
        
        


 
        reward_orca = 7*(receiving_rate - 0.7 * loss_ratio *
                       receiving_rate) / self.max_bandwidth + delay_metric
                    
                    #    receiving_rate) / self.max_bandwidth * delay_metric

        reward_bw = 3 * (receiving_rate / self.UNIT_M -
                           self.MIN_BANDWIDTH_MBPS) / self.MAX_BANDWIDTH_MBPS

        reward = reward_orca + reward_bw - 3 * self.hitwall
        self.latest_bandwidth = receiving_rate

        if self.logger_enabled:
            self.log_times.append(self.time_now)
            self.log_states.append(states)
            self.log_rewards.append(np.squeeze(reward))
            self.log_start.append(self.start_up)

        return not self.start_up, states, reward


class OnrlAgent(AgentWrapper):
    class GCCState(Enum):
        UNDERUSING = auto()
        OVERUSING = auto()
    class WLibraState(Enum):
        # Exploration = 0
        # Evaluation = 1
        # Exploitation = 2
        Ordinary = 0
       
    def __init__(self, time_interval, base_dir) -> None:
        # Constants
        self.STATE_DIM = 5
        self.UNIT_M = 1000000
        self.MAX_BANDWIDTH_MBPS = 2.5
        self.MIN_BANDWIDTH_MBPS = 0.7
        self.LOG_MAX_BANDWIDTH_MBPS = np.log(self.MAX_BANDWIDTH_MBPS)
        self.LOG_MIN_BANDWIDTH_MBPS = np.log(self.MIN_BANDWIDTH_MBPS)
        self.STEP_TIME = time_interval
        self.HITWALL_THRESHOLD = 2
        # Fields
        self.gcc_state = self.GCCState.UNDERUSING
        self.latest_bandwidth = 30000
        self.current_bandwidth = self.latest_bandwidth
        self.time_now = 0
        self.first_time = 0
        self.gcc_bitrate = self.latest_bandwidth
        self.max_bandwidth = self.latest_bandwidth
        self.min_delay = 999999
        self.start_up = True
        self.hitwall = False
        self.latest_action = self.latest_bandwidth
        self.loss_rate = 0.0
        # Objects
        self.packet_record = PacketRecord()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.gcc_rate_controller = delay_base_bwe()
        self.gcc_rate_controller.set_time(self.first_time)
        self.gcc_rate_controller.set_start_bitrate(self.gcc_bitrate)
        # Loggers
        self.logger_enabled = False
        self.episode = 0
        self.mlogger = MLogger('OnRL', base_dir)
        self.log_times = []
        self.log_rewards = []
        self.log_states = []
        self.log_actions = []
        self.log_start = []
        self.log = self.mlogger.logger
        
        #--------------RL init------------------------
        global params
        params = Params(os.path.join(os.path.dirname(__file__), 'deep_rl/params.json'))

        self.single_s_dim, a_dim = self.STATE_DIM, params.dict['action_dim']
        rec_s_dim = self.single_s_dim * params.dict['rec_dim']

        
        params.dict['train_dir'] = os.path.join(os.path.dirname(__file__), params.dict['logdir'])
        
        tf.set_random_seed(1234)
        random.seed(1234)
        np,random.seed(1234)
        tf.compat.v1.disable_eager_execution()
        summary_writer = tf.summary.FileWriterCache.get(params.dict['logdir'])
        self.agent = Agent(rec_s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer,h1_shape=params.dict['h1_shape'],
                            h2_shape=params.dict['h2_shape'],stddev=params.dict['stddev'],mem_size=params.dict['memsize'],gamma=params.dict['gamma'],
                            lr_c=params.dict['lr_c'],lr_a=params.dict['lr_a'],tau=params.dict['tau'],PER=params.dict['PER'],CDQ=params.dict['CDQ'],
                            LOSS_TYPE=params.dict['LOSS_TYPE'],noise_type=params.dict['noise_type'],noise_exp=params.dict['noise_exp'])
        self.s1 = np.zeros([self.single_s_dim])
        self.s0_rec_buffer = np.zeros([rec_s_dim])
        self.s1_rec_buffer = np.zeros([rec_s_dim])
        # self.s0_rec_buffer[-1*single_s_dim:] = s0
        tfconfig = tf.ConfigProto(allow_soft_placement=True)

        self.sess = tf.Session(config=tfconfig)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        saver = tf.train.Saver()
        self.agent.assign_sess(self.sess)

        # learner部分
        self.agent.init_target()

        # if not Train:
        # max_episodes = 1
        logging.info(os.path.join(params.dict['ckptdir']))
        
        params.dict['ckptdir'] = os.path.join(os.path.dirname(__file__), params.dict['ckptdir'])
        # model_name = 'model-0714_112251-2970' #在fixed/step 场景下训练的不错
        # model_name = 'model-0720_095709-21630' #在2970基础上，在波动很大的trace下也进行了训练
        # model_name = 'model-0721_164644-12570' 
        model_name = 'model-0129_221607-15300' #LibraForWebRTC/rl_script/ckpt_dir/model-0129_221607-15300.index
        
        
        saver.restore(self.sess, os.path.join(params.dict['ckptdir'], model_name))
        
    def __linear_to_log(self, value):
        # from 80kbps~20Mbps to 0~1 如果小于80k为0
        value = np.clip(value / self.UNIT_M, self.MIN_BANDWIDTH_MBPS,
                        self.MAX_BANDWIDTH_MBPS)
        log_value = np.log(value)
        return (log_value - self.LOG_MIN_BANDWIDTH_MBPS) / (
            self.LOG_MAX_BANDWIDTH_MBPS - self.LOG_MIN_BANDWIDTH_MBPS)

    def action(self, a: np.float64) -> np.float64:
        # print("doing action...")
        # #----------- Loki ACTION -------------
        # if self.start_up:
        #     act_raw = 2 * np.log(2) * self.latest_action
        #     # print("start up:{}".format(act_raw))
        #     # act_raw = 200000
        # else:
        #     self.hitwall = False
        #     act = int(a*10)
        #     if act == 10 :
        #         act = 9

        #     a_list = [0.5,0.67,0.84,0.95,1,1.05,1.2,1.5,1.8,2]
        #     act_raw = self.latest_action * a_list[act]
        # self.latest_action = act_raw #act_raw-->act

        # self.log_actions.append(np.squeeze(a))

        # act = np.clip(act_raw, self.MIN_BANDWIDTH_MBPS * self.UNIT_M,
        #               self.MAX_BANDWIDTH_MBPS * self.UNIT_M)

        # return act
        # #--------- Loki ACTION END -----------

        if self.start_up:
            act_raw = 2 * np.log(2) * self.latest_action
            # print("start up:".format(act_raw))
            # act_raw = 200000
        else:
            exp = a
            # if (self.gcc_state is self.GCCState.UNDERUSING) and (self.loss_rate<0.1):
            #     logging.info("UNDERUSING!")
            #     exp = 0 + (a + 1) / 2
            # elif self.gcc_state is self.GCCState.OVERUSING or self.loss_rate>0.1:
            #     logging.info("OVERUSING!")
            #     exp = -1 + (a + 1) / 2
            act_raw = np.power(2, exp) * self.latest_action
            # act_raw = np.power(1.5, a) * self.latest_action
            
            if act_raw / self.UNIT_M >= self.MAX_BANDWIDTH_MBPS or \
                    act_raw / self.UNIT_M <= self.MIN_BANDWIDTH_MBPS:
                self.hitwall = True
            else:
                self.hitwall = False
            # logging.info("a:"+str(a) + " latest_action:" + str(self.latest_action)+ " act_raw:"+str(act_raw))
            # logging.info("a:"+str(a)+" exp:"+str(exp) + " latest_action:" + str(self.latest_action)+ " act_raw:"+str(act_raw))
            act_raw = np.squeeze(act_raw)
            # print("No start up:".format(act_raw))

        act = np.clip(act_raw, self.MIN_BANDWIDTH_MBPS * self.UNIT_M,
                      self.MAX_BANDWIDTH_MBPS * self.UNIT_M)
        self.latest_action = act #act_raw-->act
        self.log_actions.append(np.squeeze(a))
        # print("final action:{}".format(act))
        return act

    def reset(self, logger_enabled: bool, episode: int) -> list:
        # Fields
        self.gcc_state = self.GCCState.UNDERUSING
        self.latest_bandwidth = 30000
        self.current_bandwidth = self.latest_bandwidth
        self.time_now = 0
        self.first_time = 0
        self.gcc_bitrate = self.latest_bandwidth
        self.max_bandwidth = self.latest_bandwidth
        self.min_delay = 999999
        self.start_up = True
        self.hitwall = False
        self.latest_action = self.latest_bandwidth
        # Objects
        self.packet_record.reset()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.gcc_rate_controller = delay_base_bwe()
        self.gcc_rate_controller.set_time(self.first_time)
        self.gcc_rate_controller.set_start_bitrate(self.gcc_bitrate)
        # Loggers
        if self.logger_enabled:
            print("Draw Overview!")
            print("---------------------------")
            print(self.log_times)
            print(self.log_rewards)
            print(self.log_actions)
            print(self.log_start)
            print("---------------------------")
            print("---------------------------")
            print("---------------------------")

            m = MultiGraph('Overview')
            for i in range(self.STATE_DIM):
                m.new_graph(
                    Graph(x=self.log_times,
                          y=[s[i] for s in self.log_states],
                          title=f'state {i}'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_rewards, title='rewards'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_actions, title='actions'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_start, title='start up'))
            self.mlogger.new_multigraph(m)
            self.mlogger.export(episode)
        self.logger_enabled = logger_enabled
        
        self.episode = episode
        self.mlogger.reset()
        self.log_times = []
        self.log_rewards = []
        self.log_states = []
        self.log_actions = []
        self.log_start = []
        return [0.0 for _ in range(self.STATE_DIM)]

    def update(self,
               packet_list: list) -> tuple:
        # print("updating...")
        # ------------从数据包中提取信息---------------
        for pkt in packet_list:
            packet_info = PacketInfo()
            packet_info.payload_type = pkt["payload_type"]
            packet_info.ssrc = pkt["ssrc"]
            packet_info.sequence_number = pkt["sequence_number"]
            packet_info.send_timestamp = pkt["send_time_ms"]
            packet_info.receive_timestamp = pkt["arrival_time_ms"]
            packet_info.padding_length = pkt["padding_length"]
            packet_info.header_length = pkt["header_length"]
            packet_info.payload_size = pkt["payload_size"]
            self.time_now = pkt["time_now"]
            # print("updating...time:{}".format(self.time_now))

            self.packet_record.on_receive(packet_info)
            # print("pkt:"+str(pkt))
            
        #---------------计算RL的关键信息----------------------
        receiving_rate = self.packet_record.calculate_receiving_rate(
            interval=self.STEP_TIME)

        self.max_bandwidth = max(self.max_bandwidth, receiving_rate)

        delay = self.packet_record.calculate_average_delay(
            interval=self.STEP_TIME)
        self.min_delay = min(
            self.min_delay,
            self.packet_record.min_seen_delay)  # min(self.min_delay, delay)

        loss_ratio = self.packet_record.calculate_loss_ratio(
            interval=self.STEP_TIME)
        
        self.loss_rate = loss_ratio
        
        self.packet_record.clear()
        
        # --------------计算 GCC 速率-----------------
        gcc_bitrate = 0
        trendline = 0
        if len(packet_list) > 0:
            now_ts = self.time_now - self.first_time
            self.gcc_ack_bitrate.ack_estimator_incoming(packet_list)
            result = self.gcc_rate_controller.delay_bwe_incoming(
                packet_list, self.gcc_ack_bitrate.ack_estimator_bitrate_bps(),
                now_ts)
            gcc_bitrate = result.bitrate
            self.gcc_bitrate = gcc_bitrate
            trendline = self.gcc_rate_controller.get_trendline_slope()
            if self.gcc_rate_controller.detector.state == 2:
                # logging.info("OVERUSING")
                self.gcc_state = self.GCCState.OVERUSING
            else:
                # logging.info("UNDERUSINg")
                self.gcc_state = self.GCCState.UNDERUSING
            if loss_ratio>0.1:
                self.gcc_state = self.GCCState.OVERUSING
            # if self.rate_control.state == kBwOverusing:
            #     self.gcc_state = self.GCCState.OVERUSING
            # else:
            #     self.gcc_state = self.GCCState.UNDERUSING


        
        

        # latest_prediction = self.packet_record.calculate_latest_prediction()

        delay = max(1, delay)
        delay = self.min_delay if self.min_delay*1.3>delay else delay
        delay_metric = self.min_delay / (delay * 0.5)

        if delay > self.min_delay * 1.5:
            self.start_up = False
            
        
        #-----------------append states---------------------------
        
        # print("loss rate:"+str(loss_ratio))
        states = []
        states.append(loss_ratio)
        states.append(trendline)
        states.append(self.latest_action/self.max_bandwidth)
        states.append(self.min_delay/delay)
        states.append(self.gcc_rate_controller.detector.state-1.0)
        
        


 
        reward_orca = 7*(receiving_rate - 0.7 * loss_ratio *
                       receiving_rate) / self.max_bandwidth + delay_metric
                    
                    #    receiving_rate) / self.max_bandwidth * delay_metric

        reward_bw = 3 * (receiving_rate / self.UNIT_M -
                           self.MIN_BANDWIDTH_MBPS) / self.MAX_BANDWIDTH_MBPS

        reward = reward_orca + reward_bw - 3 * self.hitwall
        self.latest_bandwidth = receiving_rate

        if self.logger_enabled:
            self.log_times.append(self.time_now)
            self.log_states.append(states)
            self.log_rewards.append(np.squeeze(reward))
            self.log_start.append(self.start_up)
            
        self.s1 = states
        self.r = reward_orca
        
        self.s1_rec_buffer = np.concatenate((self.s0_rec_buffer[self.single_s_dim:], self.s1) )
        
        self.s0_rec_buffer = self.s1_rec_buffer

        return not self.start_up, states, reward
    
    def get_estimated_bandwidth(self)->tuple:
        if self.start_up:
            logging.info("Start up!")
            a_rl = self.agent.get_action(self.s1_rec_buffer,False) #if i%200!=1 else agent.get_action(s0_rec_buffer,False);
            self.a_final = self.action(np.squeeze(a_rl))
            self.prev_sending_rate = self.a_final
            self.latest_action = self.a_final
            self.gcc_rate_controller.rate_control.curr_rate = self.a_final
        else:
            # print("state:{}".format(self.s1))
            # self.s1_rec_buffer = np.concatenate((self.s0_rec_buffer[self.single_s_dim:], self.s1) )
            a_rl = self.agent.get_action(self.s1_rec_buffer,False) #if i%200!=1 else agent.get_action(s0_rec_buffer,False);
            a_rl = np.squeeze(a_rl)
            self.a_final = self.action(a_rl)#按照rl得出的对应的cwnd的值
        # print("get_estimated_bandwidth:{}".format(self.a_final))
        return self.a_final,self.WLibraState.Ordinary #int(20000000) # 1Mbps

class LokiAgent(AgentWrapper):
    class GCCState(Enum):
        UNDERUSING = auto()
        OVERUSING = auto()
    class WLibraState(Enum):
        # Exploration = 0
        # Evaluation = 1
        # Exploitation = 2
        Ordinary = 0
    def __init__(self, time_interval, base_dir) -> None:
        self.STATE_DIM = 5
        self.UNIT_M = 1000000
        self.MAX_BANDWIDTH_MBPS = 2
        self.MIN_BANDWIDTH_MBPS = 0.7
        self.STEP_TIME = time_interval
        
        self.agent = LokiFusionAgent("./rl_script/loki/models/gcc_module_3.pt", "./rl_script/loki/models/rl_module_5.pt")
        self.obs = np.zeros(25) # features: lossratio, trendline, mindelay/delay, lastbw/maxbw, gcc_state
        self.delay = 10000
        self.min_delay = 10000
        self.latest_bw_estimated = 300000
        self.latest_bw = 300000
        self.max_bandwidth = self.latest_bw
        self.action_list = np.array([
            0.5, 0.72, 0.9, 0.99, 1, 1.02, 1.1, 1.28, 1.5, 2])
        self.start_up = True
        
        
        self.first_time = 0
        self.gcc_bitrate = self.latest_bw
        # Objects
        self.packet_record = PacketRecord()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.gcc_rate_controller = delay_base_bwe()
        self.gcc_rate_controller.set_time(self.first_time)
        self.gcc_rate_controller.set_start_bitrate(self.gcc_bitrate)
        
        # Loggers
        self.logger_enabled = False
        self.episode = 0
        self.mlogger = MLogger('Loki', base_dir)
        self.log_times = []
        self.log_rewards = []
        self.log_states = []
        self.log_actions = []
        self.log_start = []
        self.log = self.mlogger.logger
    def reset(self, logger_enabled: bool, episode: int) -> list:
        # Fields
        self.gcc_state = self.GCCState.UNDERUSING
        self.latest_bandwidth = 30000
        self.current_bandwidth = self.latest_bandwidth
        self.time_now = 0
        self.first_time = 0
        self.gcc_bitrate = self.latest_bandwidth
        self.max_bandwidth = self.latest_bandwidth
        self.min_delay = 999999
        self.start_up = True
        self.hitwall = False
        self.latest_action = self.latest_bandwidth
        # Objects
        self.packet_record.reset()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.gcc_rate_controller = delay_base_bwe()
        self.gcc_rate_controller.set_time(self.first_time)
        self.gcc_rate_controller.set_start_bitrate(self.gcc_bitrate)
        if self.logger_enabled:
            print("Draw Overview!")
            print("---------------------------")
            print(self.log_times)
            print(self.log_rewards)
            print(self.log_actions)
            print(self.log_start)
            print("---------------------------")
            print("---------------------------")
            print("---------------------------")

            m = MultiGraph('Overview')
            for i in range(self.STATE_DIM):
                m.new_graph(
                    Graph(x=self.log_times,
                          y=[s[i] for s in self.log_states],
                          title=f'state {i}'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_rewards, title='rewards'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_actions, title='actions'))
            m.new_graph(
                Graph(x=self.log_times, y=self.log_start, title='start up'))
            self.mlogger.new_multigraph(m)
            self.mlogger.export(episode)
        self.logger_enabled = logger_enabled
        
        self.episode = episode
        self.mlogger.reset()
        self.log_times = []
        self.log_rewards = []
        self.log_states = []
        self.log_actions = []
        self.log_start = []
        
        return [0.0 for _ in range(self.STATE_DIM)]
    def action(self, a: np.float64) -> np.float64:
        return self.latest_bw_estimated
    
    def get_estimated_bandwidth(self) -> int:
        if self.start_up:
            logging.info("start_up!")
            self.latest_bw = 2 * np.log(2) * self.latest_bw
            self.latest_bw = np.clip(self.latest_bw, 3e5, 6e6)
            self.latest_action = self.latest_bw
            self.log_actions.append(np.squeeze(self.latest_bw))
            
            return self.latest_bw,self.WLibraState.Ordinary
        print("loki:getting bandwidth:{}".format(self.latest_bw_estimated))
        self.latest_action = self.latest_bw_estimated
        self.log_actions.append(np.squeeze(self.latest_bw_estimated))

        return self.latest_bw_estimated,self.WLibraState.Ordinary #*0.6
    
    def update(self,
               packet_list: list) -> tuple:
        print("updating...")
        # ------------从数据包中提取信息---------------
        for pkt in packet_list:
            packet_info = PacketInfo()
            packet_info.payload_type = pkt["payload_type"]
            packet_info.ssrc = pkt["ssrc"]
            packet_info.sequence_number = pkt["sequence_number"]
            packet_info.send_timestamp = pkt["send_time_ms"]
            packet_info.receive_timestamp = pkt["arrival_time_ms"]
            packet_info.padding_length = pkt["padding_length"]
            packet_info.header_length = pkt["header_length"]
            packet_info.payload_size = pkt["payload_size"]
            self.time_now = pkt["time_now"]
            # print("updating...time:{}".format(self.time_now))

            self.packet_record.on_receive(packet_info)
            # print("pkt:"+str(pkt))
            
        #---------------计算RL的关键信息----------------------
        receiving_rate = self.packet_record.calculate_receiving_rate(
            interval=self.STEP_TIME)

        self.max_bandwidth = max(self.max_bandwidth, receiving_rate)

        delay = self.packet_record.calculate_average_delay(
            interval=self.STEP_TIME)
        self.min_delay = min(
            self.min_delay,
            self.packet_record.min_seen_delay)  # min(self.min_delay, delay)

        loss_ratio = self.packet_record.calculate_loss_ratio(
            interval=self.STEP_TIME)
        
        self.loss_rate = loss_ratio
        
        self.packet_record.clear()
        
        # --------------计算 GCC 速率-----------------
        gcc_bitrate = 0
        trendline = 0
        if len(packet_list) > 0:
            now_ts = self.time_now - self.first_time
            self.gcc_ack_bitrate.ack_estimator_incoming(packet_list)
            result = self.gcc_rate_controller.delay_bwe_incoming(
                packet_list, self.gcc_ack_bitrate.ack_estimator_bitrate_bps(),
                now_ts)
            gcc_bitrate = result.bitrate
            self.gcc_bitrate = gcc_bitrate
            trendline = self.gcc_rate_controller.get_trendline_slope() * min(self.gcc_rate_controller.trendline_estimator.num_of_deltas, 60)
            if self.gcc_rate_controller.detector.state == 2:
                # logging.info("OVERUSING")
                self.gcc_state = self.GCCState.OVERUSING
            else:
                # logging.info("UNDERUSINg")
                self.gcc_state = self.GCCState.UNDERUSING
            if loss_ratio>0.1:
                self.gcc_state = self.GCCState.OVERUSING
            

        delay = max(1, delay)
        delay = self.min_delay if self.min_delay*1.3>delay else delay
        delay_metric = self.min_delay / (delay * 0.5)
        print("delay:{},min:{}".format(delay,self.min_delay * 1.5))

        if delay > self.min_delay * 1.5:
            self.start_up = False
            
        
        #-----------------append states---------------------------
        
        # print("loss rate:"+str(loss_ratio))
        states = []
        states.append(loss_ratio)
        states.append(trendline)
        states.append(self.min_delay/delay)
        states.append(self.latest_action/self.max_bandwidth)
        states.append(self.gcc_rate_controller.detector.state)
        print("states:{}".format(states))
        # self.obs = np.roll(self.obs, -5)
        
        self.obs[-5:] = states
        # print("self.obs:{}".format(self.obs))
        
        # Loki 计算速率
        prediction, _ = self.agent.predict(self.obs)
        factor = self.action_list[prediction]
        print("prediction:{} ,action_list:{},factor:{}".format(prediction,self.action_list,factor))
        self.latest_bw_estimated = factor * self.latest_bw_estimated
        self.latest_bw_estimated = np.clip(self.latest_bw_estimated, 3e5, 2.5e6)
 
        reward = 7*(receiving_rate - 0.7 * loss_ratio *
                       receiving_rate) / self.max_bandwidth + delay_metric
                    

        
        self.latest_bandwidth = receiving_rate

            
        self.r = reward
        
        if self.logger_enabled:
            self.log_times.append(self.time_now)
            self.log_states.append(states)
            self.log_rewards.append(np.squeeze(reward))
            self.log_start.append(self.start_up)
        return not self.start_up, states, reward
        
class LibraAgent(AgentWrapper): 
    class GCCState(Enum):
        UNDERUSING = auto()
        OVERUSING = auto()
    class WLibraState(Enum):
    # Exploration = 0
    # Evaluation = 1
    # Exploitation = 2
        Ordinary = 0
        EI_c_1 = 1 #
        EI_r_1 = 2 #
        EI_c_2 = 3 #receive ack of the class action and generate the reward 
        EI_r_2 = 4 #receive ack of the rl action and generate the reward
        
    def __init__(self, time_interval, base_dir) -> None:
        # Constants
        self.STATE_DIM = 5
        self.UNIT_M = 1000000
        self.MAX_BANDWIDTH_MBPS = 6
        self.MIN_BANDWIDTH_MBPS = 0.1
        self.LOG_MAX_BANDWIDTH_MBPS = np.log(self.MAX_BANDWIDTH_MBPS)
        self.LOG_MIN_BANDWIDTH_MBPS = np.log(self.MIN_BANDWIDTH_MBPS)
        self.STEP_TIME = time_interval
        self.HITWALL_THRESHOLD = 2
        # Fields
        self.gcc_state = self.GCCState.UNDERUSING
        self.latest_bandwidth = 30000
        self.current_bandwidth = self.latest_bandwidth
        self.time_now = 0
        self.first_time = 0
        self.gcc_bitrate = self.latest_bandwidth
        self.max_bandwidth = self.latest_bandwidth
        self.min_delay = 999999
        self.start_up = True
        self.hitwall = False
        self.latest_action = self.latest_bandwidth
        self.receiving_rate = 1000000
        # Objects
        self.packet_record = PacketRecord()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.gcc_rate_controller = delay_base_bwe()
        self.gcc_rate_controller.set_time(self.first_time)
        self.gcc_rate_controller.set_start_bitrate(self.gcc_bitrate)
        # Loggers
        self.logger_enabled = False
        self.episode = 0
        self.mlogger = MLogger('Libra', base_dir)
        self.log_times = []
        self.log_rewards = []
        self.log_states = []
        self.log_actions = []
        self.log_start = []
        self.log = self.mlogger.logger
        
        #---------------Libra init--------------------
        self.w_state = self.WLibraState.Ordinary
        self.EI_sequence = 0
        self.u_1 = 0
        self.u_2 = 0
        self.u_3 = 0
        self.cwnd_list = []
        self.start_up = True
        self.delay = 10000
        self.min_delay = 10000
        self.max_bandwidth = self.MAX_BANDWIDTH_MBPS*self.UNIT_M
        self.hitwall = False#考虑删掉
        self.prev_sending_rate = 100000
        self.latest_action = self.latest_bandwidth
        self.a_final = 0
        self.x_r = 0
        self.x_c = 0
        
        #--------------RL init------------------------
        global params
        params = Params(os.path.join(os.path.dirname(__file__), 'deep_rl/params.json'))

        self.single_s_dim, a_dim = self.STATE_DIM, params.dict['action_dim']
        rec_s_dim = self.single_s_dim * params.dict['rec_dim']

        
        params.dict['train_dir'] = os.path.join(os.path.dirname(__file__), params.dict['logdir'])
        
        tf.set_random_seed(1234)
        random.seed(1234)
        np,random.seed(1234)
        tf.compat.v1.disable_eager_execution()
        summary_writer = tf.summary.FileWriterCache.get(params.dict['logdir'])
        self.agent = Agent(rec_s_dim, a_dim, batch_size=params.dict['batch_size'], summary=summary_writer,h1_shape=params.dict['h1_shape'],
                            h2_shape=params.dict['h2_shape'],stddev=params.dict['stddev'],mem_size=params.dict['memsize'],gamma=params.dict['gamma'],
                            lr_c=params.dict['lr_c'],lr_a=params.dict['lr_a'],tau=params.dict['tau'],PER=params.dict['PER'],CDQ=params.dict['CDQ'],
                            LOSS_TYPE=params.dict['LOSS_TYPE'],noise_type=params.dict['noise_type'],noise_exp=params.dict['noise_exp'])
        self.s1 = np.zeros([self.single_s_dim])
        self.s0_rec_buffer = np.zeros([rec_s_dim])
        self.s1_rec_buffer = np.zeros([rec_s_dim])
        # self.s0_rec_buffer[-1*single_s_dim:] = s0
        tfconfig = tf.ConfigProto(allow_soft_placement=True)

        self.sess = tf.Session(config=tfconfig)
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        saver = tf.train.Saver()
        self.agent.assign_sess(self.sess)

        # learner部分
        self.agent.init_target()

        # if not Train:
        # max_episodes = 1
        logging.info(os.path.join(params.dict['ckptdir']))
        
        params.dict['ckptdir'] = os.path.join(os.path.dirname(__file__), params.dict['ckptdir'])
        # model_name = 'model-0714_112251-2970' #在fixed/step 场景下训练的不错
        # model_name = 'model-0720_095709-21630' #在2970基础上，在波动很大的trace下也进行了训练
        # model_name = 'model-0721_164644-12570' 
        model_name = 'model-0129_221607-15300'
        
        
        saver.restore(self.sess, os.path.join(params.dict['ckptdir'], model_name))


    
    def action(self, a: np.float64) -> np.float64:
        if self.start_up:
            print("start_up!")
            act_raw = 2 * np.log(2) * self.latest_action
            # act_raw = 200000
        else:
            act_raw = np.power(1.5, a) * self.latest_action
            if act_raw / self.UNIT_M >= self.MAX_BANDWIDTH_MBPS or \
                    act_raw / self.UNIT_M <= self.MIN_BANDWIDTH_MBPS:
                self.hitwall = True
            else:
                self.hitwall = False
            # logging.info("a:"+str(a) + " latest_action:" + str(self.latest_action)+ " act_raw:"+str(act_raw))
            # logging.info("a:"+str(a)+" exp:"+str(exp) + " latest_action:" + str(self.latest_action)+ " act_raw:"+str(act_raw))
        act = np.clip(act_raw, self.MIN_BANDWIDTH_MBPS * self.UNIT_M,
                      self.MAX_BANDWIDTH_MBPS * self.UNIT_M)
        self.latest_action = act #act_raw-->act
        self.log_actions.append(np.squeeze(a))
        return act

    def reset(self, logger_enabled: bool, episode: int) -> list:
        # Fields
        self.gcc_state = self.GCCState.UNDERUSING
        self.latest_bandwidth = 30000
        self.current_bandwidth = self.latest_bandwidth
        self.time_now = 0
        self.first_time = 0
        # self.receive
        self.gcc_bitrate = self.latest_bandwidth
        self.max_bandwidth = self.latest_bandwidth
        self.min_delay = 999999
        self.start_up = True
        self.hitwall = False
        self.latest_action = self.latest_bandwidth
        # Objects
        self.packet_record.reset()
        self.gcc_ack_bitrate = Ack_bitrate_estimator()
        self.gcc_rate_controller = delay_base_bwe()
        self.gcc_rate_controller.set_time(self.first_time)
        self.gcc_rate_controller.set_start_bitrate(self.gcc_bitrate)
        # Loggers

        self.logger_enabled = logger_enabled
        self.episode = episode
        self.mlogger.reset()
        self.log_times = []
        self.log_rewards = []
        self.log_states = []
        self.log_actions = []
        self.log_start = []
        return [0.0 for _ in range(self.STATE_DIM)]

    def update(self,
               packet_list: list) -> tuple:
        # ------------从数据包中提取信息---------------
        for pkt in packet_list:
            packet_info = PacketInfo()
            packet_info.payload_type = pkt["payload_type"]
            packet_info.ssrc = pkt["ssrc"]
            packet_info.sequence_number = pkt["sequence_number"]
            packet_info.send_timestamp = pkt["send_time_ms"]
            packet_info.receive_timestamp = pkt["arrival_time_ms"]
            packet_info.padding_length = pkt["padding_length"]
            packet_info.header_length = pkt["header_length"]
            packet_info.payload_size = pkt["payload_size"]
            self.time_now = pkt["time_now"]
            self.packet_record.on_receive(packet_info)
        logging.info("time now:"+str(self.time_now))

        #---------------计算RL的关键信息----------------------
        self.receiving_rate = self.packet_record.calculate_receiving_rate(
            interval=self.STEP_TIME)

        self.max_bandwidth = max(self.max_bandwidth, self.receiving_rate)

        self.delay = self.packet_record.calculate_average_delay(
            interval=self.STEP_TIME)
        
        logging.info(" min delay:"+str(self.min_delay))
        self.min_delay = min(
            self.min_delay,
            self.packet_record.min_seen_delay)  # min(self.min_delay, delay)

        loss_ratio = self.packet_record.calculate_loss_ratio(
            interval=self.STEP_TIME)

        self.packet_record.clear()
        
        self.delay = max(1, self.delay)
        delay_metric = self.min_delay / (self.delay * 2)
        logging.info("delay:"+str(self.delay)+" min delay:"+str(self.min_delay))
        if self.start_up and (self.delay > self.min_delay * 4 or self.a_final>1500000):
            logging.info("-------------------------START UP END!---------------------------")
            self.start_up = False

        # --------------计算 GCC 速率-------------------
        gcc_bitrate = 0
        trendline = 0
        logging.info("packet_list:"+str(len(packet_list)))
        logging.info("self.w_state:"+str(self.w_state)+" "+str(self.WLibraState.Ordinary))
        # logging.info("self.w_state==WLibraState.Ordinary:"+str(self.w_state == self.WLibraState.Ordinary))

        if len(packet_list) > 0 and self.w_state is self.WLibraState.Ordinary:
            logging.info("运行GCC计算码率")
            now_ts = self.time_now - self.first_time
            self.gcc_ack_bitrate.ack_estimator_incoming(packet_list)
            result = self.gcc_rate_controller.delay_bwe_incoming(
                # packet_list, self.gcc_ack_bitrate.ack_estimator_bitrate_bps(),
                packet_list, self.receiving_rate,
                now_ts)
            gcc_bitrate = result.bitrate
            self.gcc_bitrate = gcc_bitrate
            trendline = self.gcc_rate_controller.get_trendline_slope()
            logging.info("GCC state:"+str(self.gcc_rate_controller.detector.state))
            logging.info("GCC bitrate:"+str(self.gcc_bitrate))
            if self.gcc_rate_controller.detector.state == 2:
                self.gcc_state = self.GCCState.OVERUSING
            else:
                self.gcc_state = self.GCCState.UNDERUSING
            # if self.rate_control.state == kBwOverusing:
            #     self.gcc_state = self.GCCState.OVERUSING
            # else:
            #     self.gcc_state = self.GCCState.UNDERUSING


        #-----------------append states---------------------------
        states = []
        states.append(loss_ratio)
        states.append(trendline)
        states.append(self.latest_action/self.max_bandwidth)
        states.append(self.min_delay/self.delay)
        states.append(self.gcc_rate_controller.detector.state-1.0)

        #--------------------计算reward---------------------------

        

        t = 0.9 
        alpha = 1 
        beta = 0.5
        gamma = 5

        # send_rate = self.prev_sending_rate
        # if self.w_state == self.WLibraState.EI_c_2:
        #     send_rate = self.x_c
        # if self.w_state == self.WLibraState.EI_r_2:
        #     send_rate = self.x_r
        # reward_libra = alpha*pow(send_rate,t) - beta*send_rate*max(0,delay_metric) - gamma*send_rate*loss_ratio

        reward_orca = (self.receiving_rate - 10 * loss_ratio *
                       self.receiving_rate) / self.max_bandwidth + 10*delay_metric # delay_metric
        # reward_delay = -delay
        logging.info("recv rate:"+str(self.receiving_rate)+" loss rate:"+str(loss_ratio)+" delay_metric:"+str(delay_metric))

        self.latest_bandwidth = self.receiving_rate

        if self.logger_enabled:
            self.log_times.append(self.time_now)
            self.log_states.append(states)
            self.log_rewards.append(np.squeeze(reward_orca))
            self.log_start.append(self.start_up)

        self.s1 = states
        self.r = reward_orca
        
        self.s1_rec_buffer = np.concatenate((self.s0_rec_buffer[self.single_s_dim:], self.s1) )
        
        self.s0_rec_buffer = self.s1_rec_buffer
        return self.start_up, states, reward_orca
    
    def get_estimated_bandwidth(self)->tuple:

        # print("matthew: Py is setting bandwidth!")
        '''
        Libra的逻辑在这里完成
        '''

        
        # print("self.delay:"+str(self.delay)+" a_final:"+str(self.a_final)+" ack_rate:"+str(self.gcc_bitrate))
        
        if self.start_up:
            logging.info("Start up!")
            a_rl = self.agent.get_action(self.s1_rec_buffer,False) #if i%200!=1 else agent.get_action(s0_rec_buffer,False);
            self.a_final = self.action(np.squeeze(a_rl))
            self.prev_sending_rate = self.a_final
            self.latest_action = self.a_final
            self.gcc_rate_controller.rate_control.curr_rate = self.a_final
        #增加一个排干逻辑
        elif self.delay> self.min_delay*2 and self.a_final>self.receiving_rate:
            self.a_final = self.receiving_rate*0.9
            self.w_state = self.WLibraState.Ordinary

        elif self.w_state == self.WLibraState.Ordinary:
            self.cwnd_list.clear()
            self.s1_rec_buffer = np.concatenate((self.s0_rec_buffer[self.single_s_dim:], self.s1) )
            a_rl = self.agent.get_action(self.s1_rec_buffer,False) #if i%200!=1 else agent.get_action(s0_rec_buffer,False);
            a_rl = np.squeeze(a_rl)
            self.x_r = self.action(a_rl)#按照rl得出的对应的cwnd的值
            # x_r = 1800000
            self.x_c = self.gcc_bitrate if self.gcc_bitrate>0 else 300000 #得到cubic得到的action  它就是新cwnd的值
            logging.info("Ordinary stage derives:prev_r:"+str(round(self.prev_sending_rate/1000000,2))+"mbps x_r:"+str(round(self.x_r/1000000,2))+"mbps x_c:"+str(round(self.x_c/1000000,2))+"mbps")
            if np.abs(self.x_r-self.x_c)>=0.01*self.prev_sending_rate:#prev_cwnd not defined yet
                #进入evaluation stage
                    
                #进入EI,注意：进入EI和进入evaluation stage是不一样的
                if self.x_r>self.x_c:
                    # classic_is_last = False
                    self.cwnd_list.append(self.x_c)
                    self.cwnd_list.append(self.x_r)
                    self.EI_sequence=1
                    self.a_final=self.cwnd_list[0]
                    self.w_state=self.WLibraState.EI_c_1
                else:
                    # classic_is_last = True
                    self.cwnd_list.append(self.x_r)
                    self.cwnd_list.append(self.x_c)
                    self.EI_sequence=0
                    self.a_final=self.cwnd_list[0]
                    self.w_state=self.WLibraState.EI_r_1
                    
            else:
                self.a_final=self.x_c
            # logging.info("-----The Ordinary stage end-----")

        elif self.w_state == self.WLibraState.EI_c_1:
                
            if self.EI_sequence==1:#cl先行 下一个状态是EI_r_1
                self.w_state=self.WLibraState.EI_r_1
                self.a_final=self.cwnd_list[1]
            else:         
                self.u_1=self.r#得到u1但是后续需要除episode的长度 
                logging.info("u_1[prev sending rate]:"+str(self.u_1))
                self.w_state=self.WLibraState.EI_r_2
                self.a_final=self.prev_sending_rate
                # logging.info("-----stage EI_c_1 end-----")

        elif self.w_state==self.WLibraState.EI_r_1:
            if self.EI_sequence==0:#rl先行 下一个状态是EI_c_1
                self.w_state=self.WLibraState.EI_c_1
                self.a_final=self.cwnd_list[1]
            else:  
                self.u_1=self.r#得到u1但是后续需要除episode的长度
                logging.info("u_1:"+str(self.u_1))
                self.w_state=self.WLibraState.EI_c_2
                self.a_final=self.prev_sending_rate
                # logging.info("-----stage:EI_r_1 end-----")

        elif self.w_state==self.WLibraState.EI_c_2:
            if self.EI_sequence==1:#cl先行 下一个动作是EI_r_2
                self.u_2=self.r#得到u2但是后续需要除episode的长度
                logging.info("u_2:"+str(self.u_2))
                self.w_state=self.WLibraState.EI_r_2
                            # a_final=a_r
            else:
                self.w_state=self.WLibraState.Ordinary#Evaluation结束进入Ordinary
                self.u_3=self.r#/rtt
                logging.info("u_1:"+str(self.u_1)+"  u_2:"+str(self.u_2)+" u_3:"+str(self.u_3))
                    # logging.info("********End of a Libra Control Loop*********")
                if max(self.u_2,self.u_3)>=self.u_1:
                    if(self.u_2>self.u_3):#EI——sequence=1 rl先行
                        self.a_final=self.cwnd_list[0]
                        # strRLCWND_record.strip()
                        # strRLCWND_record+=" 1"
                        self.prev_sending_rate=self.a_final
                                    # rl_count+=1
                    else:
                        self.a_final=self.cwnd_list[1]
                            # logging.info(strCLCWND_record)
                                    
                                    # logging.info(strCLCWND_record)
                        self.prev_sending_rate=self.a_final
                                    # cl_count+=1
                else:
                    self.a_final=self.prev_sending_rate
                    # if(self.u_2>u_3):#EI——sequence=1 rl先行
                    #     self.a_final=self.cwnd_list[0]
                    #         # strRLCWND_record.strip()
                    #         # strRLCWND_record+=" 1"
                    #     self.prev_sending_rate=self.a_final
                    #                 # rl_count+=1
                    # else:
                    #     self.a_final=self.cwnd_list[1]
                            # logging.info(strCLCWND_record)
                                    
                                    # logging.info(strCLCWND_record)
                    self.prev_sending_rate=self.a_final
                                    # cl_count+=1
                            # strCWND_record+=(str(int((current_time-start)*1000000)/1000000)+" "+str(self.a_final)+"\n")
                    #在libra中，给gcc和rl设定新的base sending rate
                self.latest_action = self.a_final
                self.gcc_rate_controller.rate_control.curr_rate = self.a_final
                # logging.info("-----stage:EI_c_2 end-----")
        elif self.w_state==self.WLibraState.EI_r_2:
            if self.EI_sequence==0:#rl先行 下一个动作是EI_c_2
                self.u_2=self.r
                self.w_state=self.WLibraState.EI_c_2
                            # self.a_final=a_r
            else:
                self.w_state=self.WLibraState.Ordinary#Evaluation结束进入Ordinary
                self.u_3=self.r
                logging.info("u_1:"+str(self.u_1)+"  u_2:"+str(self.u_2)+" u_3:"+str(self.u_3))
                # logging.info("********End of a Libra Control Loop*********")
                if max(self.u_2,self.u_3)>=self.u_1:
                    if(self.u_2>self.u_3):#EI——sequence=0 cl先行
                        self.a_final=self.cwnd_list[0]
                                    # strRLCWND_record.strip()
                                    # strCLCWND_record+=" 1"
                        self.prev_sending_rate=self.a_final
                                    # cl_count+=1
                    else:
                        self.a_final=self.cwnd_list[1]
                                    # strRLCWND_record.strip()
                                    # strRLCWND_record+=" 1"
                        self.prev_sending_rate=self.a_final
                                    # rl_count+=1
                else:
                    self.a_final=self.prev_sending_rate
                    # if(self.u_2>self.u_3):#EI——sequence=0 cl先行
                    #     self.a_final=self.cwnd_list[0]
                    #                 # strRLCWND_record.strip()
                    #                 # strCLCWND_record+=" 1"
                    #     self.prev_sending_rate=self.a_final
                    #                 # cl_count+=1
                    # else:
                    #     self.a_final=self.cwnd_list[1]
                    #                 # strRLCWND_record.strip()
                    #                 # strRLCWND_record+=" 1"
                    #     self.prev_sending_rate=self.a_final
                                    # rl_count+=1
                    
                    #在libra中，给gcc和rl设定新的base sending rate
        self.latest_action = self.a_final
        self.gcc_rate_controller.rate_control.curr_rate = self.a_final

        logging.info("final action:"+str(self.a_final))
        return self.a_final,self.w_state#int(20000000) # 1Mbps
