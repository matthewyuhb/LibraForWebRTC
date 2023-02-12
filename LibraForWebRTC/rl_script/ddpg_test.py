import argparse
import os
import pprint
import matplotlib.pyplot as plt
import gym
import numpy as np
import random
import torch
import glob

from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import DDPGPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.exploration import GaussianNoise
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import Actor, Critic

from draw import draw_metric

from rtc_env_standard import GymEnv
from aw import FusionAgent
import time

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='RTC')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--exploration-noise', type=float, default=0.1)
    parser.add_argument('--policy-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--update-actor-freq', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--step-per-epoch', type=int, default=240000)
    parser.add_argument('--step-per-collect', type=int, default=8)

    parser.add_argument('--update-per-step', type=float, default=0.125)

    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    args = parser.parse_known_args()[0]
    return args

def run_ddpg():
    

    pr_base = os.path.join(os.path.abspath('.'),
                           'rl_script/performance_records/')
    if not os.path.exists(pr_base):
        os.mkdir(pr_base)

    start_time = time.localtime()
    start_time_str = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))

    log_dir_base = os.path.join(pr_base,
                                time.strftime("%m%d_%H%M%S", start_time))
        
    aw = FusionAgent(0, log_dir_base)
    env = GymEnv(aw, log_dir_base, 0, 'onrl',port_num = 5564)

    
    # reward_record = []#/home/admin/LibraForWebRTC/rl_script/traces/bw501.jpg
    # w_state = env.agent.WLibraState.Ordinary
    
    trace_dir = os.path.join(os.path.dirname(__file__), "traces/varied_trace3")
    trace_set = glob.glob(f'{trace_dir}/**/*.json', recursive=True)



    trace_path=random.choice(trace_set)
    # trace_path = 'rl_script/traces/varied_cap/avg1500k_std200_duration1000ms_30secs.json'#/home/parallels/Desktop/LibraForWebRTC/rl_script/traces/varied_cap/4G_500k_expand_2.json
    # trace_path = 'rl_script/traces/varied_cap/4G_500k_expand_2.json'#/home/parallels/Desktop/LibraForWebRTC/rl_script/traces/varied_cap/4G_500k_expand_2.json
    # trace_path = 'rl_script/traces/fixed_cap/track_step2.json'#/home/parallels/Desktop/LibraForWebRTC/rl_script/traces/varied_cap/4G_500k_expand_2.json
    # logger_enabled = True if i % log_interval == 1 else False
    # trace_set = []
    # trace_set.append('rl_script/traces/fixed_cap/trace_2000k.json')
    
    for i in range(0,len(trace_set)):
        trace_path = trace_set[i]#'rl_script/traces/varied_cap/4G_500k_expand_2.json'#/home/parallels/Desktop/LibraForWebRTC/rl_script/traces/varied_cap/4G_500k_expand_2.json
        
        env.reset(trace_path,i,logger_enabled=True)
        terminal = False
        a_final = 0.0

        while not terminal:
            print("looping...")
            a_final,w_state = aw.get_estimated_bandwidth()
            start_up,s1, r, terminal, _ = env.step(a_final,w_state)        
        #draw trace
        print("train_mode:{}".format(train_mode))
        # draw(trace_path, log_dir_main, i,alg = train_mode)#5G_12mbps
        


def draw(trace_path, log_dir_base, episode_num,alg):
    
    # draw_metric(os.path.abspath('.'), 'bw', log_dir_base, episode_num, trace_path)
    # draw_metric(os.path.abspath('.'), 'rtt', log_dir_base, episode_num)
    plt.figure(figsize=(6,9))
    plt.subplot(211)
    
    draw_metric(os.path.abspath('.'),'bw',log_dir_base,episode_num,trace_path,alg=alg)
    plt.subplot(212)
    
    draw_metric(os.path.abspath('.'),'rtt',log_dir_base,episode_num,trace_path,alg=alg)
    save_path = os.path.join(log_dir_base, str(trace_path.split('/')[-1].split('.')[0])+'_merge.jpg')
    print("save path:{},trace_path:{}".format(save_path,trace_path.split('/')[-1].split('.')[0]))
    plt.savefig(save_path)
    plt.close()

if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option("-d", "--test_or_train", dest="Test_Or_Train", default=False,type='int',
                      help="#0: test 1:train")
    parser.add_option("-m", "--report_interval_ms", dest="report_interval_ms", default=400,
                      help="#if the report_interval_ms is set to 0, the monitor interval is one RTT")
    parser.add_option("-t", "--train_mode", dest="train_mode", default='gcc',
                      help="1:enable gcc like orca 0:clean-slate")
    
    (options, args) = parser.parse_args()
    # duration=int(options.Duration)
    test_or_train = bool(options.Test_Or_Train)
    report_interval_ms = int(options.report_interval_ms)
    train_mode = str(options.train_mode).rstrip()
    run_libra(test_or_train,report_interval_ms,train_mode)
    