import os
from enum import Enum

import numpy as np
import random
import glob
import logging
from deep_rl.utils import Params
from rtc_env import GymEnv
import time
import matplotlib

from collections import deque


matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optparse import OptionParser

from draw import draw_metric

from aw import SPQL_FusionAgent, GCCAgent, OrcaAgent

from spql.spql_agent import Agent

class WLibraState(Enum):
    Exploration = 0
    Evaluation = 1
    Exploitation = 2

def test(agent,env,i,trace_path,log_interval):
    logging.warn("testing...")

    logger_enabled = True if i % log_interval == 1 else False
    state = env.reset(trace_path, i, logger_enabled=logger_enabled)

    totol_reward = 0
    terminal = False
    w_state = WLibraState.Exploration

    while not terminal:# and time_step < update_interval:
        action = agent.get_test_action(state)
        # action = np.squeeze(action)
        # logging.info("action:"+str(action))
        is_valid, next_state, reward, terminal, _ = env.step(np.squeeze(action), w_state)
        state = next_state
        totol_reward += reward

    return totol_reward


def train(is_train_mode, report_interval_ms, train_mode):
    # --------------------------------------------------
    # -------------------  Logger  ---------------------
    # --------------------------------------------------

    # - rl_script
    #   - performance_records
    #     -
    logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    logger.setLevel(logging.WARNING)

    pr_base = os.path.join(os.path.abspath('.'),
                           'rl_script/performance_records/')
    if not os.path.exists(pr_base):
        os.mkdir(pr_base)

    start_time = time.localtime()
    log_dir_base = os.path.join(pr_base,
                                time.strftime("%m%d_%H%M%S", start_time))
    os.makedirs(log_dir_base, exist_ok=True)
    log_dir_main = os.path.join(log_dir_base, 'main')
    os.makedirs(log_dir_main, exist_ok=True)

    log_dir_env = os.path.join(log_dir_base, 'env')
    os.makedirs(log_dir_env, exist_ok=True)

    log_dir_rl = os.path.join(log_dir_base, 'RL')
    os.makedirs(log_dir_rl, exist_ok=True)
    # --------------------------------------------------
    # -------------------  Params  ---------------------
    # --------------------------------------------------

    max_episodes = 1600
    change_episode = max_episodes / 2
    # update_interval = 900000
    save_interval = 5
    log_interval = 5
    reward_ewma_coef = 0.9
    ewma_reward_record = []
    ewma_r = 0
    
    Train = is_train_mode
    # model_name = 'model-0714_112251-2970' #在fixed/step 场景下训练的不错

    global params
    params = Params(
        os.path.join(os.path.dirname(__file__), 'deep_rl/params.json'))

    # aw = GCCAgent(log_dir_base)
    # aw = OrcaAgent(report_interval_ms, log_dir_base)
    aw = SPQL_FusionAgent(report_interval_ms, log_dir_base)

    single_s_dim, a_dim = aw.get_state_dim(), params.dict['action_dim']
    # --------------------------------------------------
    # -------------------  Training  -------------------
    # --------------------------------------------------


    params.dict['train_dir'] = os.path.join(os.path.dirname(__file__),
                                            params.dict['logdir'])
    
    agent = Agent(single_s_dim, a_dim, \
            np.array([[-15.0, 15.0], [0.0, 10.0], [0.0, 1.0]]), \
            np.array([[-1.0, 1.0]]),\
            max_node_num=50, gamma=0.99, threshold_num=300, increase_factor=1.2, alpha=1, degree=2,
            rounds_per_backward=0, fit_ratio_threshold=0.3, ridge_alpha=1, max_sample_num_for_node=1600,
            is_sample_weighted=True, discounted_factor=0.99, epsilon=0.4, ep_decrease_factor=0.00001, split_ratio_limit=4, lowest_ep=0.05)

    # agent设置
    env = GymEnv(aw, log_dir_base, report_interval_ms, train_mode)
    reward_queue = deque(maxlen=200)
    nodes = []
    last_num = 1
    x_num = []
    record = []
    plot_record = []
    
    test_record = {}


    # agent初始化
    # agent.init_target()
    if not Train:
        max_episodes = 1
        
        logging.debug(os.path.join(params.dict['ckptdir']))
    # saver.restore(sess, os.path.join(params.dict['ckptdir'],
    #                                     model_name))

    ep_r = 0.0
    reward_record = []  # /home/admin/LibraForWebRTC/rl_script/traces/bw501.jpg
    w_state = WLibraState.Exploration

    
    trace_dir = os.path.join(os.path.dirname(__file__), "traces/fixed_cap_test")
    trace_set1 = glob.glob(f'{trace_dir}/**/*.json', recursive=True)

    trace_dir = os.path.join(os.path.dirname(__file__), "traces/varied_cap")
    trace_set2 = glob.glob(f'{trace_dir}/**/*.json', recursive=True)

    logging.warn("len of trace_set1:" + str(len(trace_set1)))
    logging.warn("len of trace_set2:" + str(len(trace_set2)))
    for i in range(1, max_episodes + 1):
        logging.warn("Epiosde: %d" % i)

        trace_path = random.choice(trace_set1)
        if i>change_episode :
            trace_path = random.choice(trace_set2)

        # trace_path = 'rl_script/traces/fixed_cap/track_step2.json'#/home/parallels/Desktop/LibraForWebRTC/rl_script/traces/varied_cap/4G_500k_expand_2.json
        # trace_path = 'rl_script/traces/fixed_cap/trace_2000k.json'
        # trace_path = 'rl_script/traces/WIRED_900kbs.json'
        logger_enabled = True if i % log_interval == 1 else False
        state = env.reset(trace_path, i, logger_enabled=logger_enabled,duration_time_ms_= 30000,loss_rate_= 0.05 if i > change_episode else 0.0)
        # print("s0:"+str(s0))
        simu_start = time.time()


        terminal = False
        time_step = 0
        ep_r = 0.0
        while not terminal:#and time_step < update_interval:
            # logging.warn(time_step)
            

            action = agent.get_action(state)
            # action = np.squeeze(action)
            # logging.warn("action:"+str(action))
            is_valid, next_state, reward, terminal, _ = env.step(np.squeeze(action), w_state)
            # logging.warn("reward:"+str(reward))
            reward = reward * -1
            # loss_in_one_ep += reward
            reward_queue.append(float(reward)*-1)
            totol = np.sum(reward_queue)

            # print("s1:"+str(s1))
            if not is_valid:
                # print("continue")
                continue
            ep_r += reward

            # pt = agent.rp_buffer.ptr
            # logging.warn("state:"+str(state)+"a:"+str(action)+" next:"+str(next_state))
            agent.get_record(state, action, next_state, reward, episode_index=i, done=terminal)
            state = next_state
            time_step += 1

        ep_r /= time_step
        ewma_r = reward_ewma_coef*ewma_r + (1-reward_ewma_coef)*ep_r
        # 
        print("current episode:{},reward:{}".format(i,ep_r))
        reward_record.append(ep_r)
        ewma_reward_record.append(ewma_r)

        test_record[i] = -1 * ep_r * time_step#avg_reward_per_test_episode 
        print("test_record:{}".format(test_record))
        
        
        logging.warn(
            f'Simulation ends in {time.time()-simu_start} for episode {i}')

        #-------------------------------------------------
        #-------------------- DRAW -----------------------
        #-------------------------------------------------

        #画出带宽-时延对比图
        if logger_enabled:
            logging.info("drawing")
            draw(trace_path, log_dir_main, i,alg=train_mode)  # 5G_12mbps

        

            #perform test
            # avg_reward_per_test_episode = test(agent,env,i,trace_path,log_interval)
            

        
        #画出reward变化图
        if i % save_interval == 0:
            plt.clf()
            f=open(os.path.join(params.dict['train_dir'], 
                'reward-{}.txt'.format(
                    time.strftime("%m%d_%H%M%S", start_time))),"w")
            for line in reward_record:      
                f.write(str(line)+'\n')
            f.close()
            plt.plot(range(len(reward_record)), reward_record,alpha = 0.2)
            plt.plot(range(len(ewma_reward_record)), ewma_reward_record)
            plt.xlabel('Episode')
            plt.ylabel('Averaged episode reward')
            plt.savefig(
                os.path.join(params.dict['train_dir'], 
                'reward-{}.jpg'.format(
                    time.strftime("%m%d_%H%M%S", start_time))))

        
            
        if logger_enabled:
            x_num.append(i)
            record.append(totol)
            if len(plot_record) == 0:
                plot_record.append(totol)
            else:
                plot_record.append(plot_record[-1] * 0.9 + totol * 0.1)
            if len(agent.node_list) > last_num:
                last_num = len(agent.node_list)
                nodes.append(500)
            else:
                nodes.append(0)
        logging.warn("Epiosde: %d END!" % max_episodes)
        
        

    
    with open('rl_script/performance_records/res_data/spql_{}.txt'.format(time.strftime("%m%d_%H%M%S", start_time)), 'w') as f:
        for p in test_record.keys():
            f.writelines("{} {}\n".format(p, test_record[p]))

    
        


def draw(trace_path, log_dir_base, episode_num,alg):
    
    draw_metric(os.path.abspath('.'), 'bw', log_dir_base, episode_num,
                trace_path,alg=alg)
    draw_metric(os.path.abspath('.'), 'rtt', log_dir_base, episode_num,trace_path,alg=alg)


if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option("-d",
                      "--test_or_train",
                      dest="Test_Or_Train",
                      default=True,
                      type='int',
                      help="#0: test 1:train")
    parser.add_option("-m",
        "--report_interval_ms",
        dest="report_interval_ms",
        default=0,
        help="#if the report_interval_ms is set to 0,\
        the monitor interval is one RTT")
    parser.add_option("-t",
                      "--train_mode",
                      dest="train_mode",
                      default=0,
                      help="1:enable gcc like orca 0:clean-slate")

    (options, args) = parser.parse_args()
    # duration=int(options.Duration)
    test_or_train = bool(options.Test_Or_Train)
    report_interval_ms = int(options.report_interval_ms)
    train_mode = str(options.train_mode).rstrip()
    train(test_or_train, report_interval_ms, train_mode)
    
