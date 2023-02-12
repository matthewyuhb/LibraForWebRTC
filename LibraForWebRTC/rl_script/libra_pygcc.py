import os
from enum import Enum
import numpy as np
import random
import glob
from gym import Env
import logging
from deep_rl.agent import Agent
from deep_rl.utils import Params
from rtc_env import GymEnv
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optparse import OptionParser
from draw import draw_metric
from aw import FusionAgent, GCCAgent, OrcaAgent,LibraAgent,OnrlAgent,LokiAgent




def run_libra(is_train_mode,report_interval_ms,train_mode):
    # --------------------------------------------------
    # -------------------  Logger  ---------------------
    # --------------------------------------------------

    # - rl_script
    #   - performance_records
    #     - 
    # tf.get_logger().setLevel(logging.WARNING)
    logger = logging.getLogger()
    # logger.setLevel(logging.WARNING)
    logger.setLevel(logging.INFO)

    pr_base = os.path.join(os.path.abspath('.'), 'rl_script/performance_records/')
    if not os.path.exists(pr_base):
        os.mkdir(pr_base)

    start_time = time.localtime()
    log_dir_base = os.path.join(pr_base, time.strftime("%m%d_%H%M%S", start_time))
    if not os.path.exists(log_dir_base):
        os.mkdir(log_dir_base)
    
    log_dir_main = os.path.join(log_dir_base, 'main')
    if not os.path.exists(log_dir_main):
        os.mkdir(log_dir_main)
    
    # log_dir_env = os.path.join(log_dir_base, 'env')
    # if not os.path.exists(log_dir_env):
    #     os.mkdir(log_dir_env)

    # log_dir_rl = os.path.join(log_dir_base, 'RL')
    # if not os.path.exists(log_dir_rl):
    #     os.makedirs(log_dir_rl)


    # --------------------------------------------------
    # -------------------  Params  ---------------------
    # --------------------------------------------------
    

    # model_name = 'model-0714_112251-2970' #在fixed/step 场景下训练的不错rl_script/ckpt_dir/model-0720_095709-21630.index

    

    global params
    params = Params(
        os.path.join(os.path.dirname(__file__), 'deep_rl/params.json'))

    # s_dim, a_dim = params.dict['state_dim'], params.dict['action_dim']
    # s_dim *= params.dict['rec_dim']
    # aw = GCCAgent(log_dir_base)
    if train_mode == 'loki':
        aw = LokiAgent(report_interval_ms, log_dir_base)
    elif train_mode == 'onrl':
        aw = OnrlAgent(report_interval_ms, log_dir_base)
    else:
        aw = LibraAgent(report_interval_ms, log_dir_base)
        
    # aw = OnrlAgent(report_interval_ms, log_dir_base)
    # aw = LibraAgent(report_interval_ms, log_dir_base)

    single_s_dim, a_dim = aw.get_state_dim(), params.dict['action_dim']
    rec_s_dim = single_s_dim * params.dict['rec_dim']
    start_up = True
    # with tf.Graph().as_default(), tf.device('/cpu:0'):


    params.dict['train_dir'] = os.path.join(os.path.dirname(__file__), params.dict['logdir'])
    # print("params.dict['logdir']:"+str(params.dict['logdir']))
    # return 
    
    
    # 通用部分
    
    env = GymEnv(aw,log_dir_base, report_interval_ms,train_mode,True)

    
    # reward_record = []#/home/admin/LibraForWebRTC/rl_script/traces/bw501.jpg
    # w_state = env.agent.WLibraState.Ordinary
    
    trace_dir = os.path.join(os.path.dirname(__file__), "traces/varied_trace3")
    trace_set = glob.glob(f'{trace_dir}/**/*.json', recursive=True)



    logging.info("Epiosde: %d" % 0)
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
        draw(trace_path, log_dir_main, i,alg = train_mode)#5G_12mbps
        


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
    