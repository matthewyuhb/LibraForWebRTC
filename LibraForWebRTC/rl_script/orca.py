import os
from enum import Enum
import tensorflow.compat.v1 as tf
import numpy as np
import random
import glob
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

from aw import FusionAgent, GCCAgent, OrcaAgent


class WLibraState(Enum):
    Exploration = 0
    Evaluation = 1
    Exploitation = 2





def train(is_train_mode, report_interval_ms, train_mode):
    # --------------------------------------------------
    # -------------------  Logger  ---------------------
    # --------------------------------------------------

    # - rl_script
    #   - performance_records
    #     -
    tf.get_logger().setLevel(logging.WARNING)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # logger.setLevel(logging.WARNING)

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

    max_episodes = 200000
    update_interval = 900000
    save_interval = 100
    log_interval = 30
    reward_ewma_coef = 0.9
    ewma_reward_record = []
    ewma_r = 0
    
    Train = is_train_mode
    # model_name = 'model-0610_164220-2700'
    # model_name = 'model-0617_154251-2500'
    # model_name = 'model-0711_142526-8000'
    # model_name = 'model-0714_112251-2970' #在fixed/step 场景下训练的不错
    model_name = 'model-0721_164644-12570'

    global params
    params = Params(
        os.path.join(os.path.dirname(__file__), 'deep_rl/params.json'))

    # aw = GCCAgent(log_dir_base)
    # aw = OrcaAgent(report_interval_ms, log_dir_base)
    aw = FusionAgent(report_interval_ms, log_dir_base)

    single_s_dim, a_dim = aw.get_state_dim(), params.dict['action_dim']
    rec_s_dim = single_s_dim * params.dict['rec_dim']

    # --------------------------------------------------
    # -------------------  Training  -------------------
    # --------------------------------------------------
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        tf.set_random_seed(1234)
        random.seed(1234)
        np.random.seed(1234)

        params.dict['train_dir'] = os.path.join(os.path.dirname(__file__),
                                                params.dict['logdir'])
        summary_writer = tf.summary.FileWriterCache.get(params.dict['logdir'])
        agent = Agent(rec_s_dim,
                      a_dim,
                      batch_size=params.dict['batch_size'],
                      summary=summary_writer,
                      h1_shape=params.dict['h1_shape'],
                      h2_shape=params.dict['h2_shape'],
                      stddev=params.dict['stddev'],
                      mem_size=params.dict['memsize'],
                      gamma=params.dict['gamma'],
                      lr_c=params.dict['lr_c'],
                      lr_a=params.dict['lr_a'],
                      tau=params.dict['tau'],
                      PER=params.dict['PER'],
                      CDQ=params.dict['CDQ'],
                      LOSS_TYPE=params.dict['LOSS_TYPE'],
                      noise_type=params.dict['noise_type'],
                      noise_exp=params.dict['noise_exp'])

        # agent设置
        agent.build_learn()
        agent.create_tf_summary()
        env = GymEnv(aw, log_dir_base, report_interval_ms, train_mode,port_num=5564)

        if params.dict['ckptdir'] is not None:
            params.dict['ckptdir'] = os.path.join(os.path.dirname(__file__),
                                                  params.dict['ckptdir'])
            # logging.info("## checkpoint dir:", params.dict['ckptdir'])
            isckpt = os.path.isfile(
                os.path.join(params.dict['ckptdir'], 'checkpoint'))
            # logging.info("## checkpoint exists?:", isckpt)
            if isckpt is False:
                logging.warning(
                    "\n# # # # # # Warning ! ! ! No checkpoint is loaded, \
                    use random model! ! ! # # # # # #\n")
        else:
            params.dict['ckptdir'] = params.dict['train_dir']
        tfconfig = tf.ConfigProto(allow_soft_placement=True)

        sess = tf.Session(config=tfconfig)
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver = tf.train.Saver()
        agent.assign_sess(sess)

        # agent初始化
        agent.init_target()
        
        # saver.restore(sess, os.path.join(params.dict['ckptdir'],
        #                                     model_name))

        ep_r = 0.0
        reward_record = [
        ]  # /home/admin/LibraForWebRTC/rl_script/traces/bw501.jpg
        w_state = WLibraState.Exploration

        # trace_dir = os.path.join(os.path.dirname(__file__), "traces/varied_cap")
        trace_dir = os.path.join(os.path.dirname(__file__), "traces/fixed_cap_test")
        
        trace_set = glob.glob(f'{trace_dir}/**/*.json', recursive=True)
        logging.info("trace_set:" + str(len(trace_set)))
        
        
        if not Train:
            max_episodes = len(trace_set)
            
            logging.debug(os.path.join(params.dict['ckptdir']))
        
        for i in range(1, max_episodes + 1):
            print("Epiosde: %d" % i)
            trace_path = random.choice(trace_set)
            if not Train:
                trace_path = trace_set[i-1]
            # trace_path = 'rl_script/traces/fixed_cap/track_step2.json'#/home/parallels/Desktop/LibraForWebRTC/rl_script/traces/varied_cap/4G_500k_expand_2.json
            # trace_path = 'rl_script/traces/fixed_cap/trace_2000k.json'
            # trace_path = 'rl_script/traces/WIRED_900kbs.json'
            logger_enabled = True if i % log_interval == 1 else False
            print("enable log:{}".format(logger_enabled))
            s0 = env.reset(trace_path, i, logger_enabled=logger_enabled)
            # print("s0:"+str(s0))
            simu_start = time.time()
            s0_rec_buffer = np.zeros([rec_s_dim])
            s1_rec_buffer = np.zeros([rec_s_dim])
            s0_rec_buffer[-1 * single_s_dim:] = s0
            terminal = False
            time_step = 0
            while not terminal and time_step < update_interval:
                # print("time step:"+str(time_step))
                # logging.info("s0:"+str(s0_rec_buffer))
                # logging.info("s0 len:"+str(len(s0_rec_buffer)))

                a = agent.get_action(
                    s0_rec_buffer) if not logger_enabled else agent.get_action(
                        s0_rec_buffer, False)
                a = np.squeeze(a)
                # logging.info("a:"+str(a))
                is_valid, s1, r, terminal, _ = env.step(a, w_state)
                # print("s1:"+str(s1))
                if not is_valid:
                    # print("continue")
                    continue
                ep_r += r
                s1_rec_buffer = np.concatenate(
                    (s0_rec_buffer[single_s_dim:], s1))
                # pt = agent.rp_buffer.ptr
                agent.store_experience(s0_rec_buffer, a, r, s1_rec_buffer,
                                       terminal)
                # logging.error(f'buffer raised for {agent.rp_buffer.ptr - pt} with {pt}')
                if time_step % 100 == 0 or terminal:
                    # if agent.rp_buffer.ptr > 400 or agent.rp_buffer.full:
                    # train_start = time.time()
                    agent.train_step()
                    agent.target_update()
                    agent.sess.run(agent.global_step)
                    # logging.error(f'Training finished in {time.time()-train_start} for {time_step}')

                s0_rec_buffer = s1_rec_buffer
                time_step += 1

            print("max bw:"+str(aw.max_bandwidth))
            logging.warn(
                f'Simulation ends in {time.time()-simu_start} for episode {i}')
            if logger_enabled:
                logging.info("drawing")
                draw(trace_path, log_dir_main, i,alg = train_mode)  # 5G_12mbps
                # draw("rl_script/traces/trace_300k.json", log_dir_main,i)
                # env.export_log(i)
            ep_r /= time_step
            
            ewma_r = reward_ewma_coef*ewma_r + (1-reward_ewma_coef)*ep_r
            # 
            reward_record.append(ep_r)
            ewma_reward_record.append(ewma_r)
            plt.cla()

            if i % save_interval == 0:
                saver.save(sess,
                           os.path.join(
                               params.dict['ckptdir'], 'model-{}'.format(
                                   time.strftime("%m%d_%H%M%S", start_time))),
                           global_step=i)
                
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


# def draw(trace_path, log_dir_base, episode_num):

#     draw_metric(os.path.abspath('.'), 'bw', log_dir_base, episode_num,
#                 trace_path)
#     draw_metric(os.path.abspath('.'), 'rtt', log_dir_base, episode_num)

def draw(trace_path, log_dir_base, episode_num,alg):
    
    # draw_metric(os.path.abspath('.'), 'bw', log_dir_base, episode_num, trace_path)
    # draw_metric(os.path.abspath('.'), 'rtt', log_dir_base, episode_num)
    plt.figure(figsize=(6,9))
    plt.subplot(211)
    
    draw_metric(os.path.abspath('.'),'bw',log_dir_base,episode_num,trace_path,alg=alg)
    plt.subplot(212)
    
    draw_metric(os.path.abspath('.'),'rtt',log_dir_base,episode_num,alg=alg)
    save_path = os.path.join(log_dir_base, str(episode_num)+'_merge.jpg')
    print("save path:{}".format(save_path))
    plt.savefig(save_path)
    plt.close()
if __name__ == '__main__':
    parser = OptionParser()

    parser.add_option("-d",
                      "--test_or_train",
                      dest="Test_Or_Train",
                      default=False,
                      type='int',
                      help="#0: test 1:train")
    parser.add_option("-m",
        "--report_interval_ms",
        dest="report_interval_ms",
        default=400,
        help="#if the report_interval_ms is set to 0,\
        the monitor interval is one RTT")
    parser.add_option("-t",
                      "--train_mode",
                      dest="train_mode",
                      default=1,
                      help="1:enable gcc like orca 0:clean-slate")

    (options, args) = parser.parse_args()
    # duration=int(options.Duration)
    test_or_train = bool(options.Test_Or_Train)
    report_interval_ms = int(options.report_interval_ms)
    train_mode = str(options.train_mode).rstrip()
    train(test_or_train, report_interval_ms, train_mode)
