import argparse
import os
import pprint
import matplotlib.pyplot as plt
import gym
import numpy as np
import torch
import random
from torch.distributions import Independent, Normal
from torch.utils.tensorboard import SummaryWriter
from torch import nn

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from draw import draw_metric

from rtc_env_standard import GymEnv
from aw import FusionAgent, GCCAgent, OrcaAgent
import time



res = []
test_record = {}
episode = 0

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Pendulum-v1')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--step-per-epoch', type=int, default=240000)
    parser.add_argument('--episode-per-collect', type=int, default=1)
    parser.add_argument('--repeat-per-collect', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[256, 256])
    parser.add_argument('--training-num', type=int, default=1)
    parser.add_argument('--test-num', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument('--update-per-step', type=float, default=0.125)
    parser.add_argument('--step-per-collect', type=int, default=8)

    # ppo special
    parser.add_argument('--vf-coef', type=float, default=0.25)
    parser.add_argument('--ent-coef', type=float, default=0.0)
    parser.add_argument('--eps-clip', type=float, default=0.2)
    parser.add_argument('--max-grad-norm', type=float, default=0.5)
    parser.add_argument('--gae-lambda', type=float, default=0.95)
    parser.add_argument('--rew-norm', type=int, default=1)
    parser.add_argument('--dual-clip', type=float, default=None)
    parser.add_argument('--value-clip', type=int, default=1)
    parser.add_argument('--norm-adv', type=int, default=1)
    parser.add_argument('--recompute-adv', type=int, default=0)
    parser.add_argument('--resume', action="store_true")
    parser.add_argument("--save-interval", type=int, default=4)
    args = parser.parse_known_args()[0]
    return args


def test_ppo(args=get_args()):
    # env = gym.make(args.task)
    torch.set_num_threads(1)  # we just need only one thread for NN

    pr_base = os.path.join(os.path.abspath('.'),
                           'rl_script/performance_records/')
    if not os.path.exists(pr_base):
        os.mkdir(pr_base)

    start_time = time.localtime()
    start_time_str = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))

    log_dir_base = os.path.join(pr_base,
                                time.strftime("%m%d_%H%M%S", start_time))

    aw = FusionAgent(0, log_dir_base)
    env = GymEnv(aw, log_dir_base, 0, 0,port_num = 5564)

    if args.task == 'Pendulum-v1':
        env.reward_threshold = 150000
    args.state_shape = env.observation_space.shape or env.observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]
    # you can also use tianshou.env.SubprocVectorEnv
    # train_envs = gym.make(args.task)

    train_envs = DummyVectorEnv(
        [lambda: GymEnv(aw, log_dir_base, 0, 0,port_num = 5565,total_episode=args.step_per_epoch) for _ in range(args.training_num)]
    )
    # test_envs = gym.make(args.task)

    test_envs = DummyVectorEnv(
        [lambda: GymEnv(aw, log_dir_base, 0, 0,port_num = 5566) for _ in range(args.test_num)]
    )

    # seed
    rand_seed  = random.randint(0,10000)
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    
    # model
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes, activation = nn.LeakyReLU,\
                  device=args.device)
    actor = ActorProb(
        net, args.action_shape, max_action=args.max_action, device=args.device
    ).to(args.device)
    critic = Critic(
        Net(args.state_shape, hidden_sizes=args.hidden_sizes, activation = nn.LeakyReLU,\
                   device=args.device),
        device=args.device
    ).to(args.device)
    actor_critic = ActorCritic(actor, critic)
    # orthogonal initialization
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.orthogonal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    # replace DiagGuassian with Independent(Normal) which is equivalent
    # pass *logits to be consistent with policy.forward
    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        max_grad_norm=args.max_grad_norm,
        eps_clip=args.eps_clip,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        reward_normalization=args.rew_norm,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
        dual_clip=args.dual_clip,
        value_clip=args.value_clip,
        gae_lambda=args.gae_lambda,
        action_space=env.action_space
    )
    # collector
    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(args.buffer_size, len(train_envs))
    )
    test_collector = Collector(policy, test_envs)
    # log
    log_path = os.path.join(args.logdir, args.task, 'ppo')
    writer = SummaryWriter(log_path)
    logger = TensorboardLogger(writer, save_interval=args.save_interval)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy.pth'))

    def stop_fn(mean_rewards):
        #可以将这个函数作为hook，每次保存一次mean reward，之后打印出来
        global res
        global episode
        global test_record
        episode += 2000
        res.append(mean_rewards)
        test_record[episode] = mean_rewards
        print("len:{}".format(len(res)))
        print("res:{}".format(res))


        return mean_rewards >= env.reward_threshold

    def save_checkpoint_fn(epoch, env_step, gradient_step):
        # see also: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        torch.save(
            {
                'model': policy.state_dict(),
                'optim': optim.state_dict(),
            }, os.path.join(log_path, 'checkpoint.pth')
        )

    

    if args.resume:
        # load from existing checkpoint
        print(f"Loading agent under {log_path}")
        ckpt_path = os.path.join(log_path, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location=args.device)
            policy.load_state_dict(checkpoint['model'])
            optim.load_state_dict(checkpoint['optim'])
            print("Successfully restore policy and optim.")
        else:
            print("Fail to restore policy and optim.")

    # trainer
    result = onpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        args.epoch,
        args.step_per_epoch,
        args.step_per_collect,
        # args.repeat_per_collect,
        args.test_num,
        args.batch_size,
        episode_per_collect=args.episode_per_collect,
        # update_per_step=args.update_per_collect,
        stop_fn=stop_fn,
        save_fn=save_fn,
        logger=logger
    )
    print("point:result['best_reward']:{}".format(result))
    global res
    print(res)

    with open('rl_script/performance_records/res_data/PPO_{}.txt'.format(start_time_str), 'w') as f:
        for p in test_record.keys():
            f.writelines("{} {}\n".format(p, test_record[p]))

    assert stop_fn(result['best_reward'])

if __name__ == '__main__':
    test_ppo()