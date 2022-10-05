# -*- coding: utf-8 -*-
# @Time    : 2022/9/23 11:10 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : test_double_dqn.py
# @Software: PyCharm


import torch.optim
import argparse
import gym
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from XuanJing.utils.net.common import MLP
from XuanJing.env.Sampling.sampler import Sampler
from XuanJing.algorithms.modelfree.double_dqn import DoubleDQN
from XuanJing.env.vector.vecbase import VectorEnv
from XuanJing.enhancement.next_state import enhance_next_state
from XuanJing.actor.epsilon_greedy_actor import EpsGreedyActor


def train_loop(envs, actor, algorithm, optimizer, args):
    sampler = Sampler(actor=actor, env=envs, args=args)
    agent = algorithm(actor, optimizer, args)
    for i in range(10):
        with tqdm(total=int(args.num_episodes / 10), desc="Iteration %d" % i) as pbar:
            for i_episode in range(int(args.num_episodes / 10)):
                sampler.sample_step(n_step=1)
                sample_data = sampler.get_sampler_data()
                enhance_sample_data = enhance_next_state(sample_data)
                agent.updata_parameter(enhance_sample_data)
                pbar.update(1)
    plt.plot(sampler.episodes_reward)
    plt.show()
    return None


def dqn_args():
    parser = argparse.ArgumentParser()
    # env
    parser.add_argument("--task", type=str, default="CartPole-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env_num", type=int, default=1)
    # agent
    parser.add_argument("--num_episodes", type=int, default=20000)
    parser.add_argument('--actor_net', type=list, default=[128])
    parser.add_argument("--epsilon", type=float, default=0.01)
    # learn
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--start_learn_buffer_size", type=int, default=500)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--update_target_interval", type=int, default=10)
    parser.add_argument("--device", type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    dqn_args = dqn_args()
    np.random.seed(dqn_args.seed)
    torch.manual_seed(dqn_args.seed)

    envs = VectorEnv([lambda: gym.make('CartPole-v0') for _ in range(dqn_args.env_num)])

    actor_net = MLP(
        input_dim=int(np.prod(envs.observation_space.shape)),
        output_dim=int(np.prod(envs.action_space.n)),
        hidden_sizes=dqn_args.actor_net,
    )

    actor = EpsGreedyActor(actor_net, envs, dqn_args)
    optim = torch.optim.Adam(actor_net.parameters(), lr=dqn_args.lr)

    train_loop(
        envs=envs,
        actor=actor,
        algorithm=DoubleDQN,
        optimizer=optim,
        args=dqn_args
    )
