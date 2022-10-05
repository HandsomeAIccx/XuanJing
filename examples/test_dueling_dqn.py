# -*- coding: utf-8 -*-
# @Time    : 2022/9/23 11:20 下午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : test_dueling_dqn.py
# @Software: PyCharm


import torch.optim
import argparse
import gym
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from XuanJing.utils.net.common import MLP
from XuanJing.env.Sampling.sampler import Sampler
from XuanJing.algorithms.modelfree.dqn import DQN
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


def dueling_dqn_args():
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


class VAnet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VAnet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc_A = torch.nn.Linear(hidden_dim, output_dim)
        self.fc_v = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        A = self.fc_A(F.relu(self.fc1(x)))
        V = self.fc_v(F.relu(self.fc1(x)))
        Q = V + A - A.mean(1).view(-1, 1)
        return Q


if __name__ == "__main__":
    dueling_dqn_args = dueling_dqn_args()
    np.random.seed(dueling_dqn_args.seed)
    torch.manual_seed(dueling_dqn_args.seed)

    envs = VectorEnv([lambda: gym.make('CartPole-v0') for _ in range(dueling_dqn_args.env_num)])

    # actor_net = MLP(
    #     input_dim=int(np.prod(envs.observation_space.shape)),
    #     output_dim=int(np.prod(envs.action_space.n)),
    #     hidden_sizes=dqn_args.actor_net,
    # )

    actor_net = VAnet(
        input_dim=int(np.prod(envs.observation_space.shape)),
        hidden_dim=dueling_dqn_args.actor_net[0],
        output_dim=int(np.prod(envs.action_space.n))
    )

    actor = EpsGreedyActor(actor_net, envs, dueling_dqn_args)
    optim = torch.optim.Adam(actor_net.parameters(), lr=dueling_dqn_args.lr)

    train_loop(
        envs=envs,
        actor=actor,
        algorithm=DQN,
        optimizer=optim,
        args=dueling_dqn_args
    )

