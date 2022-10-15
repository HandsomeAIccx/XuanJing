# -*- coding: utf-8 -*-
# @Time    : 2022/10/4 11:41 上午
# @Author  : Zhiqiang He
# @Email   : tinyzqh@163.com
# @File    : test_pointer_net.py
# @Software: PyCharm


import argparse
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from XuanJing.algorithms.offline.combinatorial_rl import CombinatorialRL
from XuanJing.utils.net.pointer_net import PointerNet
from XuanJing.gamecore.co_gc.tsp_env import TspEnv


def train_loop(env, actor_net, algorithm, optimizer, args):
    agent = algorithm(actor=actor_net, optim=optimizer, args=args)
    train_tour = []

    for _ in tqdm(range(800)):
        state = env.reset()
        return_reward = agent.updata_parameter(train_data=torch.tensor(state))
        train_tour.append(return_reward.mean().item())

    plt.plot(train_tour)
    plt.show()


def pointer_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=100000)
    parser.add_argument("--city_num", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--max_grad_norm", type=float, default=2.)
    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":

    args = pointer_args()

    actor_net = PointerNet(args)
    actor_optim = optim.Adam(actor_net.parameters(), lr=args.learning_rate)
    env = TspEnv(city_num=20, batch_size=64)
    train_loop(
        env=env,
        actor_net=actor_net,
        algorithm=CombinatorialRL,
        optimizer=actor_optim,
        args=args
    )

